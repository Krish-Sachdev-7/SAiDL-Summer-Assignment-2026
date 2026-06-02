"""RL training entry point."""
import random
import atexit
import signal
from copy import deepcopy
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from env_wrappers import make_env
    from networks import MLPActor, MLPCritic, TransformerActor
    from replay_buffer import StandardReplayBuffer, SequenceReplayBuffer
    from td3 import TD3
    from reward_model import RewardModel, PreferenceLearner
    from xlstm_actor import xLSTMActor
else:
    from .env_wrappers import make_env
    from .networks import MLPActor, MLPCritic, TransformerActor
    from .replay_buffer import StandardReplayBuffer, SequenceReplayBuffer
    from .td3 import TD3
    from .reward_model import RewardModel, PreferenceLearner
    from .xlstm_actor import xLSTMActor

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from shared.doc_pipeline import update_experiment_docs


def _safe_name(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(text)).strip("._-")
    return safe or "run"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_cpu_rng_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = [_as_cpu_rng_tensor(s) for s in state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states[: torch.cuda.device_count()])


def _as_cpu_rng_tensor(value) -> torch.ByteTensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8).cpu()


def _serialize_replay_buffer(replay) -> dict:
    state = {"type": type(replay).__name__}
    for key, value in replay.__dict__.items():
        state[key] = value
    return state


def _restore_replay_buffer(replay, state: dict | None) -> None:
    if not state:
        return
    saved_type = state.get("type")
    if saved_type is not None and saved_type != type(replay).__name__:
        raise ValueError(f"Replay buffer type mismatch: checkpoint={saved_type}, current={type(replay).__name__}")
    for key, value in state.items():
        if key == "type":
            continue
        setattr(replay, key, value)


def maybe_init_wandb(cfg):
    if not bool(cfg.logging.wandb.enable):
        return None
    import wandb

    return wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        group=cfg.logging.wandb.group,
        tags=list(cfg.logging.wandb.tags),
        name=cfg.experiment.name,
    )


def compute_attention_metrics(attn_layers) -> dict[str, float]:
    """Summarise current-token attention over the context window."""
    if not attn_layers:
        return {}

    vectors = []
    entropies = []
    max_weights = []
    effective_context = []
    for attn in attn_layers:
        if attn is None:
            continue
        probs = attn.detach().float()
        if probs.dim() != 4:
            continue
        last_query = probs[:, :, -1, :]
        last_query = last_query.clamp_min(1e-8)
        last_query = last_query / last_query.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(last_query * last_query.log()).sum(dim=-1)
        vectors.append(last_query)
        entropies.append(entropy.reshape(-1))
        max_weights.append(last_query.max(dim=-1).values.reshape(-1))
        effective_context.append(torch.exp(entropy).reshape(-1))

    if not vectors:
        return {}

    mean_vector = torch.cat([v.reshape(-1, v.size(-1)) for v in vectors], dim=0).mean(dim=0)
    entropy_values = torch.cat(entropies, dim=0)
    max_values = torch.cat(max_weights, dim=0)
    effective_values = torch.cat(effective_context, dim=0)

    metrics = {
        "attention/entropy_mean": float(entropy_values.mean().item()),
        "attention/entropy_std": float(entropy_values.std(unbiased=False).item()),
        "attention/max_weight_mean": float(max_values.mean().item()),
        "attention/effective_context_mean": float(effective_values.mean().item()),
    }
    for lag, value in enumerate(torch.flip(mean_vector, dims=[0])):
        metrics[f"attention/lag_{lag}"] = float(value.item())
    return metrics


class RunningObservationNormalizer:
    """Online per-dimension observation standardizer."""
    def __init__(self, obs_dim: int, eps: float = 1e-4, clip: float = 10.0):
        self.obs_dim = int(obs_dim)
        self.eps = float(eps)
        self.clip = float(clip)
        self.count = 0.0
        self.mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.var = np.ones(self.obs_dim, dtype=np.float64)

    def update(self, obs) -> None:
        batch = np.asarray(obs, dtype=np.float64).reshape(-1, self.obs_dim)
        if batch.size == 0:
            return

        batch_count = float(batch.shape[0])
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        if self.count == 0.0:
            self.mean = batch_mean
            self.var = np.maximum(batch_var, self.eps)
            self.count = batch_count
            return

        delta = batch_mean - self.mean
        total = self.count + batch_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.mean = self.mean + delta * batch_count / total
        self.var = np.maximum(m2 / total, self.eps)
        self.count = total

    def normalize(self, obs) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if self.count == 0.0:
            return arr.astype(np.float32, copy=False)
        normed = (arr - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + self.eps)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32, copy=False)

    def state_dict(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "eps": self.eps,
            "clip": self.clip,
            "count": self.count,
            "mean": self.mean,
            "var": self.var,
        }

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.count = float(state.get("count", 0.0))
        self.mean = np.asarray(state.get("mean", self.mean), dtype=np.float64)
        self.var = np.asarray(state.get("var", self.var), dtype=np.float64)
        self.eps = float(state.get("eps", self.eps))
        self.clip = float(state.get("clip", self.clip))


def _obs_norm_enabled(cfg) -> bool:
    return bool(OmegaConf.select(cfg, "agent.actor.obs_norm", default=False))


def _normalizer_state(normalizer: RunningObservationNormalizer | None) -> dict | None:
    return normalizer.state_dict() if normalizer is not None else None


def _normalize_obs(normalizer: RunningObservationNormalizer | None, obs) -> np.ndarray:
    if normalizer is None:
        return np.asarray(obs, dtype=np.float32)
    return normalizer.normalize(obs)


def _maybe_update_best_eval(
    eval_return: float,
    step: int,
    best_eval_return: float,
    best_eval_step: int,
) -> tuple[float, int, bool]:
    if float(eval_return) > float(best_eval_return):
        return float(eval_return), int(step), True
    return float(best_eval_return), int(best_eval_step), False


def evaluate_agent(cfg, agent, device, n_episodes: int, obs_normalizer: RunningObservationNormalizer | None = None):
    env = make_env(cfg)
    rewards = []

    context = int(getattr(cfg.agent.actor, "context_length", 1))
    act_dim = int(env.action_space.shape[0])

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0

        obs_hist = [np.zeros_like(obs, dtype=np.float32) for _ in range(max(context - 1, 0))]
        act_hist = [np.zeros(act_dim, dtype=np.float32) for _ in range(max(context - 1, 0))]

        while not (done or truncated):
            policy_obs = _normalize_obs(obs_normalizer, obs)
            if cfg.agent.type in {"transformer", "xlstm"}:
                obs_hist.append(policy_obs)
                obs_hist = obs_hist[-context:]
                act_input = act_hist[-(context - 1):] + [np.zeros(act_dim, dtype=np.float32)]
                action = agent.select_action(
                    {
                        "obs_seq": np.stack(obs_hist, axis=0),
                        "act_seq": np.stack(act_input, axis=0),
                    },
                    noise=0.0,
                )
                act_hist.append(np.asarray(action, dtype=np.float32))
                act_hist = act_hist[-(context - 1):]
            else:
                action = agent.select_action(policy_obs, noise=0.0)

            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += float(reward)

        rewards.append(ep_reward)

    env.close()
    return float(np.mean(rewards))


def evaluate_agent_with_attention(
    cfg,
    agent,
    device,
    n_episodes: int,
    obs_normalizer: RunningObservationNormalizer | None = None,
) -> dict[str, float]:
    """Evaluate a Transformer actor and collect attention summaries."""
    env = make_env(cfg)
    rewards = []
    per_decision_metrics = []

    context = int(getattr(cfg.agent.actor, "context_length", 1))
    act_dim = int(env.action_space.shape[0])

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0

        obs_hist = [np.zeros_like(obs, dtype=np.float32) for _ in range(max(context - 1, 0))]
        act_hist = [np.zeros(act_dim, dtype=np.float32) for _ in range(max(context - 1, 0))]

        while not (done or truncated):
            policy_obs = _normalize_obs(obs_normalizer, obs)
            obs_hist.append(policy_obs)
            obs_hist = obs_hist[-context:]
            act_input = act_hist[-(context - 1):] + [np.zeros(act_dim, dtype=np.float32)]
            action = agent.select_action(
                {
                    "obs_seq": np.stack(obs_hist, axis=0),
                    "act_seq": np.stack(act_input, axis=0),
                },
                noise=0.0,
            )
            metrics = compute_attention_metrics(getattr(agent.actor, "last_attn_weights", []))
            if metrics:
                per_decision_metrics.append(metrics)
            act_hist.append(np.asarray(action, dtype=np.float32))
            act_hist = act_hist[-(context - 1):]

            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += float(reward)

        rewards.append(ep_reward)

    env.close()
    eval_metrics = {
        "eval/return": float(np.mean(rewards)),
        "eval/return_std": float(np.std(rewards)),
    }
    if per_decision_metrics:
        keys = sorted({key for item in per_decision_metrics for key in item})
        for key in keys:
            values = [item[key] for item in per_decision_metrics if key in item]
            if key == "attention/entropy_std":
                continue
            eval_metrics[key] = float(np.mean(values))
        entropy_values = [item["attention/entropy_mean"] for item in per_decision_metrics if "attention/entropy_mean" in item]
        if entropy_values:
            eval_metrics["attention/entropy_std"] = float(np.std(entropy_values))
    return eval_metrics


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training loop."""
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = maybe_init_wandb(cfg)

    env = make_env(cfg)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])
    obs_normalizer = RunningObservationNormalizer(obs_dim) if _obs_norm_enabled(cfg) else None

    if cfg.agent.type == "mlp":
        actor = MLPActor(obs_dim, act_dim, list(cfg.agent.actor.hidden_dims), max_action=max_action).to(device)
    elif cfg.agent.type == "transformer":
        pos_type = "learned"
        if "pos_encoding" in cfg and "type" in cfg.pos_encoding:
            pos_type = str(cfg.pos_encoding.type)
        actor = TransformerActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            embed_dim=int(cfg.agent.actor.embed_dim),
            n_layers=int(cfg.agent.actor.n_layers),
            n_heads=int(cfg.agent.actor.n_heads),
            context_length=int(cfg.agent.actor.context_length),
            max_action=max_action,
            pos_encoding=pos_type,
        ).to(device)
    elif cfg.agent.type == "xlstm":
        actor = xLSTMActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            embed_dim=int(cfg.agent.actor.embed_dim),
            n_slstm_layers=int(cfg.agent.actor.n_slstm_layers),
            n_mlstm_layers=int(cfg.agent.actor.n_mlstm_layers),
            max_action=max_action,
        ).to(device)
    else:
        raise ValueError(f"Unsupported agent type: {cfg.agent.type}")

    critic1 = MLPCritic(obs_dim, act_dim, list(cfg.agent.critic.hidden_dims)).to(device)
    critic2 = MLPCritic(obs_dim, act_dim, list(cfg.agent.critic.hidden_dims)).to(device)
    agent = TD3(actor, critic1, critic2, cfg)

    rl_resume = str(
        OmegaConf.select(
            cfg,
            "resume",
            default=OmegaConf.select(cfg, "experiment.resume", default=""),
        )
        or ""
    )
    rl_start_step = 1
    resume_ckpt = None
    if rl_resume:
        resume_ckpt = torch.load(rl_resume, map_location=device, weights_only=False)
        actor.load_state_dict(resume_ckpt["actor"])
        critic1.load_state_dict(resume_ckpt["critic1"])
        critic2.load_state_dict(resume_ckpt["critic2"])
        agent.actor_target.load_state_dict(resume_ckpt["actor_target"])
        agent.critic1_target.load_state_dict(resume_ckpt["critic1_target"])
        agent.critic2_target.load_state_dict(resume_ckpt["critic2_target"])
        agent.actor_optim.load_state_dict(resume_ckpt["actor_optim"])
        agent.critic1_optim.load_state_dict(resume_ckpt["critic1_optim"])
        agent.critic2_optim.load_state_dict(resume_ckpt["critic2_optim"])
        rl_start_step = int(resume_ckpt["step"]) + 1
        _restore_rng_state(resume_ckpt.get("rng_state"))
        if obs_normalizer is not None:
            obs_normalizer.load_state_dict(resume_ckpt.get("obs_normalizer"))
        print(f"Resumed RL from {rl_resume} at step {rl_start_step}")

    if cfg.agent.type in {"transformer", "xlstm"}:
        replay = SequenceReplayBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            context_length=int(cfg.agent.actor.context_length),
            max_size=int(cfg.agent.buffer_size),
        )
    else:
        replay = StandardReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=int(cfg.agent.buffer_size))

    if resume_ckpt is not None:
        _restore_replay_buffer(replay, resume_ckpt.get("replay"))

    step = rl_start_step - 1
    final_eval_return = float(resume_ckpt.get("final_eval_return", float("nan"))) if resume_ckpt else float("nan")
    best_eval_return = float(resume_ckpt.get("best_eval_return", float("-inf"))) if resume_ckpt else float("-inf")
    best_eval_step = int(resume_ckpt.get("best_eval_step", 0)) if resume_ckpt else 0
    checkpoint_saved = False
    experiment_name = str(cfg.experiment.name)
    checkpoint_dir = RL_ROOT / "checkpoints" / _safe_name(experiment_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path: Path) -> None:
        nonlocal checkpoint_saved
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment_name": experiment_name,
                "agent_type": str(cfg.agent.type),
                "env_name": str(cfg.env.name),
                "seed": int(cfg.seed),
                "total_steps": int(cfg.total_steps),
                "context_length": int(getattr(cfg.agent.actor, "context_length", 1)),
                "actor": actor.state_dict(),
                "critic1": critic1.state_dict(),
                "critic2": critic2.state_dict(),
                "actor_target": agent.actor_target.state_dict(),
                "critic1_target": agent.critic1_target.state_dict(),
                "critic2_target": agent.critic2_target.state_dict(),
                "actor_optim": agent.actor_optim.state_dict(),
                "critic1_optim": agent.critic1_optim.state_dict(),
                "critic2_optim": agent.critic2_optim.state_dict(),
                "replay": _serialize_replay_buffer(replay),
                "obs_normalizer": _normalizer_state(obs_normalizer),
                "rng_state": _capture_rng_state(),
                "step": int(step),
                "final_eval_return": float(final_eval_return),
                "best_eval_return": float(best_eval_return),
                "best_eval_step": int(best_eval_step),
            },
            path,
        )
        checkpoint_saved = True

    def save_latest_checkpoint() -> None:
        save_checkpoint(checkpoint_dir / "ckpt_latest_rl.pt")
        # Keep the old checkpoint name for manual recovery.
        save_checkpoint(RL_ROOT / "ckpt_latest_rl.pt")

    def save_final_checkpoint() -> None:
        save_checkpoint(checkpoint_dir / "final_rl_model.pt")
        save_checkpoint(RL_ROOT / "final_rl_model.pt")

    def save_best_checkpoint() -> None:
        save_checkpoint(checkpoint_dir / "best_rl_model.pt")
        save_checkpoint(RL_ROOT / "best_rl_model.pt")

    def _save_on_exit(*_args):
        save_latest_checkpoint()

    atexit.register(_save_on_exit)

    def _handle_signal(signum, frame):
        save_latest_checkpoint()
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    pref_learner = None
    if str(cfg.reward.type) == "rlhf":
        rm = RewardModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=list(cfg.reward.reward_model_hidden),
        ).to(device)
        pref_learner = PreferenceLearner(rm, cfg)

    obs, _ = env.reset()
    if obs_normalizer is not None:
        obs_normalizer.update(obs)
    context = int(getattr(cfg.agent.actor, "context_length", 1))
    obs_hist = [np.zeros(obs_dim, dtype=np.float32) for _ in range(max(context - 1, 0))]
    act_hist = [np.zeros(act_dim, dtype=np.float32) for _ in range(max(context - 1, 0))]

    metrics = {"critic1_loss": float("nan"), "critic2_loss": float("nan"), "actor_loss": float("nan")}
    for step in range(rl_start_step, int(cfg.total_steps) + 1):
        policy_obs = _normalize_obs(obs_normalizer, obs)
        if step <= int(cfg.agent.start_steps):
            action = env.action_space.sample()
        else:
            if cfg.agent.type in {"transformer", "xlstm"}:
                obs_hist.append(policy_obs)
                obs_hist = obs_hist[-context:]
                act_in = act_hist[-(context - 1):] + [np.zeros(act_dim, dtype=np.float32)]
                action = agent.select_action(
                    {
                        "obs_seq": np.stack(obs_hist, axis=0),
                        "act_seq": np.stack(act_in, axis=0),
                    },
                    noise=float(cfg.agent.exploration_noise),
                )
            else:
                action = agent.select_action(policy_obs, noise=float(cfg.agent.exploration_noise))

        next_obs, reward, done, truncated, _ = env.step(action)
        terminal = bool(done or truncated)
        if obs_normalizer is not None:
            obs_normalizer.update(next_obs)
        next_policy_obs = _normalize_obs(obs_normalizer, next_obs)

        reward_to_store = float(reward)
        if pref_learner is not None and step > int(cfg.agent.start_steps):
            with torch.no_grad():
                r_pred = pref_learner.reward_model(
                    torch.tensor(policy_obs, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0),
                )
            reward_to_store = float(r_pred.item())

        replay.add(policy_obs, action, reward_to_store, next_policy_obs, terminal)

        if cfg.agent.type in {"transformer", "xlstm"}:
            act_hist.append(np.asarray(action, dtype=np.float32))
            act_hist = act_hist[-(context - 1):]

        if step > int(cfg.agent.start_steps):
            metrics = agent.update(replay, batch_size=int(cfg.agent.batch_size), step=step)
        else:
            metrics = {"critic1_loss": float("nan"), "critic2_loss": float("nan"), "actor_loss": float("nan")}

        if pref_learner is not None and step % int(cfg.reward.reward_update_interval) == 0:
            pref_learner.collect_preferences(replay, int(cfg.reward.num_comparisons))
            rm_metrics = pref_learner.train_reward_model()
            metrics.update(rm_metrics)

        if terminal:
            obs, _ = env.reset()
            if obs_normalizer is not None:
                obs_normalizer.update(obs)
            obs_hist = [np.zeros(obs_dim, dtype=np.float32) for _ in range(max(context - 1, 0))]
            act_hist = [np.zeros(act_dim, dtype=np.float32) for _ in range(max(context - 1, 0))]
        else:
            obs = next_obs

        rl_ckpt_interval = int(OmegaConf.select(cfg, "checkpoint_interval", default=50000))
        if rl_ckpt_interval > 0 and step % rl_ckpt_interval == 0:
            save_latest_checkpoint()

        if step % int(cfg.eval_interval) == 0:
            if cfg.agent.type == "transformer":
                eval_metrics = evaluate_agent_with_attention(
                    cfg,
                    agent,
                    device,
                    n_episodes=int(cfg.eval_episodes),
                    obs_normalizer=obs_normalizer,
                )
                eval_return = float(eval_metrics["eval/return"])
            else:
                eval_return = evaluate_agent(
                    cfg,
                    agent,
                    device,
                    n_episodes=int(cfg.eval_episodes),
                    obs_normalizer=obs_normalizer,
                )
                eval_metrics = {
                    "eval/return": float(eval_return),
                    "eval/return_std": 0.0,
                }
            final_eval_return = float(eval_return)
            best_eval_return, best_eval_step, improved_best = _maybe_update_best_eval(
                eval_return=eval_return,
                step=step,
                best_eval_return=best_eval_return,
                best_eval_step=best_eval_step,
            )
            if improved_best:
                save_best_checkpoint()
            if run is not None:
                import wandb

                wandb.log(
                    {
                        "step": step,
                        **{k: float(v) for k, v in eval_metrics.items()},
                        "eval/best_return": float(best_eval_return),
                        "eval/best_step": float(best_eval_step),
                        "eval/final_minus_best": float(final_eval_return - best_eval_return),
                        "obs_norm/enabled": float(obs_normalizer is not None),
                        "obs_norm/count": float(obs_normalizer.count if obs_normalizer is not None else 0.0),
                        **{k: float(v) for k, v in metrics.items()},
                    }
                )
            print(
                f"step={step} eval_return={eval_return:.2f} "
                f"best_eval_return={best_eval_return:.2f} best_step={best_eval_step} metrics={metrics}"
            )

    save_final_checkpoint()

    artifacts = update_experiment_docs(
        track="rl",
        experiment_name=str(cfg.experiment.name),
        cfg_dict={
            "agent": cfg.agent,
            "env": cfg.env,
            "reward": cfg.reward,
            "seed": cfg.seed,
            "total_steps": cfg.total_steps,
        },
        metrics={
            "final_eval_return": round(final_eval_return, 6),
            "best_eval_return": round(best_eval_return, 6),
            "best_eval_step": best_eval_step,
            "final_minus_best_eval_return": round(final_eval_return - best_eval_return, 6),
            "obs_norm_enabled": obs_normalizer is not None,
            "obs_norm_count": round(float(obs_normalizer.count if obs_normalizer is not None else 0.0), 6),
            "last_critic1_loss": round(float(metrics.get("critic1_loss", float("nan"))), 6),
            "last_critic2_loss": round(float(metrics.get("critic2_loss", float("nan"))), 6),
            "last_actor_loss": round(float(metrics.get("actor_loss", float("nan"))), 6),
        },
        run_output_dir=str(Path.cwd()),
        repo_root=REPO_ROOT,
    )
    print(f"Updated docs: {artifacts.markdown_path}")
    print(f"Updated docs: {artifacts.include_index_path}")

    env.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
