"""Algorithm Distillation bonus code.

This module trains a causal policy transformer on cross-episode learning
histories. A history file stores multiple episodes from one source-policy
rollout stream, which lets the model condition on what happened in earlier
episodes without taking gradient steps at evaluation time.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from env_wrappers import (
        CombinedPOMDPWrapper,
        DelayedRewardWrapper,
        HiddenVelocityWrapper,
        NoisyObservationWrapper,
    )
    from networks import MLPActor, TransformerActor
else:
    from .env_wrappers import (
        CombinedPOMDPWrapper,
        DelayedRewardWrapper,
        HiddenVelocityWrapper,
        NoisyObservationWrapper,
    )
    from .networks import MLPActor, TransformerActor


class ToyMemoryEnv(gym.Env):
    """Small continuous-control toy env for fast AD smoke tests."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 8, seed: int = 0):
        super().__init__()
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._hidden = 1.0
        self._step = 0
        self._prev_reward = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._hidden = float(self.rng.choice([-1.0, 1.0]))
        self._step = 0
        self._prev_reward = 0.0
        self._prev_action = np.zeros(2, dtype=np.float32)
        return self._obs(), {}

    def expert_action(self, obs=None) -> np.ndarray:
        del obs
        return np.asarray([self._hidden, -self._hidden], dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        target = self.expert_action()
        reward = -float(np.mean((action - target) ** 2))
        self._prev_reward = reward
        self._prev_action = action.copy()
        self._step += 1
        terminated = False
        truncated = self._step >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        return np.asarray(
            [
                self._step / max(1, self.max_steps),
                self._prev_reward,
                self._prev_action[0],
            ],
            dtype=np.float32,
        )


class ADTransformer(nn.Module):
    """Causal transformer for action prediction from learning histories."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        embed_dim: int,
        n_layers: int,
        n_heads: int,
        max_history_len: int,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.embed_dim = int(embed_dim)
        self.max_history_len = int(max_history_len)

        in_dim = self.obs_dim + self.act_dim + 1 + 1
        self.in_proj = nn.Linear(in_dim, self.embed_dim)
        self.pos_emb = nn.Embedding(self.max_history_len, self.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(n_heads),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.ln = nn.LayerNorm(self.embed_dim)
        self.action_head = nn.Linear(self.embed_dim, self.act_dim)

    def forward(self, obs_seq, prev_act_seq, reward_seq, done_seq):
        _, seq_len, _ = obs_seq.shape
        if seq_len > self.max_history_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_history_len={self.max_history_len}")

        x = torch.cat([obs_seq, prev_act_seq, reward_seq, done_seq], dim=-1)
        x = self.in_proj(x)
        pos = torch.arange(seq_len, device=x.device)
        x = x + self.pos_emb(pos)[None, :, :]

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.encoder(x, mask=causal_mask)
        x = self.ln(x)
        return torch.tanh(self.action_head(x))


class LearningHistoryDataset(Dataset):
    """Sliding windows over full cross-episode AD histories."""

    def __init__(self, history_dir: str | Path, context_length: int):
        self.history_dir = Path(history_dir)
        self.context_length = int(context_length)
        self.files = sorted(self.history_dir.glob("history_*.pt"))
        if not self.files:
            self.files = sorted(self.history_dir.glob("*.pt"))

        self.samples: list[dict[str, torch.Tensor]] = []
        for file_path in self.files:
            history = torch.load(file_path, map_location="cpu", weights_only=False)
            obs = _as_float_2d(history["obs"])
            act = _as_float_2d(history["act"])
            rew = _as_float_2d(history["rew"])
            done = _as_float_2d(history["done"])
            episode_id = torch.as_tensor(
                history.get("episode_id", _episode_ids_from_done(done)),
                dtype=torch.long,
            ).view(-1)

            if obs.size(0) < self.context_length:
                continue

            prev_act = _shift_with_episode_reset(act, done)
            prev_rew = _shift_with_episode_reset(rew, done)
            prev_done = torch.zeros_like(done)
            prev_done[1:] = done[:-1]

            for start in range(0, obs.size(0) - self.context_length + 1):
                end = start + self.context_length
                self.samples.append(
                    {
                        "obs_seq": obs[start:end],
                        "prev_act_seq": prev_act[start:end],
                        "reward_seq": prev_rew[start:end],
                        "done_seq": prev_done[start:end],
                        "target_act_seq": act[start:end],
                        "loss_mask": torch.ones(self.context_length, 1, dtype=torch.float32),
                        "episode_id_seq": episode_id[start:end],
                    }
                )

        if not self.samples:
            raise ValueError(
                f"No AD samples in {self.history_dir}; need at least one history with "
                f"length >= context_length={self.context_length}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {key: value.clone() for key, value in self.samples[idx].items()}


def _as_float_2d(value) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    return tensor


def _episode_ids_from_done(done: torch.Tensor) -> torch.Tensor:
    done_flat = done.view(-1).float()
    ids = torch.zeros(done_flat.numel(), dtype=torch.long)
    current = 0
    for i in range(done_flat.numel()):
        ids[i] = current
        if done_flat[i].item() > 0.5:
            current += 1
    return ids


def _shift_with_episode_reset(values: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    shifted = torch.zeros_like(values)
    if values.size(0) <= 1:
        return shifted
    shifted[1:] = values[:-1]
    reset_mask = done[:-1].view(-1) > 0.5
    if reset_mask.any():
        shifted[1:][reset_mask] = 0.0
    return shifted


def make_ad_env(env_key: str, seed: int = 0, max_steps_per_episode: int | None = None):
    """Build an AD collection/evaluation environment from a runner alias."""
    key = str(env_key)
    if key == "toy_memory":
        return ToyMemoryEnv(max_steps=max_steps_per_episode or 8, seed=seed)

    if key in {"hopper_full", "Hopper-v5"}:
        env = gym.make("Hopper-v5")
    elif key == "hopper_hidden_vel":
        env = HiddenVelocityWrapper(gym.make("Hopper-v5"))
    elif key == "hopper_noisy":
        env = NoisyObservationWrapper(gym.make("Hopper-v5"), sigma=0.1)
    elif key == "hopper_delayed":
        env = DelayedRewardWrapper(gym.make("Hopper-v5"), delay_k=10)
    elif key == "hopper_combined_pomdp":
        env = CombinedPOMDPWrapper(gym.make("Hopper-v5"), sigma=0.1, delay_k=10)
    else:
        env = gym.make(key)

    if max_steps_per_episode is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(max_steps_per_episode))
    return env


def collect_learning_histories(
    env_name: str,
    histories: int,
    episodes_per_history: int,
    save_dir: str | Path,
    seed: int = 42,
    max_steps_per_episode: int | None = None,
    source_checkpoint: str | Path | None = None,
    source_noise: float = 0.05,
) -> dict[str, object]:
    """Collect full cross-episode histories from a source policy."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("history_*.pt"):
        old.unlink()

    env = make_ad_env(env_name, seed=seed, max_steps_per_episode=max_steps_per_episode)
    rng = np.random.default_rng(seed)
    source_policy = _load_source_policy(source_checkpoint, env) if source_checkpoint else None
    source_type = "td3_checkpoint" if source_policy is not None else "expert_or_exploratory"

    saved = 0
    for history_idx in range(int(histories)):
        obs_hist: list[np.ndarray] = []
        act_hist: list[np.ndarray] = []
        rew_hist: list[list[float]] = []
        done_hist: list[list[float]] = []
        episode_ids: list[int] = []

        for episode in range(int(episodes_per_history)):
            reset_seed = int(seed + history_idx * 1000 + episode)
            obs, _ = env.reset(seed=reset_seed)
            done = False
            truncated = False
            step_count = 0
            policy_state = _SourcePolicyState(env.action_space.shape[0])

            while not (done or truncated):
                if source_policy is not None:
                    action = source_policy.select_action(obs, policy_state, noise=source_noise)
                elif hasattr(env.unwrapped, "expert_action"):
                    action = env.unwrapped.expert_action(obs)
                elif hasattr(env, "expert_action"):
                    action = env.expert_action(obs)
                else:
                    action = env.action_space.sample()
                    action = np.asarray(action, dtype=np.float32)
                    action += rng.normal(0.0, source_noise, size=action.shape).astype(np.float32)

                action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
                next_obs, reward, done, truncated, _ = env.step(action)
                terminal = bool(done or truncated)

                obs_hist.append(np.asarray(obs, dtype=np.float32))
                act_hist.append(action)
                rew_hist.append([float(reward)])
                done_hist.append([float(terminal)])
                episode_ids.append(episode)

                policy_state.record(obs, action)
                obs = next_obs
                step_count += 1
                if max_steps_per_episode is not None and step_count >= int(max_steps_per_episode):
                    break

        if obs_hist:
            data = {
                "obs": torch.tensor(np.asarray(obs_hist), dtype=torch.float32),
                "act": torch.tensor(np.asarray(act_hist), dtype=torch.float32),
                "rew": torch.tensor(np.asarray(rew_hist), dtype=torch.float32),
                "done": torch.tensor(np.asarray(done_hist), dtype=torch.float32),
                "episode_id": torch.tensor(np.asarray(episode_ids), dtype=torch.long),
                "env": str(env_name),
                "source_policy": source_type,
                "episodes_per_history": int(episodes_per_history),
            }
            torch.save(data, out / f"history_{saved:06d}.pt")
            saved += 1

    env.close()
    metrics = {
        "saved_histories": saved,
        "output_dir": str(out),
        "env": str(env_name),
        "source_policy": source_type,
    }
    return metrics


class _SourcePolicyState:
    def __init__(self, act_dim: int, context_length: int = 8):
        self.act_dim = int(act_dim)
        self.context_length = int(context_length)
        self.obs_hist: list[np.ndarray] = []
        self.act_hist: list[np.ndarray] = []

    def record(self, obs, action) -> None:
        self.obs_hist.append(np.asarray(obs, dtype=np.float32))
        self.act_hist.append(np.asarray(action, dtype=np.float32))
        self.obs_hist = self.obs_hist[-self.context_length :]
        self.act_hist = self.act_hist[-max(1, self.context_length - 1) :]


class _FrozenObservationNormalizer:
    """Observation normalizer restored from a TD3 checkpoint for teacher rollout."""

    def __init__(self, state: dict | None):
        self.enabled = bool(state) and float(state.get("count", 0.0)) > 0.0
        self.count = float(state.get("count", 0.0)) if state else 0.0
        self.mean = np.asarray(state.get("mean", []), dtype=np.float32) if state else np.asarray([], dtype=np.float32)
        self.var = np.asarray(state.get("var", []), dtype=np.float32) if state else np.asarray([], dtype=np.float32)
        self.eps = float(state.get("eps", 1e-4)) if state else 1e-4
        self.clip = float(state.get("clip", 10.0)) if state else 10.0

    def normalize(self, obs) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if not self.enabled or self.mean.shape != arr.shape or self.var.shape != arr.shape:
            return arr
        normed = (arr - self.mean) / np.sqrt(self.var + self.eps)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32, copy=False)


class _TD3ActorPolicy:
    def __init__(
        self,
        actor: nn.Module,
        agent_type: str,
        context_length: int,
        device: torch.device,
        obs_normalizer_state: dict | None = None,
    ):
        self.actor = actor
        self.agent_type = str(agent_type)
        self.context_length = int(context_length)
        self.device = device
        self.max_action = float(getattr(actor, "max_action", 1.0))
        self.obs_normalizer = _FrozenObservationNormalizer(obs_normalizer_state)

    def select_action(self, obs, state: _SourcePolicyState, noise: float = 0.0) -> np.ndarray:
        self.actor.eval()
        policy_obs = self.obs_normalizer.normalize(obs)
        with torch.no_grad():
            if self.agent_type == "transformer":
                obs_hist = [self.obs_normalizer.normalize(x) for x in state.obs_hist] + [policy_obs]
                obs_hist = obs_hist[-self.context_length :]
                act_hist = state.act_hist[-max(0, self.context_length - 1) :]
                act_input = act_hist + [np.zeros(state.act_dim, dtype=np.float32)]
                obs_seq = _left_pad_sequence(obs_hist, self.context_length, policy_obs.shape[0])
                act_seq = _left_pad_sequence(act_input, self.context_length, state.act_dim)
                obs_t = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                act_t = torch.as_tensor(act_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.actor(obs_t, act_t).squeeze(0)
            else:
                obs_t = torch.as_tensor(policy_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.actor(obs_t).squeeze(0)
        action_np = action.cpu().numpy()
        if noise > 0:
            action_np = action_np + np.random.normal(0.0, noise, size=action_np.shape)
        return np.clip(action_np, -self.max_action, self.max_action).astype(np.float32)


def _left_pad_sequence(items: Iterable[np.ndarray], length: int, dim: int) -> np.ndarray:
    arr = np.zeros((int(length), int(dim)), dtype=np.float32)
    items = list(items)[-int(length) :]
    if not items:
        return arr
    stacked = np.stack(items, axis=0).astype(np.float32)
    arr[-stacked.shape[0] :] = stacked
    return arr


def _load_source_policy(path: str | Path | None, env) -> _TD3ActorPolicy | None:
    if not path:
        return None
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_state = ckpt.get("actor")
    if not isinstance(actor_state, dict):
        return None

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])
    agent_type = str(ckpt.get("agent_type", "transformer" if "input_proj.weight" in actor_state else "mlp"))

    if agent_type == "transformer" or "input_proj.weight" in actor_state:
        embed_dim = int(actor_state["input_proj.weight"].shape[0])
        n_layers = 1 + max(
            int(key.split(".")[1])
            for key in actor_state
            if key.startswith("blocks.") and key.split(".")[1].isdigit()
        )
        context_length = int(ckpt.get("context_length", actor_state.get("pos_emb.weight").shape[0]))
        actor = TransformerActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=4,
            context_length=context_length,
            max_action=max_action,
            pos_encoding="learned",
        ).to(device)
        actor.load_state_dict(actor_state, strict=False)
        return _TD3ActorPolicy(actor, "transformer", context_length, device, ckpt.get("obs_normalizer"))

    hidden_dims = []
    layer_idx = 0
    while f"backbone.{layer_idx}.weight" in actor_state:
        hidden_dims.append(int(actor_state[f"backbone.{layer_idx}.weight"].shape[0]))
        layer_idx += 2
    if not hidden_dims:
        hidden_dims = [256, 256]
    actor = MLPActor(obs_dim, act_dim, hidden_dims, max_action=max_action).to(device)
    actor.load_state_dict(actor_state, strict=False)
    return _TD3ActorPolicy(actor, "mlp", 1, device, ckpt.get("obs_normalizer"))


def _ad_final_checkpoint_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}_final{checkpoint.suffix}")


def _ad_best_checkpoint_path(checkpoint: Path) -> Path:
    return checkpoint.parent / "ad_best.pt"


def train_ad_model(
    history_dir: str | Path,
    checkpoint: str | Path,
    context_length: int,
    train_steps: int,
    batch_size: int,
    embed_dim: int,
    n_layers: int,
    n_heads: int,
    lr: float,
    checkpoint_interval: int,
    seed: int,
    device_name: str,
    quiet: bool = False,
) -> dict[str, object]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dataset = LearningHistoryDataset(history_dir, context_length=context_length)
    first = dataset[0]
    obs_dim = int(first["obs_seq"].shape[-1])
    act_dim = int(first["target_act_seq"].shape[-1])

    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ADTransformer(obs_dim, act_dim, embed_dim, n_layers, n_heads, context_length).to(device)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = checkpoint.parent / "ad_latest.pt"
    best_checkpoint = _ad_best_checkpoint_path(checkpoint)
    final_checkpoint = _ad_final_checkpoint_path(checkpoint)

    def save_model(
        path: Path,
        step_value: int,
        loss_value: float,
        *,
        best_loss_value: float,
        best_step_value: int,
        selection: str,
    ) -> None:
        ckpt = {
            "model_state": model.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "context_length": int(context_length),
            "embed_dim": int(embed_dim),
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "train_steps": int(step_value),
            "loss": float(loss_value),
            "best_loss": float(best_loss_value),
            "best_step": int(best_step_value),
            "selection": str(selection),
        }
        torch.save(ckpt, path)

    step = 0
    last_loss = math.nan
    best_loss = math.inf
    best_step = 0
    resume_from = ""
    resume_path = latest_checkpoint if latest_checkpoint.exists() else checkpoint
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if (
            int(saved.get("obs_dim", -1)) == obs_dim
            and int(saved.get("act_dim", -1)) == act_dim
            and int(saved.get("context_length", -1)) == int(context_length)
            and int(saved.get("embed_dim", -1)) == int(embed_dim)
            and int(saved.get("n_layers", -1)) == int(n_layers)
            and int(saved.get("n_heads", -1)) == int(n_heads)
        ):
            model.load_state_dict(saved["model_state"])
            step = min(int(saved.get("train_steps", 0)), int(train_steps))
            last_loss = float(saved.get("loss", math.nan))
            best_loss = float(saved.get("best_loss", last_loss if math.isfinite(last_loss) else math.inf))
            best_step = int(saved.get("best_step", step if math.isfinite(best_loss) else 0))
            resume_from = str(resume_path)
            if not quiet and step:
                print(f"Resumed AD from {resume_path} at step {step}/{train_steps}", flush=True)

    while step < int(train_steps):
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            pred = model(batch["obs_seq"], batch["prev_act_seq"], batch["reward_seq"], batch["done_seq"])
            sq = (pred - batch["target_act_seq"]) ** 2
            mask = batch["loss_mask"].expand_as(sq)
            loss = (sq * mask).sum() / mask.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            step += 1
            last_loss = float(loss.item())
            if math.isfinite(last_loss) and last_loss < best_loss:
                best_loss = last_loss
                best_step = step
                save_model(
                    best_checkpoint,
                    step,
                    last_loss,
                    best_loss_value=best_loss,
                    best_step_value=best_step,
                    selection="best_train_loss",
                )
            if not quiet and (step == 1 or step % 100 == 0):
                print(f"step={step} ad/loss={last_loss:.6f}", flush=True)
            if int(checkpoint_interval) > 0 and step % int(checkpoint_interval) == 0:
                save_model(
                    latest_checkpoint,
                    step,
                    last_loss,
                    best_loss_value=best_loss,
                    best_step_value=best_step,
                    selection="latest",
                )
            if step >= int(train_steps):
                break

    if not best_checkpoint.exists():
        best_loss = last_loss if math.isfinite(last_loss) else math.inf
        best_step = step
        save_model(
            best_checkpoint,
            step,
            last_loss,
            best_loss_value=best_loss,
            best_step_value=best_step,
            selection="best_train_loss",
        )

    save_model(
        latest_checkpoint,
        step,
        last_loss,
        best_loss_value=best_loss,
        best_step_value=best_step,
        selection="latest",
    )
    save_model(
        final_checkpoint,
        step,
        last_loss,
        best_loss_value=best_loss,
        best_step_value=best_step,
        selection="final",
    )
    best_saved = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    torch.save(best_saved, checkpoint)
    return {
        "ad/train_loss": float(last_loss),
        "ad/best_train_loss": float(best_loss),
        "ad/best_step": int(best_step),
        "ad/train_steps": int(train_steps),
        "ad/samples": len(dataset),
        "checkpoint": str(checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "resume_from": resume_from,
    }


def evaluate_ad_model(
    checkpoint: str | Path,
    env_name: str,
    episodes: int,
    seed: int,
    max_steps_per_episode: int | None,
    device_name: str,
) -> dict[str, object]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ADTransformer(
        ckpt["obs_dim"],
        ckpt["act_dim"],
        ckpt["embed_dim"],
        ckpt["n_layers"],
        ckpt["n_heads"],
        ckpt["context_length"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    env = make_ad_env(env_name, seed=seed, max_steps_per_episode=max_steps_per_episode)
    returns = []
    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=int(seed + ep))
        done = False
        truncated = False
        ep_return = 0.0
        obs_hist: list[np.ndarray] = []
        prev_act_hist: list[np.ndarray] = []
        prev_rew_hist: list[np.ndarray] = []
        prev_done_hist: list[np.ndarray] = []
        prev_action = np.zeros(ckpt["act_dim"], dtype=np.float32)
        prev_reward = np.zeros(1, dtype=np.float32)
        prev_done = np.zeros(1, dtype=np.float32)

        while not (done or truncated):
            obs_hist.append(np.asarray(obs, dtype=np.float32))
            prev_act_hist.append(prev_action.copy())
            prev_rew_hist.append(prev_reward.copy())
            prev_done_hist.append(prev_done.copy())

            obs_seq = _left_pad_sequence(obs_hist, ckpt["context_length"], ckpt["obs_dim"])
            act_seq = _left_pad_sequence(prev_act_hist, ckpt["context_length"], ckpt["act_dim"])
            rew_seq = _left_pad_sequence(prev_rew_hist, ckpt["context_length"], 1)
            done_seq = _left_pad_sequence(prev_done_hist, ckpt["context_length"], 1)
            with torch.no_grad():
                pred = model(
                    torch.as_tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(act_seq, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(rew_seq, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.as_tensor(done_seq, dtype=torch.float32, device=device).unsqueeze(0),
                )
            action = pred[0, -1].cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)

            next_obs, reward, done, truncated, _ = env.step(action)
            ep_return += float(reward)
            prev_action = action
            prev_reward = np.asarray([reward], dtype=np.float32)
            prev_done = np.asarray([float(done or truncated)], dtype=np.float32)
            obs = next_obs

        returns.append(ep_return)
    env.close()
    return {
        "eval/return": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/episodes": int(episodes),
        "env": str(env_name),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="collect cross-episode histories")
    collect.add_argument("--env", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--histories", type=int, default=12)
    collect.add_argument("--episodes-per-history", type=int, default=4)
    collect.add_argument("--max-steps-per-episode", type=int, default=None)
    collect.add_argument("--source-checkpoint", default="")
    collect.add_argument("--source-noise", type=float, default=0.05)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--quiet", action="store_true")

    train = sub.add_parser("train", help="train AD transformer")
    train.add_argument("--history-dir", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--context-length", type=int, default=256)
    train.add_argument("--train-steps", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--embed-dim", type=int, default=128)
    train.add_argument("--n-layers", type=int, default=2)
    train.add_argument("--n-heads", type=int, default=4)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--checkpoint-interval", type=int, default=1000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="")
    train.add_argument("--quiet", action="store_true")

    evaluate = sub.add_parser("evaluate", help="evaluate AD policy without gradient updates")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--env", required=True)
    evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.add_argument("--max-steps-per-episode", type=int, default=None)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", default="")
    evaluate.add_argument("--quiet", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown and not getattr(args, "quiet", False):
        print(f"Ignoring extra runner overrides: {' '.join(unknown)}", file=sys.stderr)

    if args.command == "collect":
        metrics = collect_learning_histories(
            env_name=args.env,
            histories=args.histories,
            episodes_per_history=args.episodes_per_history,
            save_dir=args.output_dir,
            seed=args.seed,
            max_steps_per_episode=args.max_steps_per_episode,
            source_checkpoint=args.source_checkpoint or None,
            source_noise=args.source_noise,
        )
    elif args.command == "train":
        metrics = train_ad_model(
            history_dir=args.history_dir,
            checkpoint=args.checkpoint,
            context_length=args.context_length,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            embed_dim=args.embed_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            lr=args.lr,
            checkpoint_interval=args.checkpoint_interval,
            seed=args.seed,
            device_name=args.device,
            quiet=args.quiet,
        )
    elif args.command == "evaluate":
        metrics = evaluate_ad_model(
            checkpoint=args.checkpoint,
            env_name=args.env,
            episodes=args.episodes,
            seed=args.seed,
            max_steps_per_episode=args.max_steps_per_episode,
            device_name=args.device,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
