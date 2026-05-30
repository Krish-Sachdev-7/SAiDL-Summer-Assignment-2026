"""Export attention plots from saved TD3 checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


RL_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = RL_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from env_wrappers import HiddenVelocityWrapper
from networks import TransformerActor
import gymnasium as gym


def make_hopper_env(setting: str):
    env = gym.make("Hopper-v5")
    if setting == "full":
        return env
    if setting == "hidden":
        return HiddenVelocityWrapper(env)
    raise ValueError(f"Unsupported setting: {setting}")


class ActorAgent:
    def __init__(self, actor: TransformerActor):
        self.actor = actor
        self.device = next(actor.parameters()).device
        self.max_action = float(getattr(actor, "max_action", 1.0))

    def select_action(self, obs_seq: np.ndarray, act_seq: np.ndarray) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            act_t = torch.as_tensor(act_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.actor(obs_t, act_t).squeeze(0)
        return action.cpu().numpy().clip(-self.max_action, self.max_action)


def load_actor(checkpoint_path: Path, setting: str, device: torch.device) -> tuple[ActorAgent, int]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("agent_type") != "transformer":
        raise ValueError(f"Checkpoint is not a Transformer actor: {checkpoint_path}")

    env = make_hopper_env(setting)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    max_action = float(env.action_space.high[0])
    env.close()

    context_length = int(ckpt.get("context_length", 8))
    actor = TransformerActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        embed_dim=128,
        n_layers=2,
        n_heads=4,
        context_length=context_length,
        max_action=max_action,
        pos_encoding="learned",
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    return ActorAgent(actor), context_length


def rollout_attention(agent: ActorAgent, setting: str, context_length: int, episodes: int, seed: int):
    env = make_hopper_env(setting)
    act_dim = int(env.action_space.shape[0])
    attention_vectors = []
    entropies = []
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_hist = [np.zeros_like(obs, dtype=np.float32) for _ in range(max(context_length - 1, 0))]
        act_hist = [np.zeros(act_dim, dtype=np.float32) for _ in range(max(context_length - 1, 0))]
        done = False
        truncated = False
        ep_return = 0.0

        while not (done or truncated):
            obs_hist.append(np.asarray(obs, dtype=np.float32))
            obs_hist = obs_hist[-context_length:]
            act_seq = act_hist[-(context_length - 1):] + [np.zeros(act_dim, dtype=np.float32)]
            obs_seq = np.stack(obs_hist, axis=0)
            act_seq_arr = np.stack(act_seq, axis=0)

            action = agent.select_action(obs_seq, act_seq_arr)
            attn_layers = getattr(agent.actor, "last_attn_weights", [])
            if attn_layers:
                stacked = torch.stack(attn_layers, dim=0).detach().cpu()
                last_query = stacked[:, 0, :, -1, :]
                vec = last_query.mean(dim=(0, 1)).numpy()
                attention_vectors.append(vec)
                probs = last_query.clamp_min(1e-8)
                ent = -(probs * probs.log()).sum(dim=-1)
                entropies.append(float(ent.mean().item()))

            next_obs, reward, done, truncated, _ = env.step(action)
            ep_return += float(reward)
            act_hist.append(np.asarray(action, dtype=np.float32))
            act_hist = act_hist[-(context_length - 1):]
            obs = next_obs

        returns.append(ep_return)

    env.close()
    if not attention_vectors:
        raise RuntimeError("No attention weights were captured from the actor.")
    return {
        "mean_attention": np.stack(attention_vectors, axis=0).mean(axis=0),
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
    }


def save_comparison(results: dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label in labels:
        ax.plot(results[label]["mean_attention"], marker="o", label=label)
    ax.set_xlabel("Past timestep index within context window")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Mean attention over past timesteps")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_attention_mean_weights.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(
        labels,
        [results[label]["entropy_mean"] for label in labels],
        yerr=[results[label]["entropy_std"] for label in labels],
    )
    ax.set_ylabel("Attention entropy")
    ax.set_title("Attention entropy over rollout decisions")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_attention_entropy_rollout.png", dpi=180)
    plt.close(fig)

    with (out_dir / "attention_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["setting", "return_mean", "return_std", "entropy_mean", "entropy_std"],
        )
        writer.writeheader()
        for label in labels:
            row = {key: results[label][key] for key in writer.fieldnames if key != "setting"}
            row["setting"] = label
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--hidden-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for setting, checkpoint in [("full", args.full_checkpoint), ("hidden", args.hidden_checkpoint)]:
        agent, context_length = load_actor(checkpoint, setting=setting, device=device)
        results[setting] = rollout_attention(
            agent=agent,
            setting=setting,
            context_length=context_length,
            episodes=int(args.episodes),
            seed=int(args.seed),
        )
    save_comparison(results, args.out_dir)
    print(f"Wrote attention analysis artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
