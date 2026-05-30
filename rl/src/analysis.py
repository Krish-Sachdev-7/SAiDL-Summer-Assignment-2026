"""Attention plotting helpers."""
from collections import deque

import torch
import numpy as np
import matplotlib.pyplot as plt


def rollout_with_attention(agent, env, n_episodes: int = 5) -> list[dict]:
    """Roll out and save attention weights."""
    episodes = []
    context_len = int(getattr(agent.actor, "context_length", 1))
    act_dim = int(getattr(agent.actor, "act_dim", env.action_space.shape[0]))

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False

        obs_hist = deque(maxlen=context_len)
        act_hist = deque(maxlen=context_len)
        obs_list, act_list, rew_list, attn_list = [], [], [], []

        while not (done or truncated):
            obs_hist.append(np.asarray(obs, dtype=np.float32))
            while len(obs_hist) < context_len:
                obs_hist.appendleft(np.zeros_like(obs_hist[0]))

            while len(act_hist) < context_len - 1:
                act_hist.appendleft(np.zeros(act_dim, dtype=np.float32))
            act_hist_full = list(act_hist) + [np.zeros(act_dim, dtype=np.float32)]

            obs_seq = np.stack(list(obs_hist), axis=0)
            act_seq = np.stack(act_hist_full, axis=0)

            action = agent.select_action({"obs_seq": obs_seq, "act_seq": act_seq}, noise=0.0)
            obs_next, reward, done, truncated, _ = env.step(action)

            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(reward)

            if hasattr(agent.actor, "last_attn_weights") and agent.actor.last_attn_weights:
                stacked = torch.stack(agent.actor.last_attn_weights, dim=0).detach().cpu().numpy()
                attn_list.append(stacked)

            act_hist.append(np.asarray(action, dtype=np.float32))
            obs = obs_next

        episodes.append(
            {
                "observations": np.asarray(obs_list, dtype=np.float32),
                "actions": np.asarray(act_list, dtype=np.float32),
                "rewards": np.asarray(rew_list, dtype=np.float32),
                "attention_weights": np.asarray(attn_list, dtype=np.float32),
            }
        )
    return episodes


def plot_mean_attention(episodes: list[dict], title: str, save_path: str):
    """Plot mean attention over the context window."""
    weights = []
    for ep in episodes:
        attn = ep.get("attention_weights")
        if attn is None or len(attn) == 0:
            continue
        # Use the last query and average layers/heads.
        attn = np.asarray(attn)
        vec = attn[..., -1, :].mean(axis=(1, 2, 3))
        weights.append(vec)

    if not weights:
        return

    min_t = min(w.shape[0] for w in weights)
    arr = np.stack([w[:min_t] for w in weights], axis=0)
    mean_vec = arr.mean(axis=(0, 1))

    plt.figure(figsize=(8, 4))
    plt.plot(mean_vec)
    plt.title(title)
    plt.xlabel("Past timestep index")
    plt.ylabel("Mean attention")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """Compute attention entropy per timestep."""
    attn = attention_weights.clamp_min(1e-8)
    return -(attn * attn.log()).sum(dim=-1)


def plot_entropy_vs_training(checkpoints: list, env, save_path: str):
    """Plot entropy across checkpoints."""
    steps = []
    entropy_vals = []
    for item in checkpoints:
        if isinstance(item, dict):
            step = int(item.get("step", 0))
            attn = item.get("attention")
        else:
            step, attn = item
        if attn is None:
            continue

        attn_t = torch.as_tensor(attn, dtype=torch.float32)
        ent = compute_attention_entropy(attn_t).mean().item()
        steps.append(step)
        entropy_vals.append(ent)

    if not steps:
        return

    order = np.argsort(np.asarray(steps))
    steps = np.asarray(steps)[order]
    entropy_vals = np.asarray(entropy_vals)[order]

    plt.figure(figsize=(8, 4))
    plt.plot(steps, entropy_vals, marker="o")
    plt.xlabel("Training step")
    plt.ylabel("Mean attention entropy")
    plt.title("Attention entropy vs training")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
