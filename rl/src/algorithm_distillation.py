"""Algorithm Distillation bonus code."""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import gymnasium as gym
import numpy as np


class ADTransformer(nn.Module):
    """Causal transformer for AD histories."""
    def __init__(self, obs_dim: int, act_dim: int, embed_dim: int, n_layers: int,
                 n_heads: int, max_history_len: int):
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

    def forward(self, obs_seq, act_seq, reward_seq, done_seq):
        bsz, seq_len, _ = obs_seq.shape
        x = torch.cat([obs_seq, act_seq, reward_seq, done_seq], dim=-1)
        x = self.in_proj(x)
        pos = torch.arange(seq_len, device=x.device)
        x = x + self.pos_emb(pos)[None, :, :]

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=mask)
        x = self.ln(x)
        return torch.tanh(self.action_head(x))


class LearningHistoryDataset(Dataset):
    """Cross-episode AD history dataset."""
    def __init__(self, history_dir: str, context_length: int):
        self.history_dir = Path(history_dir)
        self.context_length = int(context_length)
        self.files = sorted(self.history_dir.glob("*.pt"))
        self.samples = []

        for f in self.files:
            d = torch.load(f, map_location="cpu")
            obs = d["obs"]
            act = d["act"]
            rew = d["rew"]
            done = d["done"]
            length = obs.shape[0]

            if length < self.context_length:
                continue

            for start in range(0, length - self.context_length + 1):
                end = start + self.context_length
                self.samples.append(
                    (
                        obs[start:end],
                        act[start:end],
                        rew[start:end],
                        done[start:end],
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, act, rew, done = self.samples[idx]
        return {
            "obs_seq": obs.float(),
            "act_seq": act.float(),
            "reward_seq": rew.float(),
            "done_seq": done.float(),
        }


def collect_learning_histories(env_name: str, n_checkpoints: int, episodes_per_checkpoint: int,
                                save_dir: str):
    """Collect source-policy learning histories."""
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_name)

    rng = np.random.default_rng(42)
    file_count = 0

        # Later source checkpoints use less noise.
    for ckpt in range(int(n_checkpoints)):
        exploration = max(0.05, 1.0 - ckpt / max(1, n_checkpoints - 1))

        for ep in range(int(episodes_per_checkpoint)):
            obs, _ = env.reset()
            done = False
            truncated = False

            obs_hist = []
            act_hist = []
            rew_hist = []
            done_hist = []

            while not (done or truncated):
                action = rng.normal(0.0, exploration, size=env.action_space.shape[0])
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_obs, rew, done, truncated, _ = env.step(action)
                obs_hist.append(obs)
                act_hist.append(action)
                rew_hist.append([rew])
                done_hist.append([float(done or truncated)])
                obs = next_obs

            data = {
                "obs": torch.tensor(np.asarray(obs_hist), dtype=torch.float32),
                "act": torch.tensor(np.asarray(act_hist), dtype=torch.float32),
                "rew": torch.tensor(np.asarray(rew_hist), dtype=torch.float32),
                "done": torch.tensor(np.asarray(done_hist), dtype=torch.float32),
                "checkpoint": ckpt,
                "episode": ep,
            }
            torch.save(data, out / f"history_{file_count:06d}.pt")
            file_count += 1

    env.close()
    return {"saved_histories": file_count, "output_dir": str(out)}
