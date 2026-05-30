"""Learned reward model path."""
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class RewardModel(nn.Module):
    """Scalar learned reward model."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [obs_dim + act_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.cat([obs, action], dim=-1)
        return self.head(self.backbone(x)).squeeze(-1)


class PreferenceLearner:
    """Preference data and reward-model trainer."""
    def __init__(self, reward_model: RewardModel, cfg):
        self.reward_model = reward_model
        self.cfg = cfg
        self.segment_length = int(cfg.reward.segment_length)
        self.device = next(reward_model.parameters()).device
        self.optimizer = torch.optim.Adam(
            reward_model.parameters(), lr=float(cfg.reward.reward_model_lr)
        )
        self.pref_data = []

    def collect_preferences(self, replay_buffer, n_comparisons: int):
        """Sample labeled segment pairs."""
        if replay_buffer.size < self.segment_length + 1:
            return 0

        max_start = replay_buffer.size - self.segment_length
        for _ in range(int(n_comparisons)):
            s1 = random.randint(0, max_start - 1)
            s2 = random.randint(0, max_start - 1)

            obs1 = torch.as_tensor(replay_buffer.obs[s1 : s1 + self.segment_length], dtype=torch.float32)
            act1 = torch.as_tensor(replay_buffer.action[s1 : s1 + self.segment_length], dtype=torch.float32)
            rew1 = float(replay_buffer.reward[s1 : s1 + self.segment_length].sum())

            obs2 = torch.as_tensor(replay_buffer.obs[s2 : s2 + self.segment_length], dtype=torch.float32)
            act2 = torch.as_tensor(replay_buffer.action[s2 : s2 + self.segment_length], dtype=torch.float32)
            rew2 = float(replay_buffer.reward[s2 : s2 + self.segment_length].sum())

            label = 1.0 if rew1 > rew2 else 0.0
            self.pref_data.append((obs1, act1, obs2, act2, label))
        return len(self.pref_data)

    def train_reward_model(self):
        """Fit the preference reward model."""
        if not self.pref_data:
            return {"reward_model_loss": float("nan")}

        obs1 = torch.stack([x[0] for x in self.pref_data])
        act1 = torch.stack([x[1] for x in self.pref_data])
        obs2 = torch.stack([x[2] for x in self.pref_data])
        act2 = torch.stack([x[3] for x in self.pref_data])
        labels = torch.tensor([x[4] for x in self.pref_data], dtype=torch.float32)

        ds = TensorDataset(obs1, act1, obs2, act2, labels)
        dl = DataLoader(ds, batch_size=64, shuffle=True)

        self.reward_model.train()
        losses = []
        for b_obs1, b_act1, b_obs2, b_act2, b_label in dl:
            b_obs1 = b_obs1.to(self.device)
            b_act1 = b_act1.to(self.device)
            b_obs2 = b_obs2.to(self.device)
            b_act2 = b_act2.to(self.device)
            b_label = b_label.to(self.device)

            r1 = self.reward_model(b_obs1.view(-1, b_obs1.size(-1)), b_act1.view(-1, b_act1.size(-1)))
            r2 = self.reward_model(b_obs2.view(-1, b_obs2.size(-1)), b_act2.view(-1, b_act2.size(-1)))

            r1 = r1.view(b_obs1.size(0), b_obs1.size(1)).sum(dim=1)
            r2 = r2.view(b_obs2.size(0), b_obs2.size(1)).sum(dim=1)
            logits = r1 - r2

            loss = nn.functional.binary_cross_entropy_with_logits(logits, b_label)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))

        return {"reward_model_loss": float(sum(losses) / max(1, len(losses)))}
