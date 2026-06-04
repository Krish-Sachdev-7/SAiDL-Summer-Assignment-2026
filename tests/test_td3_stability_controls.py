import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl" / "src"))


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_action = 1.0
        self.linear = nn.Linear(3, 2)

    def forward(self, obs):
        return torch.tanh(self.linear(obs))


class TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, obs, action):
        return self.linear(torch.cat([obs, action], dim=-1))


class TinySequenceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_action = 1.0
        self.linear = nn.Linear(3, 2)

    def forward(self, obs, act_seq=None):
        if act_seq is not None:
            obs = obs[:, -1]
        return torch.tanh(self.linear(obs))


class TinySequenceCritic(nn.Module):
    supports_sequence = True

    def __init__(self):
        super().__init__()
        self.seq_calls = 0
        self.linear = nn.Linear(5, 1)

    def forward(self, obs, action, obs_seq=None, act_seq=None):
        if obs_seq is not None and act_seq is not None:
            self.seq_calls += 1
            obs = obs_seq[:, -1]
        return self.linear(torch.cat([obs, action], dim=-1))


class FixedReplay:
    def sample(self, batch_size):
        return {
            "obs": torch.zeros(batch_size, 3),
            "action": torch.zeros(batch_size, 2),
            "reward": torch.full((batch_size, 1), 1000.0),
            "next_obs": torch.zeros(batch_size, 3),
            "done": torch.zeros(batch_size, 1),
        }


class FixedSequenceReplay:
    def sample(self, batch_size):
        obs_seq = torch.zeros(batch_size, 4, 3)
        act_seq = torch.zeros(batch_size, 4, 2)
        next_obs_seq = torch.zeros(batch_size, 4, 3)
        next_act_seq = torch.zeros(batch_size, 4, 2)
        obs_seq[:, -1, 0] = 1.0
        next_obs_seq[:, -1, 0] = 2.0
        return {
            "obs": obs_seq[:, -1],
            "action": torch.zeros(batch_size, 2),
            "reward": torch.ones(batch_size, 1),
            "next_obs": next_obs_seq[:, -1],
            "done": torch.zeros(batch_size, 1),
            "obs_seq": obs_seq,
            "act_seq": act_seq,
            "next_obs_seq": next_obs_seq,
            "next_act_seq": next_act_seq,
        }


def stability_cfg():
    return SimpleNamespace(
        agent=SimpleNamespace(
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
            target_noise=0.2,
            target_noise_final=0.02,
            target_noise_decay_steps=100,
            noise_clip=0.5,
            noise_clip_final=0.1,
            noise_clip_decay_steps=100,
            grad_clip_norm=0.1,
            critic_loss="huber",
            huber_beta=0.5,
            target_q_clip=1.0,
            lr=1e-3,
            start_steps=100,
        )
    )


class TD3StabilityControlTests(unittest.TestCase):
    def test_transformer_critic_accepts_history_and_action(self):
        from networks import TransformerCritic

        critic = TransformerCritic(
            obs_dim=5,
            act_dim=3,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            context_length=8,
        )

        q = critic(
            torch.randn(2, 5),
            torch.randn(2, 3),
            obs_seq=torch.randn(2, 8, 5),
            act_seq=torch.randn(2, 8, 3),
        )

        self.assertEqual(tuple(q.shape), (2, 1))
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(getattr(critic, "supports_sequence", False))

    def test_td3_clips_targets_uses_robust_loss_and_reports_diagnostics(self):
        from td3 import TD3

        torch.manual_seed(0)
        agent = TD3(TinyActor(), TinyCritic(), TinyCritic(), stability_cfg())

        metrics = agent.update(FixedReplay(), batch_size=4, step=150)

        self.assertLessEqual(metrics["target_q_abs_max"], 1.000001)
        self.assertGreater(metrics["target_q_clipped_frac"], 0.0)
        self.assertEqual(metrics["critic_loss_type"], "huber")
        self.assertLess(metrics["target_noise_std"], 0.2)
        self.assertGreater(metrics["target_noise_std"], 0.02)
        self.assertLess(metrics["target_noise_clip"], 0.5)
        for key in ["critic1_grad_norm", "critic2_grad_norm", "q_gap_abs_mean"]:
            self.assertTrue(math.isfinite(metrics[key]))

    def test_td3_uses_sequence_critic_when_available(self):
        from td3 import TD3

        torch.manual_seed(0)
        critic1 = TinySequenceCritic()
        critic2 = TinySequenceCritic()
        agent = TD3(TinySequenceActor(), critic1, critic2, stability_cfg())

        metrics = agent.update(FixedSequenceReplay(), batch_size=4, step=150)

        self.assertGreater(agent.critic1.seq_calls, 0)
        self.assertGreater(agent.critic2.seq_calls, 0)
        self.assertEqual(metrics["critic_sequence_context"], 1.0)

    def test_exploration_noise_schedule_decays_after_random_warmup(self):
        from train import _scheduled_exploration_noise

        cfg = SimpleNamespace(
            agent=SimpleNamespace(
                exploration_noise=0.1,
                exploration_noise_final=0.01,
                exploration_noise_decay_steps=100,
                start_steps=50,
            )
        )

        self.assertAlmostEqual(_scheduled_exploration_noise(cfg, 50), 0.1)
        self.assertAlmostEqual(_scheduled_exploration_noise(cfg, 100), 0.055)
        self.assertAlmostEqual(_scheduled_exploration_noise(cfg, 150), 0.01)
        self.assertAlmostEqual(_scheduled_exploration_noise(cfg, 500), 0.01)


if __name__ == "__main__":
    unittest.main()
