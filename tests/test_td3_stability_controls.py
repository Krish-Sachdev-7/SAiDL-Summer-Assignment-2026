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


class FixedReplay:
    def sample(self, batch_size):
        return {
            "obs": torch.zeros(batch_size, 3),
            "action": torch.zeros(batch_size, 2),
            "reward": torch.full((batch_size, 1), 1000.0),
            "next_obs": torch.zeros(batch_size, 3),
            "done": torch.zeros(batch_size, 1),
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
