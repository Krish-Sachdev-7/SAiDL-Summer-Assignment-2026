import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl" / "src"))


class RLAttentionAnalysisTests(unittest.TestCase):
    def test_running_observation_normalizer_round_trips_state(self):
        from train import RunningObservationNormalizer

        norm = RunningObservationNormalizer(obs_dim=3)
        norm.update([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

        normalized = norm.normalize([1.0, 2.0, 3.0])
        restored = RunningObservationNormalizer(obs_dim=3)
        restored.load_state_dict(norm.state_dict())

        self.assertEqual(tuple(normalized.shape), (3,))
        self.assertTrue(torch.isfinite(torch.as_tensor(normalized)).all().item())
        self.assertTrue(
            torch.allclose(
                torch.as_tensor(normalized),
                torch.as_tensor(restored.normalize([1.0, 2.0, 3.0])),
            )
        )

    def test_best_eval_tracking_keeps_larger_return(self):
        from train import _maybe_update_best_eval

        best_return, best_step, improved = _maybe_update_best_eval(10.0, 100, float("-inf"), 0)
        self.assertTrue(improved)
        self.assertEqual(best_return, 10.0)
        self.assertEqual(best_step, 100)

        best_return, best_step, improved = _maybe_update_best_eval(7.0, 200, best_return, best_step)
        self.assertFalse(improved)
        self.assertEqual(best_return, 10.0)
        self.assertEqual(best_step, 100)

    def test_attention_metrics_use_recent_lag_indexing_and_entropy(self):
        from train import compute_attention_metrics

        probs = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        attn = probs.view(1, 1, 1, 4).repeat(2, 3, 4, 1)

        metrics = compute_attention_metrics([attn])

        expected_entropy = float(-(probs * probs.clamp_min(1e-8).log()).sum().item())
        self.assertAlmostEqual(metrics["attention/entropy_mean"], expected_entropy, places=6)
        self.assertAlmostEqual(metrics["attention/lag_0"], 0.4, places=6)
        self.assertAlmostEqual(metrics["attention/lag_1"], 0.3, places=6)
        self.assertAlmostEqual(metrics["attention/lag_2"], 0.2, places=6)
        self.assertAlmostEqual(metrics["attention/lag_3"], 0.1, places=6)
        self.assertAlmostEqual(metrics["attention/max_weight_mean"], 0.4, places=6)
        self.assertAlmostEqual(metrics["attention/effective_context_mean"], math.exp(expected_entropy), places=6)

    def test_attention_metrics_work_after_transformer_actor_forward(self):
        from networks import TransformerActor
        from train import compute_attention_metrics

        actor = TransformerActor(
            obs_dim=5,
            act_dim=3,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            context_length=8,
            max_action=1.0,
        )
        action = actor(torch.randn(2, 8, 5), torch.randn(2, 8, 3))

        metrics = compute_attention_metrics(actor.last_attn_weights)

        self.assertEqual(tuple(action.shape), (2, 3))
        self.assertTrue(torch.isfinite(action).all().item())
        self.assertIn("attention/entropy_mean", metrics)
        self.assertIn("attention/effective_context_mean", metrics)
        for i in range(8):
            self.assertIn(f"attention/lag_{i}", metrics)
            self.assertTrue(math.isfinite(metrics[f"attention/lag_{i}"]))
        lag_sum = sum(metrics[f"attention/lag_{i}"] for i in range(8))
        self.assertAlmostEqual(lag_sum, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
