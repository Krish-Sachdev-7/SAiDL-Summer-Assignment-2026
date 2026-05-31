import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PYTHON = WORKSPACE / ".venv" / "Scripts" / "python.exe"
AD_SCRIPT = ROOT / "rl" / "src" / "algorithm_distillation.py"


class AlgorithmDistillationTests(unittest.TestCase):
    def test_dataset_uses_previous_actions_and_cross_episode_windows(self):
        from rl.src.algorithm_distillation import LearningHistoryDataset

        with tempfile.TemporaryDirectory() as tmp:
            history_dir = Path(tmp)
            obs = torch.arange(24, dtype=torch.float32).view(8, 3)
            act = torch.arange(16, dtype=torch.float32).view(8, 2) / 10.0
            rew = torch.arange(8, dtype=torch.float32).view(8, 1)
            done = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0], [1.0]])
            episode_id = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
            torch.save(
                {
                    "obs": obs,
                    "act": act,
                    "rew": rew,
                    "done": done,
                    "episode_id": episode_id,
                },
                history_dir / "history_000000.pt",
            )

            dataset = LearningHistoryDataset(history_dir, context_length=5)
            sample = dataset[0]

            self.assertIn("prev_act_seq", sample)
            self.assertIn("target_act_seq", sample)
            self.assertIn("loss_mask", sample)
            self.assertTrue(torch.allclose(sample["prev_act_seq"][0], torch.zeros(2)))
            self.assertTrue(torch.allclose(sample["prev_act_seq"][1], act[0]))
            self.assertTrue(torch.equal(sample["target_act_seq"], act[:5]))
            self.assertGreater(sample["episode_id_seq"].unique().numel(), 1)

    def test_cli_collect_train_and_evaluate_toy_cross_episode_histories(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            history_dir = tmp_dir / "histories"
            checkpoint = tmp_dir / "ad.pt"

            collect = subprocess.run(
                [
                    str(PYTHON),
                    str(AD_SCRIPT),
                    "collect",
                    "--env",
                    "toy_memory",
                    "--output-dir",
                    str(history_dir),
                    "--histories",
                    "2",
                    "--episodes-per-history",
                    "2",
                    "--max-steps-per-episode",
                    "4",
                    "--seed",
                    "7",
                    "--quiet",
                ],
                cwd=ROOT / "rl",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=45,
            )
            self.assertEqual(collect.returncode, 0, msg=collect.stderr)
            history_files = sorted(history_dir.glob("history_*.pt"))
            self.assertEqual(len(history_files), 2)
            saved = torch.load(history_files[0], map_location="cpu", weights_only=False)
            self.assertGreater(saved["episode_id"].unique().numel(), 1)

            train = subprocess.run(
                [
                    str(PYTHON),
                    str(AD_SCRIPT),
                    "train",
                    "--history-dir",
                    str(history_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--context-length",
                    "4",
                    "--train-steps",
                    "2",
                    "--batch-size",
                    "2",
                    "--embed-dim",
                    "16",
                    "--n-layers",
                    "1",
                    "--n-heads",
                    "2",
                    "--checkpoint-interval",
                    "1",
                    "--device",
                    "cpu",
                    "--quiet",
                ],
                cwd=ROOT / "rl",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            self.assertEqual(train.returncode, 0, msg=train.stderr)
            self.assertTrue(checkpoint.exists())
            self.assertTrue((checkpoint.parent / "ad_latest.pt").exists())

            evaluate = subprocess.run(
                [
                    str(PYTHON),
                    str(AD_SCRIPT),
                    "evaluate",
                    "--checkpoint",
                    str(checkpoint),
                    "--env",
                    "toy_memory",
                    "--episodes",
                    "2",
                    "--max-steps-per-episode",
                    "4",
                    "--device",
                    "cpu",
                    "--quiet",
                ],
                cwd=ROOT / "rl",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            self.assertEqual(evaluate.returncode, 0, msg=evaluate.stderr)
            metrics = json.loads(evaluate.stdout.strip().splitlines()[-1])
            self.assertIn("eval/return", metrics)
            self.assertEqual(metrics["eval/episodes"], 2)


if __name__ == "__main__":
    unittest.main()
