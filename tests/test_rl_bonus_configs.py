import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PYTHON = WORKSPACE / ".venv" / "Scripts" / "python.exe"
RUNNER = WORKSPACE / "saidl-assignment-runner.ipynb"


def runner_source() -> str:
    notebook = json.loads(RUNNER.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


class RLBonusConfigTests(unittest.TestCase):
    def test_hydra_composes_positional_combined_and_xlstm_configs(self):
        commands = [
            [
                str(PYTHON),
                "src/train.py",
                "--cfg",
                "job",
                "agent=td3_transformer",
                "env=hopper_hidden_vel",
                "+pos_encoding=rope",
                "logging.wandb.enable=false",
            ],
            [
                str(PYTHON),
                "src/train.py",
                "--cfg",
                "job",
                "agent=td3_transformer",
                "env=hopper_combined_pomdp",
                "agent.actor.context_length=32",
                "logging.wandb.enable=false",
            ],
            [
                str(PYTHON),
                "src/train.py",
                "--cfg",
                "job",
                "agent=td3_xlstm",
                "env=hopper_hidden_vel",
                "logging.wandb.enable=false",
            ],
        ]
        for cmd in commands:
            with self.subTest(cmd=" ".join(cmd[2:])):
                result = subprocess.run(
                    cmd,
                    cwd=ROOT / "rl",
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_runner_runs_only_aft_and_rl_bonus_commands(self):
        source = runner_source()

        self.assertIn("coreml_bonus_aft_full_ctx512", source)
        self.assertIn("coreml_bonus_aft_local_ctx512", source)
        self.assertIn("coreml_bonus_aft_simple_ctx512", source)
        self.assertIn("run_block(\"core_bonus_aft\"", source)
        self.assertIn("rl_bonus_pos_learned_seed", source)
        self.assertIn("rl_bonus_pos_sinusoidal_seed", source)
        self.assertIn("rl_bonus_pos_rope_seed", source)
        self.assertIn("rl_bonus_combined_L32_seed", source)
        self.assertIn("rl_bonus_xlstm_hidden_seed", source)
        self.assertIn("rl_bonus_xlstm_delayed_seed", source)
        self.assertIn("algorithm_distillation.py collect", source)
        self.assertIn("algorithm_distillation.py train", source)
        self.assertIn("algorithm_distillation.py evaluate", source)
        self.assertIn("run_block(\"rl_bonus\"", source)
        self.assertIn("RL_AD_SOURCE_CKPT", source)
        self.assertNotIn("run_block(\"rl_required\"", source)
        self.assertNotIn("run_block(\"core_required\"", source)

    def test_runner_reruns_finalization_when_checkpoint_reached_target(self):
        source = runner_source()

        self.assertIn("rerunning finalization from checkpoint", source)


if __name__ == "__main__":
    unittest.main()
