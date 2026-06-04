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

    def test_runner_runs_only_stabilized_rl_bonus_suite(self):
        source = runner_source()

        expected_bonus_runs = [
            "rl_bonus_pos_learned_stable_seed",
            "rl_bonus_pos_sinusoidal_stable_seed",
            "rl_bonus_pos_rope_stable_seed",
            "rl_bonus_combined_L32_stable_seed",
            "rl_bonus_xlstm_hidden_stable_seed",
            "rl_bonus_xlstm_delayed_stable_seed",
        ]
        for name in expected_bonus_runs:
            with self.subTest(name=name):
                self.assertIn(name, source)

        self.assertIn("agent.lr=1e-4", source)
        self.assertIn("agent.exploration_noise=0.05", source)
        self.assertIn("agent.target_noise=0.1", source)
        self.assertIn("agent.target_noise_final=0.05", source)
        self.assertIn("agent.target_noise_decay_steps=300000", source)
        self.assertIn("agent.noise_clip=0.25", source)
        self.assertIn("agent.noise_clip_final=0.15", source)
        self.assertIn("agent.noise_clip_decay_steps=300000", source)
        self.assertIn("agent.tau=0.0025", source)
        self.assertIn("agent.grad_clip_norm=10.0", source)
        self.assertIn("agent.exploration_noise_final=0.01", source)
        self.assertIn("agent.exploration_noise_decay_steps=300000", source)
        self.assertIn("agent.critic_loss=huber", source)
        self.assertIn("agent.huber_beta=1.0", source)
        self.assertIn("agent.target_q_clip=5000.0", source)
        self.assertIn("agent.policy_delay=3", source)
        self.assertIn("agent.critic.type=transformer", source)
        self.assertIn("def _checkpoint_architecture_matches_cmd", source)
        self.assertIn("critic_type", source)
        self.assertIn("wanted == \"transformer\" and saved is None", source)
        self.assertIn("agent.actor.obs_norm=true", source)
        self.assertIn("logging.wandb.tags=[final,stability_compare,rl_bonus]", source)
        self.assertIn("algorithm_distillation.py collect", source)
        self.assertIn("algorithm_distillation.py train", source)
        self.assertIn("algorithm_distillation.py evaluate", source)
        self.assertIn("RL_AD_SOURCE_CKPT", source)
        self.assertIn("ad_transformer_stable.pt", source)
        self.assertIn("run_block(\"rl_bonus\"", source)
        self.assertNotIn("coreml_bonus_aft_full_ctx512", source)
        self.assertNotIn("run_block(\"core_bonus_aft\"", source)
        self.assertNotIn("run_block(\"rl_required\"", source)
        self.assertNotIn("run_block(\"core_required\"", source)

    def test_runner_reruns_finalization_when_checkpoint_reached_target(self):
        source = runner_source()

        self.assertIn("rerunning finalization from checkpoint", source)


if __name__ == "__main__":
    unittest.main()
