import sys
import unittest
import importlib.util
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core_ml" / "src"))

from model import DecoderLM

_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "core_ml_train_for_tests",
    ROOT / "core_ml" / "src" / "train.py",
)
core_ml_train = importlib.util.module_from_spec(_TRAIN_SPEC)
assert _TRAIN_SPEC.loader is not None
_TRAIN_SPEC.loader.exec_module(core_ml_train)
_filter_extrapolation_lengths = core_ml_train._filter_extrapolation_lengths


def tiny_aft_cfg(variant: str):
    base = OmegaConf.load(ROOT / "core_ml" / "conf" / "config.yaml")
    model = OmegaConf.load(ROOT / "core_ml" / "conf" / "model" / "aft.yaml")
    pos = OmegaConf.load(ROOT / "core_ml" / "conf" / "pos_encoding" / "absolute.yaml")
    conv = OmegaConf.load(ROOT / "core_ml" / "conf" / "conv" / "none.yaml")
    data = OmegaConf.load(ROOT / "core_ml" / "conf" / "data" / "wikitext2.yaml")
    trainer = OmegaConf.load(ROOT / "core_ml" / "conf" / "trainer" / "base.yaml")
    cfg = OmegaConf.merge(
        base,
        {
            "model": model,
            "pos_encoding": pos,
            "conv": conv,
            "data": data,
            "trainer": trainer,
        },
    )
    cfg.model.vocab_size = 97
    cfg.model.d_model = 32
    cfg.model.n_heads = 4
    cfg.model.n_layers = 2
    cfg.model.max_seq_len = 16
    cfg.model.dropout = 0.0
    cfg.model.attention.aft_variant = variant
    cfg.model.attention.aft_chunk_size = 4
    cfg.model.attention.window_size = 5
    cfg.data.context_length = 16
    return cfg


class CoreMLAFTTests(unittest.TestCase):
    def test_all_aft_variants_run_forward_backward_inside_decoder(self):
        for variant in ["full", "local", "simple"]:
            with self.subTest(variant=variant):
                torch.manual_seed(1234)
                model = DecoderLM(tiny_aft_cfg(variant))
                idx = torch.randint(0, 97, (2, 16))

                logits = model(idx)
                loss = logits.float().pow(2).mean()
                loss.backward()

                self.assertEqual(tuple(logits.shape), (2, 16, 97))
                self.assertTrue(torch.isfinite(logits).all().item())
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        self.assertTrue(torch.isfinite(param.grad).all().item())
                        grad_norm += float(param.grad.detach().abs().sum().item())
                self.assertGreater(grad_norm, 0.0)

    def test_aft_rejects_sequences_longer_than_configured_context(self):
        model = DecoderLM(tiny_aft_cfg("full"))
        idx = torch.randint(0, 97, (1, 17))

        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            model(idx)

    def test_aft_absolute_extrapolation_skips_lengths_past_max_seq_len(self):
        cfg = tiny_aft_cfg("full")
        valid_lengths, skipped_lengths = _filter_extrapolation_lengths(cfg, [16, 32, 64])

        self.assertEqual(valid_lengths, [16])
        self.assertEqual(skipped_lengths, [32, 64])


if __name__ == "__main__":
    unittest.main()
