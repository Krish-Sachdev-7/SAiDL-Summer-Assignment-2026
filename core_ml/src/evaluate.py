"""Evaluation helpers."""
import math
from copy import deepcopy

import torch
import torch.nn.functional as F

try:
    from .data import build_dataloaders
except ImportError:
    from data import build_dataloaders


def _scaled_extrapolation_batch_size(cfg, eval_context_length: int) -> int:
    """Shrink eval batch size for long contexts."""
    base_context = max(1, int(cfg.data.context_length))
    base_batch = max(1, int(cfg.trainer.batch_size))
    eval_context = max(1, int(eval_context_length))
    scaled = (base_batch * base_context) // eval_context
    return max(1, min(base_batch, int(scaled)))


def evaluate_perplexity(model, loader, device, max_batches: int, use_amp: bool = False) -> tuple[float, float]:
    """Return validation loss and perplexity."""
    model.eval()
    losses = []
    it = iter(loader)
    amp_enabled = bool(use_amp) and device.type == "cuda"
    with torch.no_grad():
        for _ in range(max_batches):
            try:
                x, y = next(it)
            except StopIteration:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(float(loss.item()))

    if not losses:
        return float("nan"), float("nan")

    val_loss = sum(losses) / len(losses)
    val_ppl = math.exp(min(20.0, val_loss))
    return val_loss, val_ppl


def extrapolation_test(model, cfg, eval_lengths: list[int], device) -> dict:
    """Run the context extrapolation checks."""
    results = {}
    base_batch = int(cfg.trainer.batch_size)
    use_amp = bool(getattr(cfg.trainer, "use_amp", False))
    for length in eval_lengths:
        test_cfg = deepcopy(cfg)
        test_cfg.data.context_length = int(length)
        test_cfg.trainer.batch_size = _scaled_extrapolation_batch_size(cfg, int(length))
        if int(test_cfg.trainer.batch_size) != base_batch:
            print(
                f"[extrapolation] context={length}: eval batch_size "
                f"{base_batch} -> {int(test_cfg.trainer.batch_size)}"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _, val_loader = build_dataloaders(test_cfg)
        _, ppl = evaluate_perplexity(
            model,
            val_loader,
            device,
            max_batches=int(cfg.trainer.eval_batches),
            use_amp=use_amp,
        )
        results[int(length)] = float(ppl)
    return results


def measure_throughput(
    model,
    context_length: int,
    batch_size: int,
    device,
    n_batches: int = 10,
    use_amp: bool = False,
) -> float:
    """Measure inference tokens per second."""
    model.eval()
    amp_enabled = bool(use_amp) and device.type == "cuda"
    x = torch.randint(
        low=0,
        high=model.cfg.model.vocab_size,
        size=(batch_size, context_length),
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if device.type == "cuda":
        start.record()
    else:
        import time
        t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(n_batches):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                _ = model(x)

    if device.type == "cuda":
        end.record()
        torch.cuda.synchronize()
        elapsed_sec = start.elapsed_time(end) / 1000.0
    else:
        import time
        elapsed_sec = time.perf_counter() - t0

    tokens = n_batches * batch_size * context_length
    return float(tokens / max(elapsed_sec, 1e-8))


def measure_peak_memory() -> float:
    """Return peak GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0
