"""Shared helpers."""
import random
import time
import torch
import numpy as np


def set_seed(seed: int):
    """Seed the usual RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick CUDA when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def peak_memory_gb() -> float:
    """Read peak CUDA memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


class Timer:
    """Tiny timing context manager."""
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
