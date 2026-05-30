"""WikiText-2 loading helpers."""
import random

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer


def _token_limit(value) -> int | None:
    if value is None:
        return None
    limit = int(value)
    return limit if limit > 0 else None


class TokenChunkDataset(Dataset):
    """Fixed-length token chunk dataset."""
    def __init__(self, tokens: torch.Tensor, context_length: int, random_start: bool = True):
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D tensor.")
        self.tokens = tokens
        self.context_length = int(context_length)
        self.random_start = bool(random_start)
        self.max_start = max(0, len(tokens) - self.context_length - 1)

        if len(tokens) <= self.context_length:
            raise ValueError(
                f"Need at least context_length+1 tokens ({self.context_length + 1}), got {len(tokens)}."
            )

    def __len__(self) -> int:
        return self.max_start + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.random_start:
            start = random.randint(0, self.max_start)
        else:
            start = int(idx)

        x = self.tokens[start : start + self.context_length]
        y = self.tokens[start + 1 : start + self.context_length + 1]
        return x.long(), y.long()


def load_wikitext2(cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and tokenize WikiText-2."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    ds_train = load_dataset(cfg.data.dataset_name, cfg.data.dataset_config, split="train")
    ds_val = load_dataset(cfg.data.dataset_name, cfg.data.dataset_config, split="validation")

    train_text = "\n\n".join(ds_train["text"])
    val_text = "\n\n".join(ds_val["text"])

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    train_limit = _token_limit(getattr(cfg.data, "train_token_limit", None))
    val_limit = _token_limit(getattr(cfg.data, "val_token_limit", None))
    if train_limit is not None:
        train_ids = train_ids[:train_limit]
    if val_limit is not None:
        val_ids = val_ids[:val_limit]

    train_tokens = torch.tensor(train_ids, dtype=torch.long)
    val_tokens = torch.tensor(val_ids, dtype=torch.long)
    return train_tokens, val_tokens


def build_dataloaders(cfg) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders."""
    train_tokens, val_tokens = load_wikitext2(cfg)
    context = int(cfg.data.context_length)

    train_set = TokenChunkDataset(train_tokens, context_length=context, random_start=True)
    val_set = TokenChunkDataset(val_tokens, context_length=context, random_start=False)

    num_workers = int(cfg.trainer.num_workers)
    batch_size = int(cfg.trainer.batch_size)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader
