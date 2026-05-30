"""Attention layers used by the language model."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    """Plain causal multi-head attention."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = _split_heads(self.q_proj(x), self.n_heads)
        k = _split_heads(self.k_proj(x), self.n_heads)
        v = _split_heads(self.v_proj(x), self.n_heads)

        q, k, extra_bias = _prepare_position_terms(q, k, pos_bias)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if extra_bias is not None:
            scores = scores + extra_bias

        mask = _causal_mask(seq_len, x.device)
        y = _masked_softmax_attention(scores, v, mask, self.attn_drop)
        y = _merge_heads(y)
        return self.out_proj(y)


class SlidingWindowAttention(nn.Module):
    """Attention over a fixed recent window."""
    def __init__(self, d_model: int, n_heads: int, window_size: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = int(window_size)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor = None) -> torch.Tensor:
        _, seq_len, _ = x.shape
        q = _split_heads(self.q_proj(x), self.n_heads)
        k = _split_heads(self.k_proj(x), self.n_heads)
        v = _split_heads(self.v_proj(x), self.n_heads)

        q, k, extra_bias = _prepare_position_terms(q, k, pos_bias)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if extra_bias is not None:
            scores = scores + extra_bias

        mask = _causal_mask(seq_len, x.device) & _window_mask(seq_len, self.window_size, x.device)
        y = _masked_softmax_attention(scores, v, mask, self.attn_drop)
        y = _merge_heads(y)
        return self.out_proj(y)


class SparseBlockAttention(nn.Module):
    """Blockwise sparse attention."""
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = int(block_size)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor = None) -> torch.Tensor:
        _, seq_len, _ = x.shape
        q = _split_heads(self.q_proj(x), self.n_heads)
        k = _split_heads(self.k_proj(x), self.n_heads)
        v = _split_heads(self.v_proj(x), self.n_heads)

        q, k, extra_bias = _prepare_position_terms(q, k, pos_bias)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if extra_bias is not None:
            scores = scores + extra_bias

        mask = _sparse_block_mask(seq_len, self.block_size, x.device)
        y = _masked_softmax_attention(scores, v, mask, self.attn_drop)
        y = _merge_heads(y)
        return self.out_proj(y)


class LinearAttention(nn.Module):
    """Kernelized causal attention."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = int(kwargs.get("linear_chunk_size", 64))
        self.feature_eps = float(kwargs.get("linear_feature_eps", 1e-6))
        self.output_clip = float(kwargs.get("linear_output_clip", 64.0))
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor = None) -> torch.Tensor:
        q = _split_heads(self.q_proj(x), self.n_heads)
        k = _split_heads(self.k_proj(x), self.n_heads)
        v = _split_heads(self.v_proj(x), self.n_heads)

        if isinstance(pos_bias, dict):
            if pos_bias.get("alibi") is not None or pos_bias.get("relative") is not None:
                raise ValueError("LinearAttention supports RoPE only for positional terms.")
        q, k, _ = _prepare_position_terms(q, k, pos_bias)

        out_dtype = q.dtype
        q_phi = (F.elu(q.float()) + 1.0).clamp_min(self.feature_eps)
        k_phi = (F.elu(k.float()) + 1.0).clamp_min(self.feature_eps)
        q_phi = q_phi / q_phi.sum(dim=-1, keepdim=True).clamp_min(self.feature_eps)
        k_phi = k_phi / k_phi.sum(dim=-1, keepdim=True).clamp_min(self.feature_eps)
        v_float = v.float()

        y_chunks = []
        kv_state = torch.zeros(
            q_phi.size(0),
            q_phi.size(1),
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=torch.float32,
        )
        k_state = torch.zeros(
            q_phi.size(0),
            q_phi.size(1),
            self.head_dim,
            device=x.device,
            dtype=torch.float32,
        )

        # Prefix sums avoid the full attention matrix.
        for start in range(0, q_phi.size(2), self.chunk_size):
            end = min(start + self.chunk_size, q_phi.size(2))
            q_chunk = q_phi[:, :, start:end, :]
            k_chunk = k_phi[:, :, start:end, :]
            v_chunk = v_float[:, :, start:end, :]

            kv_chunk = torch.einsum("bhtd,bhte->bhtde", k_chunk, v_chunk)
            kv_prefix = kv_chunk.cumsum(dim=2) + kv_state.unsqueeze(2)
            k_prefix = k_chunk.cumsum(dim=2) + k_state.unsqueeze(2)

            numer = torch.einsum("bhtd,bhtde->bhte", q_chunk, kv_prefix)
            denom = torch.einsum("bhtd,bhtd->bht", q_chunk, k_prefix).unsqueeze(-1)
            y_chunks.append(numer / denom.clamp_min(self.feature_eps))

            kv_state = kv_prefix[:, :, -1, :, :]
            k_state = k_prefix[:, :, -1, :]

        y = torch.cat(y_chunks, dim=2)
        y = torch.nan_to_num(y, nan=0.0, posinf=self.output_clip, neginf=-self.output_clip)
        if self.output_clip > 0:
            y = y.clamp(min=-self.output_clip, max=self.output_clip)
        y = y.to(dtype=out_dtype)
        y = _merge_heads(y)
        y = self.out_drop(y)
        return self.out_proj(y)


class MultiQueryAttention(nn.Module):
    """One K/V head shared across all query heads."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = _split_heads(self.q_proj(x), self.n_heads)
        k = self.k_proj(x).view(bsz, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, 1, self.head_dim).transpose(1, 2)

        q, k, extra_bias = _prepare_position_terms(q, k, pos_bias)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if extra_bias is not None:
            scores = scores + extra_bias

        mask = _causal_mask(seq_len, x.device)
        y = _masked_softmax_attention(scores, v, mask, self.attn_drop)
        y = _merge_heads(y)
        return self.out_proj(y)


class GroupedQueryAttention(nn.Module):
    """Query heads grouped over shared K/V heads."""
    def __init__(self, d_model: int, n_heads: int, num_kv_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        if n_heads % num_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )

        self.n_heads = n_heads
        self.num_kv_heads = int(num_kv_heads)
        self.repeat_factor = n_heads // self.num_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = _split_heads(self.q_proj(x), self.n_heads)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.repeat_factor, dim=1)
        v = v.repeat_interleave(self.repeat_factor, dim=1)

        q, k, extra_bias = _prepare_position_terms(q, k, pos_bias)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if extra_bias is not None:
            scores = scores + extra_bias

        mask = _causal_mask(seq_len, x.device)
        y = _masked_softmax_attention(scores, v, mask, self.attn_drop)
        y = _merge_heads(y)
        return self.out_proj(y)


class SoftmaxFreeAttention(nn.Module):
    """ReLU-normalized attention."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor = None) -> torch.Tensor:
        _, seq_len, _ = x.shape
        q = _split_heads(self.q_proj(x), self.n_heads)
        k = _split_heads(self.k_proj(x), self.n_heads)
        v = _split_heads(self.v_proj(x), self.n_heads)

        q, k, extra_bias = _prepare_position_terms(q, k, pos_bias)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if extra_bias is not None:
            scores = scores + extra_bias

        mask = _causal_mask(seq_len, x.device)
        scores = scores.masked_fill(~mask[None, None, :, :], 0.0)
        weights = F.relu(scores)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.attn_drop(weights)

        y = torch.matmul(weights, v)
        y = _merge_heads(y)
        return self.out_proj(y)


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    bsz, seq_len, dim = x.shape
    head_dim = dim // n_heads
    return x.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    bsz, n_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(bsz, seq_len, n_heads * head_dim)


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()


def _window_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    dist = idx[:, None] - idx[None, :]
    return (dist >= 0) & (dist < window_size)


def _sparse_block_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(seq_len, device=device)
    q = idx[:, None]
    k = idx[None, :]
    causal = q >= k
    same_block = (q // block_size) == (k // block_size)
    global_tokens = (k % block_size) == 0
    return causal & (same_block | global_tokens)


def _prepare_position_terms(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_bias,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    extra_bias = None
    if isinstance(pos_bias, dict):
        rope = pos_bias.get("rope")
        if rope is not None:
            q, k = rope(q, k)

        alibi = pos_bias.get("alibi")
        if alibi is not None:
            extra_bias = alibi.unsqueeze(0) if alibi.dim() == 3 else alibi

        relative = pos_bias.get("relative")
        if relative is not None:
            rel_bias = relative(q, q.size(2))
            extra_bias = rel_bias if extra_bias is None else (extra_bias + rel_bias)

    elif torch.is_tensor(pos_bias):
        extra_bias = pos_bias.unsqueeze(0) if pos_bias.dim() == 3 else pos_bias

    return q, k, extra_bias


def _masked_softmax_attention(
    scores: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    attn_drop: nn.Dropout,
) -> torch.Tensor:
    scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = attn_drop(attn)
    return torch.matmul(attn, v)


# Config names map to attention classes.
ATTENTION_REGISTRY = {
    "full": FullAttention,
    "sliding_window": SlidingWindowAttention,
    "sparse_block": SparseBlockAttention,
    "linear": LinearAttention,
    "mqa": MultiQueryAttention,
    "gqa": GroupedQueryAttention,
    "softmax_free": SoftmaxFreeAttention,
}


def build_attention(cfg) -> nn.Module:
    """Build the attention layer from config."""
    attn_type = cfg.model.attention.type
    if attn_type == "aft":
        try:
            from .aft import build_aft
        except ImportError:
            from aft import build_aft
        return build_aft(cfg)
    cls = ATTENTION_REGISTRY[attn_type]
    return cls(
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        window_size=cfg.model.attention.window_size,
        block_size=cfg.model.attention.block_size,
        num_kv_heads=cfg.model.attention.num_kv_heads,
        linear_chunk_size=getattr(cfg.model.attention, "linear_chunk_size", 64),
        linear_feature_eps=getattr(cfg.model.attention, "linear_feature_eps", 1e-6),
        linear_output_clip=getattr(cfg.model.attention, "linear_output_clip", 64.0),
    )
