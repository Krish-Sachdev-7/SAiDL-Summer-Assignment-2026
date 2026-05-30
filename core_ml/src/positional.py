"""Position encoding options."""
import math

import torch
import torch.nn as nn


class AbsolutePositionalEncoding(nn.Module):
    """Learned absolute positions."""
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.pos_emb = nn.Embedding(self.max_seq_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Forward pass."""
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Requested seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})."
            )
        positions = torch.arange(seq_len, device=device)
        return self.pos_emb(positions).unsqueeze(0)


class RoPE(nn.Module):
    """RoPE for Q/K."""
    def __init__(self, head_dim: int, max_seq_len: int = 4096, scale_factor: float = 1.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}.")
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.scale_factor = float(scale_factor)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        seq_len = q.size(2)
        if seq_len > self.max_seq_len:
            # Long eval contexts can exceed the trained window.
            # Warn instead of failing during extrapolation probes.
            pass

        positions = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        positions = positions / max(self.scale_factor, 1e-6)
        freqs = torch.einsum("t,d->td", positions, self.inv_freq)

        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)[None, None, :, :].to(dtype=q.dtype)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)[None, None, :, :].to(dtype=q.dtype)

        q_out = (q * cos) + (_rotate_half(q) * sin)
        k_out = (k * cos) + (_rotate_half(k) * sin)
        return q_out, k_out


class ALiBi(nn.Module):
    """ALiBi attention bias."""
    def __init__(self, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = int(n_heads)
        self.max_seq_len = int(max_seq_len)
        slopes = self._build_slopes(self.n_heads)
        self.register_buffer("slopes", slopes.view(self.n_heads, 1, 1), persistent=False)

    def get_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return ALiBi score bias."""
        idx = torch.arange(seq_len, device=device)
        dist = (idx[:, None] - idx[None, :]).clamp(min=0).float()
        return -self.slopes.to(device) * dist

    @staticmethod
    def _build_slopes(n_heads: int) -> torch.Tensor:
        # Different heads get different distance slopes.
        heads = torch.arange(1, n_heads + 1, dtype=torch.float32)
        return 1.0 / torch.pow(2.0, (8.0 * (heads - 1.0) / n_heads))


class RelativePositionalEncoding(nn.Module):
    """Learned relative position bias."""
    def __init__(self, head_dim: int, max_relative_positions: int = 128):
        super().__init__()
        self.head_dim = int(head_dim)
        self.max_relative_positions = int(max_relative_positions)
        n_rel = 2 * self.max_relative_positions + 1
        self.rel_embedding = nn.Embedding(n_rel, self.head_dim)

    def forward(self, q: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Forward pass."""
        pos = torch.arange(seq_len, device=q.device)
        rel = pos[:, None] - pos[None, :]
        rel = rel.clamp(-self.max_relative_positions, self.max_relative_positions)
        rel_idx = rel + self.max_relative_positions

        rel_emb = self.rel_embedding(rel_idx)
        return torch.einsum("bhtd,tjd->bhtj", q, rel_emb)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


PE_REGISTRY = {
    "absolute": AbsolutePositionalEncoding,
    "rope": RoPE,
    "alibi": ALiBi,
    "relative": RelativePositionalEncoding,
}
