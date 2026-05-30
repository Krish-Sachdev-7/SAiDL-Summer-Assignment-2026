"""AFT variants kept around for the bonus path."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class AFTFull(nn.Module):
    """Full-bias AFT, chunked to keep memory sane."""
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.chunk_size = max(1, int(kwargs.get("aft_chunk_size", kwargs.get("chunk_size", 16))))
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.pos_bias = nn.Parameter(torch.zeros(self.max_seq_len, self.max_seq_len))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"AFTFull got seq_len={seq_len}, max_seq_len={self.max_seq_len}")

        q = torch.sigmoid(self.q_proj(x))
        k = self.k_proj(x)
        v = self.v_proj(x)
        out_dtype = q.dtype

        key_positions = torch.arange(seq_len, device=x.device)
        chunks = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            query_positions = torch.arange(start, end, device=x.device)
            causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            wb = self.pos_bias[start:end, :seq_len]

            def chunk_fn(q_chunk: torch.Tensor, k_full: torch.Tensor, v_full: torch.Tensor, wb_chunk: torch.Tensor) -> torch.Tensor:
                q_f = q_chunk.float()
                k_f = k_full.float()
                v_f = v_full.float()
                wb_f = wb_chunk.float()
                mask = causal.view(1, end - start, seq_len, 1)
                logits = wb_f.unsqueeze(0).unsqueeze(-1) + k_f.unsqueeze(1)
                logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                weights = torch.exp(logits - logits.max(dim=2, keepdim=True).values)
                weights = weights.masked_fill(~mask, 0.0)
                numer = (weights * v_f.unsqueeze(1)).sum(dim=2)
                denom = weights.sum(dim=2).clamp_min(1e-6)
                return q_f * (numer / denom)

            q_chunk = q[:, start:end]
            if torch.is_grad_enabled() and self.training:
                chunk = checkpoint(chunk_fn, q_chunk, k, v, wb, use_reentrant=False)
            else:
                chunk = chunk_fn(q_chunk, k, v, wb)
            chunks.append(chunk)

        out = torch.cat(chunks, dim=1).to(out_dtype)
        return self.out_proj(self.drop(out))


class AFTLocal(nn.Module):
    """Windowed AFT."""
    def __init__(self, d_model: int, max_seq_len: int, window_size: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.window_size = int(window_size)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.pos_bias = nn.Parameter(torch.zeros(self.max_seq_len, self.max_seq_len))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        q = torch.sigmoid(self.q_proj(x))
        k = self.k_proj(x)
        v = self.v_proj(x)

        idx = torch.arange(seq_len, device=x.device)
        dist = idx[:, None] - idx[None, :]
        local = (dist >= 0) & (dist < self.window_size)

        wb = self.pos_bias[:seq_len, :seq_len].masked_fill(~local, float("-inf"))
        logits = wb.unsqueeze(0).unsqueeze(-1) + k.unsqueeze(1)
        weights = torch.exp(logits - logits.max(dim=2, keepdim=True).values) * local.unsqueeze(0).unsqueeze(-1)
        numer = (weights * v.unsqueeze(1)).sum(dim=2)
        denom = weights.sum(dim=2).clamp_min(1e-6)
        out = q * (numer / denom)
        return self.out_proj(self.drop(out))


class AFTSimple(nn.Module):
    """AFT without position bias."""
    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.sigmoid(self.q_proj(x))
        k = self.k_proj(x)
        v = self.v_proj(x)
        out_dtype = q.dtype

        w = torch.exp(k.float())
        numer = torch.cumsum(w * v.float(), dim=1)
        denom = torch.cumsum(w, dim=1).clamp_min(1e-6)
        out = (q.float() * (numer / denom)).to(out_dtype)
        return self.out_proj(self.drop(out))


AFT_REGISTRY = {
    "full": AFTFull,
    "local": AFTLocal,
    "simple": AFTSimple,
}


def build_aft(cfg) -> nn.Module:
    """Build the requested AFT variant."""
    variant = cfg.model.attention.get("aft_variant", "full")
    cls = AFT_REGISTRY[variant]
    return cls(
        d_model=cfg.model.d_model,
        max_seq_len=cfg.model.max_seq_len,
        window_size=cfg.model.attention.window_size,
        dropout=cfg.model.dropout,
        aft_chunk_size=cfg.model.attention.get("aft_chunk_size", 16),
    )
