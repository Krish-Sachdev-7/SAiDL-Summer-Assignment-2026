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

    def forward(self, x: torch.Tensor, pos_bias=None) -> torch.Tensor:
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

            def chunk_fn(
                q_chunk: torch.Tensor,
                k_full: torch.Tensor,
                v_full: torch.Tensor,
                wb_chunk: torch.Tensor,
                causal_mask: torch.Tensor,
            ) -> torch.Tensor:
                q_f = q_chunk.float()
                k_f = k_full.float()
                v_f = v_full.float()
                wb_f = wb_chunk.float()
                mask = causal_mask.view(1, causal_mask.size(0), causal_mask.size(1), 1)
                logits = wb_f.unsqueeze(0).unsqueeze(-1) + k_f.unsqueeze(1)
                logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                weights = torch.exp(logits - logits.max(dim=2, keepdim=True).values)
                weights = weights.masked_fill(~mask, 0.0)
                numer = (weights * v_f.unsqueeze(1)).sum(dim=2)
                denom = weights.sum(dim=2).clamp_min(1e-6)
                return q_f * (numer / denom)

            q_chunk = q[:, start:end]
            if torch.is_grad_enabled() and self.training:
                chunk = checkpoint(chunk_fn, q_chunk, k, v, wb, causal, use_reentrant=False)
            else:
                chunk = chunk_fn(q_chunk, k, v, wb, causal)
            chunks.append(chunk)

        out = torch.cat(chunks, dim=1).to(out_dtype)
        return self.out_proj(self.drop(out))


class AFTLocal(nn.Module):
    """Windowed AFT."""
    def __init__(self, d_model: int, max_seq_len: int, window_size: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.window_size = int(window_size)
        self.chunk_size = max(1, int(kwargs.get("aft_chunk_size", kwargs.get("chunk_size", 16))))
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.pos_bias = nn.Parameter(torch.zeros(self.max_seq_len, self.max_seq_len))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias=None) -> torch.Tensor:
        _, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"AFTLocal got seq_len={seq_len}, max_seq_len={self.max_seq_len}")
        q = torch.sigmoid(self.q_proj(x))
        k = self.k_proj(x)
        v = self.v_proj(x)
        out_dtype = q.dtype

        chunks = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            key_start = max(0, start - self.window_size + 1)
            query_positions = torch.arange(start, end, device=x.device)
            key_positions = torch.arange(key_start, end, device=x.device)
            local = (
                (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
                & ((query_positions.unsqueeze(1) - key_positions.unsqueeze(0)) < self.window_size)
            )
            wb = self.pos_bias[start:end, key_start:end]
            q_chunk = q[:, start:end]
            k_local = k[:, key_start:end]
            v_local = v[:, key_start:end]

            def chunk_fn(
                q_part: torch.Tensor,
                k_part: torch.Tensor,
                v_part: torch.Tensor,
                wb_part: torch.Tensor,
                local_mask: torch.Tensor,
            ) -> torch.Tensor:
                q_f = q_part.float()
                k_f = k_part.float()
                v_f = v_part.float()
                wb_f = wb_part.float()
                mask = local_mask.view(1, local_mask.size(0), local_mask.size(1), 1)
                logits = wb_f.unsqueeze(0).unsqueeze(-1) + k_f.unsqueeze(1)
                logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                weights = torch.exp(logits - logits.max(dim=2, keepdim=True).values)
                weights = weights.masked_fill(~mask, 0.0)
                numer = (weights * v_f.unsqueeze(1)).sum(dim=2)
                denom = weights.sum(dim=2).clamp_min(1e-6)
                return q_f * (numer / denom)

            if torch.is_grad_enabled() and self.training:
                chunk = checkpoint(chunk_fn, q_chunk, k_local, v_local, wb, local, use_reentrant=False)
            else:
                chunk = chunk_fn(q_chunk, k_local, v_local, wb, local)
            chunks.append(chunk)

        out = torch.cat(chunks, dim=1).to(out_dtype)
        return self.out_proj(self.drop(out))


class AFTSimple(nn.Module):
    """AFT without position bias."""
    def __init__(self, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.max_seq_len = int(kwargs.get("max_seq_len", kwargs.get("seq_len", 0)) or 0)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias=None) -> torch.Tensor:
        if self.max_seq_len > 0 and x.size(1) > self.max_seq_len:
            raise ValueError(f"AFTSimple got seq_len={x.size(1)}, max_seq_len={self.max_seq_len}")
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
