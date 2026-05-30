"""TD3 network modules."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPActor(nn.Module):
    """Basic MLP actor."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int], max_action: float = 1.0):
        super().__init__()
        dims = [obs_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], act_dim)
        self.max_action = float(max_action)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone(obs)
        return torch.tanh(self.head(h)) * self.max_action


class MLPCritic(nn.Module):
    """Basic MLP critic."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [obs_dim + act_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        h = self.backbone(x)
        return self.head(h)


class _CausalTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.n_heads = int(n_heads)
        self.head_dim = embed_dim // self.n_heads

    def forward(self, x: torch.Tensor, rope_fn=None):
        bsz, seq_len, dim = x.shape
        h = self.ln1(x)

        q = self.q_proj(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if rope_fn is not None:
            q, k = rope_fn(q, k)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        x = x + self.o_proj(y)
        x = x + self.ff(self.ln2(x))
        return x, attn


class TransformerActor(nn.Module):
    """Transformer actor over recent history."""
    def __init__(self, obs_dim: int, act_dim: int, embed_dim: int, n_layers: int,
                 n_heads: int, context_length: int, max_action: float = 1.0,
                 pos_encoding: str = "learned"):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.embed_dim = int(embed_dim)
        self.context_length = int(context_length)
        self.max_action = float(max_action)
        self.pos_encoding = str(pos_encoding)

        self.input_proj = nn.Linear(self.obs_dim + self.act_dim, self.embed_dim)
        self.pos_emb = nn.Embedding(self.context_length, self.embed_dim)

        self.blocks = nn.ModuleList(
            [_CausalTransformerBlock(self.embed_dim, n_heads) for _ in range(int(n_layers))]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.act_dim)
        self.last_attn_weights = []

        if self.pos_encoding == "rope":
            if (self.embed_dim // n_heads) % 2 != 0:
                raise ValueError("head_dim must be even to use RoPE.")
            inv_freq = 1.0 / (
                10000 ** (torch.arange(0, self.embed_dim // n_heads, 2).float() / (self.embed_dim // n_heads))
            )
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        else:
            self.register_buffer("rope_inv_freq", torch.empty(0), persistent=False)

        if self.pos_encoding == "sinusoidal":
            self.register_buffer("sin_table", self._build_sinusoidal_table(self.context_length, self.embed_dim), persistent=False)
        else:
            self.register_buffer("sin_table", torch.empty(0), persistent=False)

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        bsz, seq_len, _ = obs_seq.shape
        x = torch.cat([obs_seq, act_seq], dim=-1)
        x = self.input_proj(x)

        if self.pos_encoding == "learned":
            pos = torch.arange(seq_len, device=x.device)
            x = x + self.pos_emb(pos).unsqueeze(0)
        elif self.pos_encoding == "sinusoidal":
            x = x + self.sin_table[:seq_len].unsqueeze(0).to(x.dtype)

        self.last_attn_weights = []
        rope_fn = self._apply_rope if self.pos_encoding == "rope" else None
        for block in self.blocks:
            x, attn = block(x, rope_fn=rope_fn)
            self.last_attn_weights.append(attn)

        x = self.ln_f(x)
        action = torch.tanh(self.head(x[:, -1])) * self.max_action
        return action

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.size(2)
        positions = torch.arange(seq_len, device=q.device, dtype=self.rope_inv_freq.dtype)
        freqs = torch.einsum("t,d->td", positions, self.rope_inv_freq)
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)[None, None, :, :].to(q.dtype)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)[None, None, :, :].to(q.dtype)
        return self._rope_mix(q, cos, sin), self._rope_mix(k, cos, sin)

    @staticmethod
    def _rope_mix(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rot = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
        return x * cos + rot * sin

    @staticmethod
    def _build_sinusoidal_table(length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe
