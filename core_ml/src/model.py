"""Small decoder-only LM."""
import torch
import torch.nn as nn

try:
    from .attention import build_attention
    from .positional import ALiBi, AbsolutePositionalEncoding, RelativePositionalEncoding, RoPE
    from .conv_blocks import CONV_REGISTRY
except ImportError:
    from attention import build_attention
    from positional import ALiBi, AbsolutePositionalEncoding, RelativePositionalEncoding, RoPE
    from conv_blocks import CONV_REGISTRY


class TransformerBlock(nn.Module):
    """One configurable transformer block."""
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        d_model = int(cfg.model.d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.resid_drop = nn.Dropout(float(cfg.model.dropout))

        self.conv_type = str(cfg.conv.type)
        kernel_size = int(cfg.conv.kernel_size)

        self.pre_attention_conv = None
        self.interleaved_conv = None
        self.depthwise_conv = None

        if self.conv_type == "pre_attention":
            self.pre_attention_conv = CONV_REGISTRY["pre_attention"](
                d_model=d_model,
                kernel_size=kernel_size,
            )

        self.use_attention = True
        if self.conv_type == "interleaved" and self.layer_idx % 2 == 0:
            self.interleaved_conv = CONV_REGISTRY["interleaved"](
                d_model=d_model,
                kernel_size=kernel_size,
            )
            self.use_attention = False

        if self.conv_type == "depthwise" and self.layer_idx % 2 == 0:
            self.depthwise_conv = CONV_REGISTRY["depthwise"](
                d_model=d_model,
                kernel_size=kernel_size,
            )
            self.use_attention = False

        self.attn = build_attention(cfg) if self.use_attention else None

        head_dim = d_model // int(cfg.model.n_heads)
        self.pos_type = str(cfg.pos_encoding.type)
        self.rope = None
        self.alibi = None
        self.relative = None

        if self.pos_type == "rope":
            scale_factor = 1.0
            if "interpolation" in cfg.pos_encoding and bool(cfg.pos_encoding.interpolation.enabled):
                scale_factor = float(cfg.pos_encoding.interpolation.scale_factor)
            self.rope = RoPE(
                head_dim=head_dim,
                max_seq_len=int(cfg.model.max_seq_len),
                scale_factor=scale_factor,
            )
        elif self.pos_type == "alibi":
            self.alibi = ALiBi(
                n_heads=int(cfg.model.n_heads),
                max_seq_len=int(cfg.model.max_seq_len),
            )
        elif self.pos_type == "relative":
            self.relative = RelativePositionalEncoding(
                head_dim=head_dim,
                max_relative_positions=int(cfg.pos_encoding.max_relative_positions),
            )

        ff_hidden = d_model * int(cfg.model.ff_multiplier)
        if self.conv_type == "gated_ffn":
            self.ff = CONV_REGISTRY["gated_ffn"](
                d_model=d_model,
                ff_multiplier=int(cfg.model.ff_multiplier),
                kernel_size=kernel_size,
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, ff_hidden),
                nn.GELU(),
                nn.Dropout(float(cfg.model.dropout)),
                nn.Linear(ff_hidden, d_model),
                nn.Dropout(float(cfg.model.dropout)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)

        if self.pre_attention_conv is not None:
            h = self.pre_attention_conv(h)

        if self.use_attention:
            pos_terms = {}
            if self.rope is not None:
                pos_terms["rope"] = self.rope
            if self.alibi is not None:
                pos_terms["alibi"] = self.alibi.get_bias(h.size(1), h.device)
            if self.relative is not None:
                pos_terms["relative"] = self.relative
            if not pos_terms:
                pos_terms = None

            attn_out = self.attn(h, pos_bias=pos_terms)
        elif self.interleaved_conv is not None:
            attn_out = self.interleaved_conv(h)
        else:
            attn_out = self.depthwise_conv(h)

        x = x + self.resid_drop(attn_out)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderLM(nn.Module):
    """Tiny GPT-style decoder."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.model.d_model)
        vocab_size = int(cfg.model.vocab_size)
        max_seq_len = int(cfg.model.max_seq_len)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_type = str(cfg.pos_encoding.type)

        self.abs_pos = None
        if self.pos_type == "absolute":
            self.abs_pos = AbsolutePositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)

        self.drop = nn.Dropout(float(cfg.model.dropout))
        self.blocks = nn.ModuleList([TransformerBlock(cfg, layer_idx=i) for i in range(int(cfg.model.n_layers))])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

        # Tie embeddings and logits; it helps at this scale.
        self.head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small GPT-style init."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        _, seq_len = idx.shape
        x = self.tok_emb(idx)

        if self.abs_pos is not None:
            x = x + self.abs_pos(seq_len, idx.device)

        x = self.drop(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)
