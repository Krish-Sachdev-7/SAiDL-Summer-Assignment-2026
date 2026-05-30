"""Conv blocks used in the Core ML hybrids."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal 1D conv for sequence blocks."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.left_pad = int(kernel_size) - 1
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=int(kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.transpose(1, 2)
        x = F.pad(x, (self.left_pad, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class PreAttentionConv(nn.Module):
    """Run a causal conv before attention."""
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.conv = CausalConv1d(d_model, d_model, kernel_size)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.conv(x)))


class InterleavedConvBlock(nn.Module):
    """Conv block for interleaved layers."""
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv1 = CausalConv1d(d_model, d_model, kernel_size)
        self.conv2 = CausalConv1d(d_model, d_model, kernel_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.conv1(h))
        h = self.drop(self.conv2(h))
        return x + h


class DepthwiseSeparableConv(nn.Module):
    """Cheaper depthwise conv block."""
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.left_pad = int(kernel_size) - 1
        self.depthwise = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=int(kernel_size),
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h = F.pad(h, (self.left_pad, 0))
        h = self.depthwise(h)
        h = self.pointwise(self.act(h))
        h = self.drop(h).transpose(1, 2)
        return x + h


class GatedConvFFN(nn.Module):
    """Gated conv feedforward block."""
    def __init__(self, d_model: int, ff_multiplier: int, kernel_size: int):
        super().__init__()
        ff_dim = int(d_model) * int(ff_multiplier)
        self.norm = nn.LayerNorm(d_model)
        self.conv = CausalConv1d(d_model, 2 * ff_dim, kernel_size)
        self.proj = nn.Linear(ff_dim, d_model)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.conv(h)
        gate, value = h.chunk(2, dim=-1)
        h = torch.sigmoid(gate) * value
        h = self.drop(self.proj(h))
        return x + h


CONV_REGISTRY = {
    "none": None,
    "pre_attention": PreAttentionConv,
    "interleaved": InterleavedConvBlock,
    "depthwise": DepthwiseSeparableConv,
    "gated_ffn": GatedConvFFN,
}
