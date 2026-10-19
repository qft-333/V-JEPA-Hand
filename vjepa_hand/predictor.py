"""Joint self-attention over video features, frame queries and an intrinsics token."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

Positions = tuple[Tensor, Tensor, Tensor]


class RoPE3D(nn.Module):
    """Apply separate rotary embeddings along time, height and width."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.seg = 2 * ((head_dim // 3) // 2)
        self.base = base

    def rotate(self, x: Tensor, positions: Positions) -> Tensor:
        if self.seg == 0:
            return x
        half = self.seg // 2
        inv_freq = 1.0 / (
            self.base ** (torch.arange(half, device=x.device, dtype=torch.float32) / half)
        )
        parts = []
        for axis, pos in enumerate(positions):
            part = x[..., axis * self.seg : (axis + 1) * self.seg]
            angles = pos.to(torch.float32).unsqueeze(-1) * inv_freq
            cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(x.dtype)[None, None]
            sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(x.dtype)[None, None]
            rotated = torch.cat((-part[..., half:], part[..., :half]), dim=-1)
            parts.append(part * cos + rotated * sin)
        parts.append(x[..., 3 * self.seg :])
        return torch.cat(parts, dim=-1)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: RoPE3D, positions: Positions) -> Tensor:
        batch, count, dim = x.shape
        qkv = self.qkv(x).reshape(batch, count, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rope.rotate(q, positions), rope.rotate(k, positions)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(x.transpose(1, 2).reshape(batch, count, dim))


class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


class Block(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float,
        qkv_bias: bool, drop_path: float, layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, rope: RoPE3D, positions: Positions) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), rope, positions))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ViTPredictor(nn.Module):
    """Main-model predictor; returns raw trajectory and normalized intrinsics."""

    def __init__(
        self,
        encoder_hidden_dim: int = 1024,
        hidden_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        num_query_tokens: int = 16,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        qkv_bias: bool = True,
        layer_norm_eps: float = 1e-6,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if min(encoder_hidden_dim, hidden_dim, depth, num_heads, num_query_tokens) < 1:
            raise ValueError("Dimensions, depth, head count and query count must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if mlp_ratio <= 0 or int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must give a positive hidden width")
        if not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0, 1)")
        self.encoder_hidden_dim = encoder_hidden_dim
        self.num_query_tokens = num_query_tokens
        self.use_checkpoint = use_checkpoint
        self.input_proj = (
            nn.Linear(encoder_hidden_dim, hidden_dim)
            if encoder_hidden_dim != hidden_dim else nn.Identity()
        )
        self.input_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.rope = RoPE3D(hidden_dim // num_heads)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, hidden_dim))
        self.intrinsic_embed = nn.Linear(4, hidden_dim)
        self.no_intrinsic_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.intrinsic_head = nn.Linear(hidden_dim, 4)
        rates = torch.linspace(0, drop_path_rate, depth, device="cpu").tolist()
        self.blocks = nn.ModuleList([
            Block(hidden_dim, num_heads, mlp_ratio, qkv_bias, rate, layer_norm_eps)
            for rate in rates
        ])
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.head = nn.Linear(hidden_dim, 119)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        nn.init.trunc_normal_(self.no_intrinsic_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        encoder_tokens: Tensor,
        token_grid: tuple[int, int, int],
        intrinsic: Tensor | None = None,
        intrinsic_given: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if encoder_tokens.ndim != 3 or encoder_tokens.shape[-1] != self.encoder_hidden_dim:
            raise ValueError("Expected encoder tokens with shape (B, N, encoder_hidden_dim)")
        if len(token_grid) != 3 or any(type(g) is not int or g < 1 for g in token_grid):
            raise ValueError("token_grid must contain three positive integers")
        batch, count, _ = encoder_tokens.shape
        time, height, width = token_grid
        if count != time * height * width:
            raise ValueError(f"Token count {count} does not match grid {token_grid}")
        if intrinsic is not None and intrinsic.shape != (batch, 4):
            raise ValueError("intrinsic must have shape (B, 4)")
        if intrinsic_given is not None:
            if intrinsic_given.shape != (batch,) or intrinsic_given.dtype != torch.bool:
                raise ValueError("intrinsic_given must be a boolean tensor with shape (B,)")
            if intrinsic is None and intrinsic_given.any().item():
                raise ValueError("Known intrinsics were requested without intrinsic values")

        x = self.input_proj(encoder_tokens)
        queries = self.query_tokens.expand(batch, -1, -1).to(x.dtype)
        no_intrinsic = self.no_intrinsic_token.expand(batch, 1, -1).to(x.dtype)
        intr_token = no_intrinsic
        if intrinsic is not None:
            embedded = self.intrinsic_embed(intrinsic.to(device=x.device, dtype=x.dtype))
            given = (
                torch.ones(batch, device=x.device, dtype=torch.bool)
                if intrinsic_given is None else intrinsic_given.to(x.device)
            )
            intr_token = torch.where(given[:, None, None], embedded[:, None], no_intrinsic)

        index = torch.arange(count, device=x.device)
        enc_pos = (
            (index // (height * width)).float(),
            ((index % (height * width)) // width).float(),
            (index % width).float(),
        )
        query_time = torch.arange(self.num_query_tokens, device=x.device).float()
        query_time = query_time * (time / self.num_query_tokens)
        zero_query = torch.zeros_like(query_time)
        query_pos = (query_time, zero_query, zero_query)
        positions = tuple(
            torch.cat((ep.new_zeros(1), ep, qp)) for ep, qp in zip(enc_pos, query_pos)
        )
        pos_t, pos_h, pos_w = positions
        all_pos = (pos_t, pos_h, pos_w)
        x = self.input_norm(torch.cat((intr_token, x, queries), dim=1))
        for block in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, self.rope, all_pos, use_reentrant=False)
            else:
                x = block(x, self.rope, all_pos)
        x = self.norm(x)
        return self.head(x[:, -self.num_query_tokens:]), self.intrinsic_head(x[:, 0])
