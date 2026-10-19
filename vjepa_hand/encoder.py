"""Adapter for an externally supplied official V-JEPA 2.1 encoder."""

from __future__ import annotations

from torch import Tensor, nn


class VJEPAEncoder(nn.Module):
    """Wrap a backend taking (B, C, T, H, W) and returning (B, N, D)."""

    def __init__(
        self, backend: nn.Module, hidden_dim: int = 1024,
        patch_size: int = 16, tubelet_size: int = 2,
    ) -> None:
        super().__init__()
        if min(hidden_dim, patch_size, tubelet_size) < 1:
            raise ValueError("Encoder dimensions must be positive")
        self.backend = backend
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

    def token_grid(self, video: Tensor) -> tuple[int, int, int]:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError("Expected video shape (B, T, 3, H, W)")
        _, frames, _, height, width = video.shape
        if min(frames, height, width) < 1:
            raise ValueError("Video dimensions must be positive")
        if frames % self.tubelet_size or height % self.patch_size or width % self.patch_size:
            raise ValueError("Video dimensions must be divisible by the tubelet and patch sizes")
        return frames // self.tubelet_size, height // self.patch_size, width // self.patch_size

    def forward(self, video: Tensor) -> Tensor:
        time, height, width = self.token_grid(video)
        tokens = self.backend(video.permute(0, 2, 1, 3, 4).contiguous())
        if not isinstance(tokens, Tensor):
            raise TypeError("Encoder backend must return a single patch-token tensor")
        expected = (video.shape[0], time * height * width, self.hidden_dim)
        if tokens.shape != expected:
            raise ValueError(f"Expected encoder output {expected}, got {tuple(tokens.shape)}")
        return tokens
