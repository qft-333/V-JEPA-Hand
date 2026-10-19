"""Fixed-length main-model architecture and raw output layout."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor, nn

from .encoder import VJEPAEncoder
from .predictor import ViTPredictor


@dataclass(frozen=True)
class ModelConfig:
    num_frames: int = 16
    image_size: int = 384
    encoder_hidden_dim: int = 1024
    patch_size: int = 16
    tubelet_size: int = 2
    hidden_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6
    use_checkpoint: bool = False

    def __post_init__(self) -> None:
        dimensions = (
            self.num_frames, self.image_size, self.encoder_hidden_dim,
            self.patch_size, self.tubelet_size, self.hidden_dim, self.depth, self.num_heads,
        )
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("Model dimensions must be positive integers")
        if self.num_frames % self.tubelet_size or self.image_size % self.patch_size:
            raise ValueError("Configured frames and image size must match the encoder tokenization")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= self.drop_path_rate < 1 or self.mlp_ratio <= 0 or self.layer_norm_eps <= 0:
            raise ValueError("Invalid drop path, MLP ratio or normalization epsilon")


@dataclass(frozen=True)
class TargetSpec:
    per_hand_dim: int = field(default=55, init=False)
    per_frame_dim: int = field(default=119, init=False)

    def split(self, trajectory: Tensor) -> dict[str, Tensor]:
        if trajectory.ndim < 1 or trajectory.shape[-1] != self.per_frame_dim:
            raise ValueError("The last trajectory dimension must be 119")
        return {
            "eef_pose_left": trajectory[..., :9],
            "mano_left": trajectory[..., 9:54],
            "presence_left": trajectory[..., 54:55],
            "eef_pose_right": trajectory[..., 55:64],
            "mano_right": trajectory[..., 64:109],
            "presence_right": trajectory[..., 109:110],
            "camera": trajectory[..., 110:119],
        }


class HandPoseModel(nn.Module):
    def __init__(
        self, encoder: VJEPAEncoder, predictor: ViTPredictor, config: ModelConfig,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.config = config
        self.targets = TargetSpec()

    def forward(
        self, video: Tensor, intrinsic: Tensor | None = None,
        intrinsic_given: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        grid = self.encoder.token_grid(video)
        if video.shape[1] != self.config.num_frames:
            raise ValueError(f"Expected {self.config.num_frames} input frames")
        if not video.is_floating_point():
            raise ValueError("Video must be a normalized floating-point tensor")
        tokens = self.encoder(video)
        return self.predictor(tokens, grid, intrinsic, intrinsic_given)


def build_model(encoder_backend: nn.Module, config: ModelConfig | None = None) -> HandPoseModel:
    """Attach the released predictor to a caller-provided video encoder."""
    config = ModelConfig() if config is None else config
    encoder = VJEPAEncoder(
        encoder_backend, config.encoder_hidden_dim, config.patch_size, config.tubelet_size,
    )
    predictor = ViTPredictor(
        encoder_hidden_dim=config.encoder_hidden_dim,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        num_query_tokens=config.num_frames,
        mlp_ratio=config.mlp_ratio,
        drop_path_rate=config.drop_path_rate,
        qkv_bias=config.qkv_bias,
        layer_norm_eps=config.layer_norm_eps,
        use_checkpoint=config.use_checkpoint,
    )
    return HandPoseModel(encoder, predictor, config)
