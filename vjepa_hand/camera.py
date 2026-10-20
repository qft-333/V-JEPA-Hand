"""Optional calibrated-image rectification for the centered-camera convention."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

CAMERA_HELP = (
    "Optional: enter fx, fy, cx, cy in PIXELS at the uploaded video's resolution "
    "(for example: 600, 610, 320, 240 for a calibrated 640 x 480 video). "
    "Do not enter normalized values fx/W or fy/H. Use calibration for the actual "
    "uploaded video, after any previous resizing/cropping. With calibration, we "
    "resample to equal pixel focal lengths, crop around the principal point, and "
    "use the corrected intrinsics for inference and rendering. Output shows the "
    "corrected crop, so edge content may be removed. Leave blank to predict focal "
    "lengths and assume a centered principal point; this does NOT correct an "
    "unknown off-center principal point. Input must already be lens-undistorted; "
    "fisheye distortion and time-varying calibration are not handled."
)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("Camera intrinsics must be finite numbers in pixels")
        if min(self.fx, self.fy) <= 0:
            raise ValueError("Camera focal lengths fx and fy must be positive")

    @classmethod
    def parse(cls, text: str | None) -> CameraIntrinsics | None:
        if text is None or not text.strip():
            return None
        parts = text.replace(",", " ").split()
        if len(parts) != 4:
            raise ValueError("Enter all four camera intrinsics: fx, fy, cx, cy (in pixels)")
        try:
            fx, fy, cx, cy = (float(value) for value in parts)
        except ValueError as error:
            raise ValueError("Camera intrinsics must be four numbers in pixels") from error
        return cls(fx, fy, cx, cy)


class CameraRectification:
    def __init__(self, camera: CameraIntrinsics, width: int, height: int) -> None:
        if not (0 < camera.cx < width and 0 < camera.cy < height):
            raise ValueError(
                f"Principal point must lie inside the {width} x {height} input video; "
                "provide cx, cy in pixels at this resolution"
            )
        self.source_shape = (height, width, 3)
        focal = min(camera.fx, camera.fy)
        sx, sy = focal / camera.fx, focal / camera.fy
        self.width = 2 * math.floor(min(camera.cx, width - camera.cx) * sx)
        self.height = 2 * math.floor(min(camera.cy, height - camera.cy) * sy)
        if min(self.width, self.height) < 2:
            raise ValueError("Camera calibration leaves no usable centered crop (at least 2 x 2)")
        self.intrinsics = np.array(
            [focal / self.width, focal / self.height, 0.5, 0.5], dtype=np.float32,
        )
        if not np.isfinite(self.intrinsics).all() or (self.intrinsics[:2] <= 0).any():
            raise ValueError("Corrected camera focal lengths are outside the supported numeric range")
        # Combine anisotropic resize and subpixel-centered crop in one resampling.
        x = (torch.arange(self.width, dtype=torch.float32) - self.width / 2) / sx + camera.cx
        y = (torch.arange(self.height, dtype=torch.float32) - self.height / 2) / sy + camera.cy
        yy, xx = torch.meshgrid(2 * (y + 0.5) / height - 1, 2 * (x + 0.5) / width - 1, indexing="ij")
        self.grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        self.identity = (
            camera.fx == camera.fy and camera.cx == width / 2 and camera.cy == height / 2
            and self.width == width and self.height == height
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.shape != self.source_shape or image.dtype != np.uint8:
            raise ValueError("Rectification requires RGB uint8 frames at the calibrated resolution")
        if self.identity:
            return image
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        corrected = F.grid_sample(tensor, self.grid, mode="bilinear", align_corners=False)
        return corrected[0].permute(1, 2, 0).round().clamp_(0, 255).byte().numpy()
