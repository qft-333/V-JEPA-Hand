"""Depth-tested bilateral mesh overlay with offscreen OpenGL."""

from __future__ import annotations

import os

import numpy as np


class OverlayRenderer:
    def __init__(self, width: int, height: int, faces: tuple[np.ndarray, np.ndarray]) -> None:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import pyrender

        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.faces = faces
        self.width, self.height = width, height

    def close(self) -> None:
        self.renderer.delete()

    def render(
        self, frame: np.ndarray, vertices: np.ndarray,
        presence: np.ndarray, intrinsics: np.ndarray, threshold: float,
    ) -> tuple[np.ndarray, int, int]:
        import pyrender
        import trimesh

        fx, fy = intrinsics[:2] * [self.width, self.height]
        if not np.isfinite(intrinsics).all() or min(fx, fy) <= 0:
            raise ValueError("The model predicted invalid camera intrinsics")
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.38] * 3)
        # The training/rendering convention fixes the principal point at image centre.
        scene.add(pyrender.IntrinsicsCamera(fx, fy, self.width / 2, self.height / 2))
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0))
        count = 0
        for side, color in enumerate(((0.025, 0.23, 0.82, 1.0), (0.95, 0.20, 0.025, 1.0))):
            if presence[side] < threshold:
                continue
            mesh = trimesh.Trimesh(
                vertices[side] * [1, -1, -1], self.faces[side], process=False,
            )
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.65,
                alphaMode="OPAQUE",
            )
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
            count += 1
        rgb, depth = self.renderer.render(scene)
        mask = depth > 0
        output = frame.copy()
        output[mask] = np.clip(0.95 * rgb[mask] + 0.05 * frame[mask], 0, 255).astype(np.uint8)
        return output, count, int(mask.sum())
