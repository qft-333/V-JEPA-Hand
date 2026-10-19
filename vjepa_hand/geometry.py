"""Decode the main model's neutral-shape hand parameters using licensed MANO assets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.nn import functional as F


def rotation_6d(values: Tensor) -> Tensor:
    first = F.normalize(values[..., :3], dim=-1, eps=1e-8)
    second = values[..., 3:]
    second = F.normalize(second - (first * second).sum(-1, keepdim=True) * first, dim=-1, eps=1e-8)
    return torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-2)


def boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    counts = Counter(tuple(sorted(edge)) for edge in edges)
    boundary = [tuple(edge) for edge in edges if counts[tuple(sorted(edge))] == 1]
    following = dict(boundary)
    if len(following) != len(boundary):
        raise ValueError("MANO mesh boundary is not manifold")
    loops = []
    while following:
        start = current = next(iter(following))
        loop = []
        while True:
            loop.append(current)
            if current not in following:
                raise ValueError("MANO mesh boundary is not closed")
            current = following.pop(current)
            if current == start:
                break
        loops.append(np.asarray(loop))
    return loops


class HandMeshes:
    def __init__(self, mano_dir: str) -> None:
        path = Path(mano_dir).expanduser().resolve() / "MANO_RIGHT.pkl"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Obtain MANO from https://mano.is.tue.mpg.de/ "
                "under its license; model assets are not bundled."
            )
        try:
            self.layer = smplx.MANOLayer(model_path=str(path), is_rhand=True).eval()
        except ImportError as error:
            raise RuntimeError(
                "Could not load MANO dependencies. For a legacy Chumpy asset, "
                "run tools/prepare_mano.py in the conversion environment documented in README.md."
            ) from error
        self.layer.requires_grad_(False)
        self.loops = boundary_loops(self.layer.faces)
        cap_faces = [
            [int(loop[(i + 1) % len(loop)]), int(vertex), 778 + offset]
            for offset, loop in enumerate(self.loops) for i, vertex in enumerate(loop)
        ]
        right = np.concatenate((self.layer.faces, np.asarray(cap_faces, dtype=np.int64).reshape(-1, 3)))
        self.faces = (right[:, [0, 2, 1]], right)

    @torch.inference_mode()
    def vertices(self, trajectory: Tensor) -> np.ndarray:
        if trajectory.ndim != 2 or trajectory.shape[-1] != 119:
            raise ValueError("Expected trajectory (T, 119)")
        trajectory = trajectory.detach().float().cpu()
        if not torch.isfinite(trajectory).all():
            raise ValueError("Model produced non-finite trajectory values")
        hands = torch.stack((trajectory[:, :55], trajectory[:, 55:110]), dim=1)
        flat = hands.reshape(-1, 55)
        angles = flat[:, 9:54].numpy().reshape(-1, 3)
        poses = torch.from_numpy(Rotation.from_euler("xyz", angles).as_matrix()).float()
        batch = flat.shape[0]
        result = self.layer(
            global_orient=torch.eye(3).reshape(1, 1, 3, 3).expand(batch, -1, -1, -1),
            hand_pose=poses.reshape(batch, 15, 3, 3),
            betas=torch.zeros(batch, 10),
        )
        local = result.vertices - result.joints[:, :1]
        local[0::2, :, 0] *= -1
        vertices = torch.einsum("nij,nkj->nki", rotation_6d(flat[:, 3:9]), local)
        vertices = (vertices + flat[:, None, :3]).reshape(-1, 2, 778, 3).numpy()
        if self.loops:
            centers = np.stack([vertices[:, :, loop].mean(axis=2) for loop in self.loops], axis=2)
            vertices = np.concatenate((vertices, centers), axis=2)
        return vertices
