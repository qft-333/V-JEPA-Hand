"""Stream an input video through Ours and write a silent hand-mesh overlay."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .geometry import HandMeshes
from .loading import DEFAULT_MODEL_ID, load_model
from .model import HandPoseModel
from .rendering import OverlayRenderer

LOGGER = logging.getLogger(__name__)


def preprocess(frames: np.ndarray) -> Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 frames with shape (T, H, W, 3)")
    height, width = frames.shape[1:3]
    target_width = max(16, round(384 * width / height / 16) * 16)
    video = torch.from_numpy(frames).float().div_(255).permute(0, 3, 1, 2)
    if (height, width) != (384, target_width):
        video = F.interpolate(video, size=(384, target_width), mode="bilinear", align_corners=False)
    mean = video.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = video.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((video - mean) / std).unsqueeze(0)


@torch.inference_mode()
def predict_parameters(
    model: HandPoseModel, frames: np.ndarray, device: torch.device, precision: str,
) -> tuple[Tensor, Tensor]:
    count = len(frames)
    if not 1 <= count <= 16:
        raise ValueError("A window must contain between 1 and 16 real frames")
    if precision not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    padded = np.concatenate((frames, np.repeat(frames[-1:], 16 - count, axis=0)))
    video = preprocess(padded).to(device)
    context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if precision == "bf16" else nullcontext()
    )
    with context:
        trajectory, intrinsics = model(video)
    trajectory = trajectory[0, :count].float().cpu()
    intrinsics = intrinsics[0].float().cpu()
    if not torch.isfinite(trajectory).all() or not torch.isfinite(intrinsics).all():
        raise ValueError("Inference returned non-finite outputs")
    if (intrinsics[:2] <= 0).any():
        raise ValueError("Inference returned non-positive focal lengths")
    return trajectory, intrinsics


class VideoDemo:
    def __init__(
        self, model: HandPoseModel, mano_dir: str, precision: str = "bf16",
        presence_threshold: float = 0.5,
    ) -> None:
        if precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        if not math.isfinite(presence_threshold) or not 0 <= presence_threshold <= 1:
            raise ValueError("presence threshold must be in [0, 1]")
        self.model = model
        self.device = next(model.parameters()).device
        if precision == "bf16" and self.device.type != "cuda":
            raise ValueError("BF16 demo inference requires CUDA; use --precision fp32 on CPU")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16; use --precision fp32")
        self.precision = precision
        self.threshold = presence_threshold
        self.meshes = HandMeshes(mano_dir)

    @torch.inference_mode()
    def predict(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        trajectory, intrinsics = predict_parameters(
            self.model, frames, self.device, self.precision,
        )
        presence = trajectory[:, [54, 109]].sigmoid().numpy()
        return self.meshes.vertices(trajectory), presence, intrinsics.numpy()

    def run(
        self, input_video: str, output_video: str,
        progress: Callable[[int], None] | None = None,
        max_frames: int | None = None,
    ) -> dict[str, int | float | str]:
        source, destination = Path(input_video).resolve(), Path(output_video).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination == source:
            raise ValueError("Input and output must be different files")
        if destination.suffix.lower() != ".mp4":
            raise ValueError("Output must have an .mp4 extension")
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".vjepa-", suffix=".mp4", dir=destination.parent)
        os.close(descriptor)
        temp_path = Path(temporary)
        renderer = None
        count = windows = hands = pixels = 0
        try:
            with av.open(str(source)) as reader:
                if not reader.streams.video:
                    raise ValueError("Input contains no video stream")
                stream = reader.streams.video[0]
                stream.thread_type = "SLICE"
                stream.codec_context.thread_count = 2
                rate = stream.average_rate
                if rate is None or rate <= 0 or stream.time_base is None:
                    raise ValueError("Cannot determine the input video frame rate/time base")
                time_base = Fraction(stream.time_base)
                with av.open(str(temp_path), mode="w", options={"movflags": "+faststart"}) as writer:
                    output = writer.add_stream("libx264", rate=rate)
                    output.pix_fmt = "yuv420p"
                    output.time_base = time_base
                    output.codec_context.time_base = time_base
                    output.codec_context.thread_count = 2
                    output.options = {"crf": "18", "preset": "fast"}
                    frames: list[np.ndarray] = []
                    times: list[int] = []
                    first_pts = previous_pts = None
                    dimensions = None

                    def write_window() -> None:
                        nonlocal count, windows, hands, pixels
                        vertices, presence, intrinsics = self.predict(np.stack(frames))
                        for index, image in enumerate(frames):
                            rendered, hand_count, pixel_count = renderer.render(
                                image, vertices[index], presence[index], intrinsics, self.threshold,
                            )
                            # H.264 yuv420p requires even dimensions; preserve all source pixels.
                            rendered = np.pad(
                                rendered, ((0, rendered.shape[0] % 2), (0, rendered.shape[1] % 2), (0, 0)),
                                mode="edge",
                            )
                            encoded = av.VideoFrame.from_ndarray(rendered, format="rgb24")
                            encoded.pts = times[index]
                            encoded.time_base = time_base
                            for packet in output.encode(encoded):
                                writer.mux(packet)
                            hands += hand_count
                            pixels += pixel_count
                            count += 1
                        windows += 1
                        LOGGER.info("Rendered %d frames (%d windows)", count, windows)
                        if progress is not None:
                            progress(count)
                        frames.clear()
                        times.clear()

                    for frame in reader.decode(stream):
                        if max_frames is not None and count + len(frames) >= max_frames:
                            raise ValueError(f"Video exceeds the {max_frames}-frame demo limit; trim it first")
                        if frame.pts is None:
                            raise ValueError("Input frame has no presentation timestamp")
                        pts = int(Fraction(frame.pts) * Fraction(frame.time_base) / time_base)
                        if previous_pts is not None and pts <= previous_pts:
                            raise ValueError("Input timestamps must be strictly increasing")
                        previous_pts = pts
                        if first_pts is None:
                            first_pts = pts
                        image = frame.to_ndarray(format="rgb24")
                        height, width = image.shape[:2]
                        if dimensions is None:
                            dimensions = (height, width)
                            output.width, output.height = width + width % 2, height + height % 2
                            renderer = OverlayRenderer(width, height, self.meshes.faces)
                        elif dimensions != (height, width):
                            raise ValueError("Video resolution changes within the stream")
                        frames.append(image)
                        times.append(pts - first_pts)
                        if len(frames) == 16:
                            write_window()
                    if frames:
                        write_window()
                    if count == 0:
                        raise ValueError("Input contains no decodable frames")
                    for packet in output.encode():
                        writer.mux(packet)
            if renderer is not None:
                finished_renderer, renderer = renderer, None
                finished_renderer.close()
            # Reopen the result before publishing it under the requested filename.
            with av.open(str(temp_path)) as check:
                decoded = sum(1 for _ in check.decode(video=0))
                if decoded != count:
                    raise RuntimeError(f"Output frame count mismatch: {decoded} != {count}")
            if pixels == 0:
                LOGGER.warning("No visible hand mesh pixels were rendered; inspect the input and predictions")
            if destination.exists():
                raise FileExistsError(destination)
            os.link(temp_path, destination)
            return {
                "output": str(destination), "frames": count, "windows": windows,
                "rendered_hands": hands, "mesh_pixels": pixels,
                "fps": float(rate),
            }
        finally:
            try:
                if renderer is not None:
                    renderer.close()
            finally:
                temp_path.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help=f"Model bundle directory or Hugging Face model ID (default: {DEFAULT_MODEL_ID})",
    )
    result.add_argument("--revision", help="Hugging Face model revision")
    result.add_argument("--mano-dir", required=True, help="Licensed directory containing MANO_RIGHT.pkl")
    result.add_argument("--encoder-source", help="Trusted local official V-JEPA 2.1 checkout (offline)")
    result.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    result.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    result.add_argument("--presence-threshold", type=float, default=0.5)
    result.add_argument("--input", required=True, help="Input video")
    result.add_argument("--output", required=True, help="New output .mp4 path")
    return result


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.set_num_threads(4)
    model = load_model(args.model, args.device, args.encoder_source, args.revision)
    demo = VideoDemo(model, args.mano_dir, args.precision, args.presence_threshold)
    print(json.dumps(demo.run(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
