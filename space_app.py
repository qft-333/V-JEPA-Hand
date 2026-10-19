"""ZeroGPU entry point; MANO assets must be configured separately by the operator."""

from __future__ import annotations

import os

# ZeroGPU provides CUDA for PyTorch, not an NVIDIA OpenGL rendering context.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import spaces

import logging
import math
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import av
import gradio as gr
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

from vjepa_hand.geometry import HandMeshes
from vjepa_hand.inference import VideoDemo, predict_parameters
from vjepa_hand.loading import DEFAULT_MODEL_ID, file_sha256, load_model
from vjepa_hand.model import HandPoseModel
from vjepa_hand.rendering import OverlayRenderer

LOGGER = logging.getLogger(__name__)
MISSING_MANO = (
    "Mesh rendering is unavailable: the server has no configured MANO asset. "
    "The model check below runs real inference on up to the first 16 frames; "
    "it does not render a video."
)


def resolve_mano_dir() -> str | None:
    directory = os.environ.get("MANO_MODEL_DIR")
    repository = os.environ.get("MANO_ASSET_REPO")
    if directory and repository:
        raise ValueError("Set only one of MANO_MODEL_DIR and MANO_ASSET_REPO")
    if not repository:
        return directory
    revision = os.environ.get("MANO_ASSET_REVISION")
    checksum = os.environ.get("MANO_ASSET_SHA256")
    if not revision or len(revision) != 40 or not checksum or len(checksum) != 64:
        raise ValueError("Remote MANO requires a full MANO_ASSET_REVISION and MANO_ASSET_SHA256")
    info = HfApi().model_info(repository, revision=revision)
    if not info.private:
        raise ValueError("MANO_ASSET_REPO must remain private")
    path = Path(hf_hub_download(repository, "MANO_RIGHT.pkl", revision=revision))
    if file_sha256(path) != checksum:
        raise ValueError("MANO asset checksum mismatch")
    LOGGER.info("Loaded checksum-verified MANO asset from private repository %s", repository)
    return str(path.parent)


def configure_software_renderer() -> None:
    from pyrender.platforms.egl import query_devices

    # Mesa's software device has no DRM node; hardware device indices vary by host.
    indices = [index for index, device in enumerate(query_devices()) if device.name is None]
    if len(indices) != 1:
        raise RuntimeError(f"Expected one Mesa software EGL device, found {len(indices)}")
    os.environ["EGL_DEVICE_ID"] = str(indices[0])


def validate_renderer(mano_dir: str) -> None:
    from OpenGL.GL import GL_RENDERER, glGetString

    configure_software_renderer()
    meshes = HandMeshes(mano_dir)
    trajectory = torch.zeros(1, 119)
    for base, x in ((0, -0.08), (55, 0.08)):
        trajectory[0, base:base + 3] = torch.tensor([x, 0, 0.6])
        trajectory[0, base + 3:base + 9] = torch.tensor([1, 0, 0, 0, 1, 0])
    vertices = meshes.vertices(trajectory)[0]
    renderer = OverlayRenderer(128, 128, meshes.faces)
    try:
        _, hands, pixels = renderer.render(
            np.zeros((128, 128, 3), dtype=np.uint8),
            vertices, np.ones(2), np.ones(4), 0.5,
        )
        if hands != 2 or pixels == 0:
            raise RuntimeError("MANO rendering self-check produced no visible mesh")
        LOGGER.info("MANO rendering self-check passed: %d hands, %d visible pixels", hands, pixels)
        LOGGER.info("OpenGL renderer: %s", glGetString(GL_RENDERER))
    finally:
        renderer.close()


def first_window(video: str) -> np.ndarray:
    frames = []
    with av.open(video) as reader:
        if not reader.streams.video:
            raise ValueError("Input contains no video stream")
        stream = reader.streams.video[0]
        stream.codec_context.thread_count = 2
        for frame in reader.decode(stream):
            if frame.width > 4096 or frame.height > 4096:
                raise ValueError("Model check accepts videos up to 4096 pixels per dimension")
            image = frame.to_ndarray(format="rgb24")
            if frames and frames[0].shape != image.shape:
                raise ValueError("Video resolution changes within the window")
            frames.append(image)
            if len(frames) == 16:
                break
    if not frames:
        raise ValueError("Input contains no decodable frames")
    height, width = frames[0].shape[:2]
    if width / height > 3:
        raise ValueError("This hosted demo accepts an aspect ratio of at most 3:1")
    return np.stack(frames)


def render_duration(video: str | None) -> int:
    if not video:
        raise gr.Error("Upload a video first.")
    try:
        with av.open(video) as reader:
            if not reader.streams.video:
                raise ValueError("Input contains no video stream")
            stream = reader.streams.video[0]
            if stream.frames <= 0:
                return 120
            scale = max(1.0, stream.width * stream.height / (1280 * 720))
            return min(120, max(15, math.ceil(stream.frames * 0.3 * scale)))
    except (ValueError, OSError, RuntimeError) as error:
        LOGGER.exception("Cannot estimate video runtime")
        raise gr.Error(f"Cannot read input video: {error}") from error


def create_space(
    model: HandPoseModel, output_dir: Path, mano_dir: str | None, max_frames: int = 900,
) -> gr.Blocks:
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if mano_dir and not (Path(mano_dir).expanduser() / "MANO_RIGHT.pkl").is_file():
        raise FileNotFoundError("MANO_MODEL_DIR is set but contains no MANO_RIGHT.pkl")
    if not mano_dir:
        LOGGER.warning(MISSING_MANO)

    @spaces.GPU(duration=60)
    def check_model(video: str | None) -> dict:
        if not video:
            raise gr.Error("Upload a video first.")
        try:
            frames = first_window(video)
            trajectory, intrinsics = predict_parameters(
                model, frames, torch.device("cuda"), "bf16",
            )
        except (ValueError, OSError, RuntimeError) as error:
            LOGGER.exception("Model check failed")
            raise gr.Error(f"Model check failed: {error}") from error
        return {
            "status": "Model inference passed; no mesh rendering was performed.",
            "frames_checked": len(frames),
            "trajectory_shape": list(trajectory.shape),
            "intrinsics_shape": list(intrinsics.shape),
            "finite_outputs": True,
            "positive_focal_lengths": True,
        }

    @spaces.GPU(duration=render_duration)
    def render_video(video: str | None) -> tuple[str, str]:
        if not mano_dir:
            raise gr.Error(MISSING_MANO)
        if not video:
            raise gr.Error("Upload a video first.")
        try:
            for previous in output_dir.glob("*.mp4"):
                if previous.stat().st_mtime < time.time() - 3600:
                    previous.unlink()
            output = output_dir / f"{uuid.uuid4().hex}.mp4"
            configure_software_renderer()
            demo = VideoDemo(model, mano_dir)
            result = demo.run(video, str(output), max_frames=max_frames)
        except (ValueError, OSError, RuntimeError) as error:
            LOGGER.exception("Video reconstruction failed")
            raise gr.Error(f"Reconstruction failed: {error}") from error
        status = (
            f"Rendered {result['frames']} frames; {result['rendered_hands']} hand predictions; "
            f"{result['mesh_pixels']} visible mesh pixels."
        )
        if result["mesh_pixels"] == 0:
            status += " No visible hand overlay was produced; inspect the input and predictions."
        return str(output), status

    with gr.Blocks(title="V-JEPA Hand", delete_cache=(3600, 3600)) as app:
        gr.Markdown(
            "# V-JEPA Hand\nReconstruct bilateral hand motion from egocentric video.\n\n"
            "[Project](https://qft-333.github.io/V-JEPA-Hand-Page/) | "
            "[Code](https://github.com/qft-333/V-JEPA-Hand) | "
            "[MANO registration and license](https://mano.is.tue.mpg.de/)"
        )
        if not mano_dir:
            gr.Markdown(f"**Rendering not configured.** {MISSING_MANO}")
        video = gr.Video(label="Input video", sources=["upload"])
        with gr.Tab("Mesh overlay"):
            gr.Markdown(
                f"Blue: left hand; orange: right hand. Silent output. Up to {max_frames} frames. "
                "ZeroGPU requests have a 120-second runtime limit; use a shorter video if a request times out."
            )
            render = gr.Button("Reconstruct hands", variant="primary", interactive=bool(mano_dir))
            result = gr.Video(label="Rendered video")
            status = gr.Textbox(label="Status", interactive=False)
            if mano_dir:
                render.click(render_video, video, [result, status], api_name="render", concurrency_limit=1,
                             concurrency_id="gpu")
        with gr.Tab("Model check"):
            gr.Markdown(
                "Run the real model on up to the first 16 frames. "
                "This checks GPU inference only, not full-video reconstruction or rendering."
            )
            check = gr.Button("Check model (first 16 frames)")
            report = gr.JSON(label="Inference check")
            check.click(check_model, video, report, api_name="check_model", concurrency_limit=1,
                        concurrency_id="gpu")
    return app.queue(max_size=4)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    torch.set_num_threads(2)
    model_id = os.environ.get("VJEPA_HAND_MODEL", DEFAULT_MODEL_ID)
    LOGGER.info("HF_TOKEN configured: %s", bool(os.environ.get("HF_TOKEN")))
    mano_dir = resolve_mano_dir()
    if mano_dir:
        # Mesa must not be initialized in the parent that ZeroGPU forks.
        subprocess.run(
            [sys.executable, "-c",
             "import logging, sys; logging.basicConfig(level=logging.INFO); "
             "from space_app import validate_renderer; validate_renderer(sys.argv[1])",
             mano_dir],
            cwd=Path(__file__).parent, check=True, timeout=60,
        )
    model = load_model(
        model_id, device="cuda", revision=os.environ.get("VJEPA_HAND_REVISION"),
        encoder_source=os.environ.get("VJEPA_ENCODER_SOURCE"),
    )
    LOGGER.info("Loaded model %s with %d state tensors", model_id, len(model.state_dict()))
    with tempfile.TemporaryDirectory(prefix="vjepa-hand-space-") as directory:
        app = create_space(
            model, Path(directory), mano_dir,
            max_frames=int(os.environ.get("VJEPA_MAX_FRAMES", "900")),
        )
        app.launch(
            server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
            server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
            allowed_paths=[directory], max_file_size="200mb", share=False,
            blocked_paths=(
                [str(Path(mano_dir).expanduser().resolve()),
                 str((Path(mano_dir).expanduser() / "MANO_RIGHT.pkl").resolve())]
                if mano_dir else []
            ),
        )


if __name__ == "__main__":
    main()
