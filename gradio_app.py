"""Gradio upload-and-render interface using the same pipeline as demo.py."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import torch

from vjepa_hand.camera import CAMERA_HELP, CameraIntrinsics
from vjepa_hand.inference import VideoDemo
from vjepa_hand.loading import DEFAULT_MODEL_ID, load_model


def create_app(demo: VideoDemo, output_dir: Path, max_frames: int = 900) -> gr.Blocks:
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    def run(video: str | None, camera_intrinsics: str = "", progress=gr.Progress()) -> tuple[str, str]:
        if not video:
            raise gr.Error("Upload a video first.")
        for previous in output_dir.glob("*.mp4"):
            if previous.stat().st_mtime < time.time() - 3600:
                previous.unlink()
        output = output_dir / f"{uuid.uuid4().hex}.mp4"
        try:
            camera = CameraIntrinsics.parse(camera_intrinsics)
            result = demo.run(
                video, str(output),
                progress=lambda count: progress(min(count / max_frames, 0.99), desc=f"Rendered {count} frames"),
                max_frames=max_frames,
                camera=camera,
            )
        except (ValueError, OSError, RuntimeError) as error:
            logging.exception("Video reconstruction failed")
            raise gr.Error(f"Reconstruction failed: {error}") from error
        return str(output), (
            f"Rendered {result['frames']} frames; {result['rendered_hands']} hand predictions. "
            f"Camera: {result['camera_mode']}; output {result['width']} x {result['height']}."
        )

    with gr.Blocks(title="V-JEPA Hand", delete_cache=(3600, 3600)) as app:
        gr.Markdown(
            "# V-JEPA Hand\nUpload an egocentric video to reconstruct and overlay both hands. "
            "Blue: left hand; orange: right hand. Output is silent. "
            f"This demo accepts at most {max_frames} frames per video."
        )
        with gr.Row():
            video = gr.Video(label="Input video", sources=["upload"])
            result = gr.Video(label="Rendered video")
        camera_intrinsics = gr.Textbox(
            label="Optional camera intrinsics (pixels): fx, fy, cx, cy",
            value="", placeholder="Leave blank if unknown", info=CAMERA_HELP,
        )
        button = gr.Button("Reconstruct hands", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        button.click(run, inputs=[video, camera_intrinsics], outputs=[result, status],
                     concurrency_limit=1, api_name="render")
    return app.queue(max_size=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("VJEPA_HAND_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--mano-dir", default=os.environ.get("MANO_MODEL_DIR"))
    parser.add_argument("--encoder-source", default=os.environ.get("VJEPA_ENCODER_SOURCE"))
    parser.add_argument("--revision", default=os.environ.get("VJEPA_HAND_REVISION"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--max-frames", type=int, default=900)
    args = parser.parse_args()
    if not args.mano_dir:
        parser.error("Provide --mano-dir (or MANO_MODEL_DIR)")
    if not args.model:
        parser.error("--model or VJEPA_HAND_MODEL must not be empty")
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    model = load_model(args.model, args.device, args.encoder_source, args.revision)
    demo = VideoDemo(model, args.mano_dir, args.precision)
    with tempfile.TemporaryDirectory(prefix="vjepa-hand-demo-") as directory:
        output_dir = Path(directory)
        app = create_app(demo, output_dir, args.max_frames)
        app.launch(
            server_name=args.host, server_port=args.port,
            allowed_paths=[str(output_dir)], max_file_size="200mb", share=False,
        )


if __name__ == "__main__":
    main()
