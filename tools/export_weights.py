"""Export a trusted research checkpoint as a model-only inference bundle."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from vjepa_hand.loading import ENCODER_REVISION, file_sha256
from vjepa_hand.model import ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-checkpoint", action="store_true", help="Allow pickle from your own trusted training run")
    args = parser.parse_args()
    if not args.trust_checkpoint:
        parser.error("Training checkpoints can execute pickle code. Export only your own checkpoint with --trust-checkpoint")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if payload.get("global_step") != 30519:
        raise ValueError("Expected the final step-30519 training checkpoint")
    state = payload["model"]
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Invalid model state dictionary")
    args.output.mkdir(parents=True)
    weights = args.output / "model.safetensors"
    save_file({k: v.contiguous() for k, v in state.items()}, str(weights))
    manifest = {
        "format": "vjepa-hand-v1",
        "model_config": asdict(ModelConfig()),
        "encoder_revision": ENCODER_REVISION,
        "training_step": 30519,
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "weights_sha256": file_sha256(weights),
    }
    (args.output / "config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Model-only bundle saved to {args.output}")


if __name__ == "__main__":
    main()
