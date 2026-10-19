"""Load a model-only bundle from disk or a Hugging Face model repository."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from .model import HandPoseModel, ModelConfig, build_model

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_ID = "qft-333/V-JEPA-Hand"
ENCODER_REVISION = "204698b45b3712590f06245fbfba32d3be539812"
ENCODER_REPO = f"facebookresearch/vjepa2:{ENCODER_REVISION}"
ENCODER_ENTRY = "vjepa2_1_vit_large_384"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_bundle(model: str, revision: str | None = None) -> Path:
    local = Path(model).expanduser()
    if local.is_dir():
        return local.resolve()
    if local.exists() or model.startswith(("/", ".", "~")):
        raise FileNotFoundError(f"Expected a model bundle directory: {local}")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=model, revision=revision,
        allow_patterns=["config.json", "model.safetensors"],
    ))


def load_model(
    model: str, device: str = "cuda", encoder_source: str | None = None,
    revision: str | None = None,
) -> HandPoseModel:
    bundle = resolve_bundle(model, revision)
    manifest = json.loads((bundle / "config.json").read_text())
    if manifest.get("format") != "vjepa-hand-v1":
        raise ValueError("Unsupported model bundle format")
    if manifest.get("model_config") != asdict(ModelConfig()):
        raise ValueError("This demo requires the released fixed-length main-model configuration")
    weights = bundle / "model.safetensors"
    if file_sha256(weights) != manifest["weights_sha256"]:
        raise ValueError("Model weight checksum mismatch; re-download the bundle")
    if manifest.get("encoder_revision") != ENCODER_REVISION:
        raise ValueError("The bundle requires a different encoder code revision")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch or use --device cpu")
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda")

    source = ENCODER_REPO
    source_kind = "github"
    if encoder_source is not None:
        directory = Path(encoder_source).expanduser().resolve(strict=True)
        if not (directory / "hubconf.py").is_file():
            raise FileNotFoundError(f"No hubconf.py in {directory}")
        source, source_kind = str(directory), "local"
    LOGGER.info("Building encoder from %s (no pretrained weight download)", source)
    backend, _ = torch.hub.load(
        source, ENCODER_ENTRY, source=source_kind, trust_repo=True,
        pretrained=False, num_frames=16,
    )
    network = build_model(backend)
    state = load_file(str(weights), device="cpu")
    network.load_state_dict(state, strict=True)
    del state
    return network.to(target).eval()
