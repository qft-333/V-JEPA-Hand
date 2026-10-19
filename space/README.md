---
title: V-JEPA Hand
emoji: "\U0001F590"
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.50.0
python_version: "3.10"
app_file: space_app.py
pinned: false
---

# V-JEPA Hand

[Project](https://qft-333.github.io/V-JEPA-Hand-Page/) |
[Code](https://github.com/qft-333/V-JEPA-Hand) |
[Model](https://huggingface.co/qft-333/V-JEPA-Hand)

ZeroGPU demo for egocentric hand reconstruction.

Video inference uses consecutive, non-overlapping 16-frame windows
(frames 1-16, 17-32, 33-48, ...), without blending or temporal smoothing.

The **Model check** runs inference on up to the first 16 frames of an uploaded
video and reports output validity. It does not produce a rendered video.
**Mesh overlay** is enabled only when an authorized MANO asset is configured
server-side. No MANO assets are included in this repository.

Obtain MANO under its terms from the [official website](https://mano.is.tue.mpg.de/).

## Deployment

Use this file as the Space card and `space_app.py` as the entry point. Upload
`space_app.py`, `vjepa_hand/`, `pyproject.toml`, `requirements.txt` and `packages.txt`.
The Space uses PyTorch 2.11, Gradio 5.50 and Mesa EGL software rendering.

Optional variables: `VJEPA_HAND_MODEL`, `VJEPA_HAND_REVISION`, and
`VJEPA_MAX_FRAMES` (default 900). The model check requests 60 seconds of GPU time.
Rendering requests 15-120 seconds based on video length and resolution; the
platform may reserve additional quota for overhead.

Provision an authorized server-side MANO asset and set `MANO_MODEL_DIR`, or use a
separate **private** asset repository with `MANO_ASSET_REPO`,
`MANO_ASSET_REVISION` (full commit hash), and `MANO_ASSET_SHA256` (file checksum).
For private repositories, configure a read-only `HF_TOKEN` **Secret** with access
to the required model and asset repositories. The asset is verified before loading,
excluded from file serving, and tested by a startup rendering check.

Without a configured asset, mesh rendering is disabled and only the first-window
model check is available. Do not commit MANO files or credentials into the
application or Ours weight repository, and never make the asset repository public.
Confirm the applicable permissions before enabling server-side use.
