# V-JEPA Hand

**V-JEPA Hand** is a video-based model for egocentric
two-hand reconstruction and camera-motion estimation.

[Project page](https://qft-333.github.io/V-JEPA-Hand-Page/) |
[Online demo](https://huggingface.co/spaces/qft-333/V-JEPA-Hand) |
[Model weights](https://huggingface.co/qft-333/V-JEPA-Hand)

## Video demo

Input an egocentric video and output a hand-mesh overlay: **blue for the left
hand, orange for the right hand**. No hand boxes or camera calibration are required.

### Setup

Use Python 3.10+ on Linux with an NVIDIA GPU and a compatible driver:

```bash
git clone https://github.com/qft-333/V-JEPA-Hand.git
cd V-JEPA-Hand
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Tested PyTorch build (choose a CUDA build compatible with your driver):
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

Headless rendering requires EGL/OpenGL (`libegl1` and `libgl1` on Ubuntu).
The full Ours checkpoint (about 2.43 GB, including the fine-tuned encoder) is
downloaded automatically from
[qft-333/V-JEPA-Hand](https://huggingface.co/qft-333/V-JEPA-Hand) on the first run
and cached for reuse. No Hugging Face login is required to download the public weights.

### Prepare MANO

Register at [the official MANO website](https://mano.is.tue.mpg.de/), accept its
license, and download your own `MANO_RIGHT.pkl`. MANO assets are not included
in this repository or the Ours checkpoint.

For legacy MANO files containing Chumpy objects, convert them once in a separate
Python 3.10 environment (skip this for an already NumPy-only file):

```bash
python3.10 -m venv .mano-convert
.mano-convert/bin/pip install numpy==1.23.5 scipy==1.10.1 wheel
.mano-convert/bin/pip install --no-build-isolation chumpy==0.70
.mano-convert/bin/python tools/prepare_mano.py --trust-model \
  --input /path/to/original/MANO_RIGHT.pkl --output /path/to/mano/MANO_RIGHT.pkl
```

### Run

```bash
python demo.py \
  --mano-dir /path/to/mano \
  --input input.mp4 \
  --output output.mp4
```

The installed `vjepa-hand` command accepts the same arguments. To use a local
bundle containing `config.json` and `model.safetensors`, pass
`--model /path/to/ours-model`.

**The CLI processes the entire video with no application-imposed frame or duration
limit.** It buffers one inference window at a time, so long videos do not need to
fit in memory. Runtime and output disk usage grow with video length. Unlike the
hosted demo, local inference has no ZeroGPU quota or per-request time limit.

The tool processes the full video in consecutive, non-overlapping 16-frame windows
(frames 1-16, 17-32, 33-48, ...), without averaging or smoothing across windows.
A short final window is padded by repeating its last frame, and padded
predictions are discarded. Every source frame is written exactly once, preserving
frame timestamps and the original view.

Output is a silent H.264 MP4; odd image dimensions receive one pixel of edge
padding. Default inference uses CUDA/BF16; use `--precision fp32` on GPUs without
BF16 support, or add `--device cpu` for CPU inference.

The official V-JEPA 2.1 encoder code is downloaded automatically at revision
`204698b45b3712590f06245fbfba32d3be539812`. For offline use, pass
`--encoder-source /path/to/vjepa2` with a trusted checkout of that revision.

### Web interface

```bash
python -m pip install -e '.[demo]'
python gradio_app.py --mano-dir /path/to/mano
```

Open `http://localhost:7860`, upload a video, and click **Reconstruct hands**.
The interface uses the same pipeline as the CLI, with one request at a time,
a 200 MB upload limit and a 900-frame limit (adjustable with `--max-frames`).
The CLI has no frame limit.

To deploy a ZeroGPU Space, see [Space setup](space/README.md#deployment).

## Architecture

```text
RGB video -> V-JEPA 2.1 encoder -> Self-attention predictor
                                      |
                                      +-> Bilateral hand pose and presence
                                      +-> Camera motion and intrinsics
```

The predictor jointly processes video features, per-frame queries, and a
camera-intrinsics token using self-attention with 3D rotary position embeddings.

## Code

- [demo.py](demo.py): minimal video-to-rendered-video command.
- [encoder.py](vjepa_hand/encoder.py): V-JEPA 2.1 encoder interface.
- [predictor.py](vjepa_hand/predictor.py): self-attention trajectory predictor.
- [model.py](vjepa_hand/model.py): model configuration, assembly, and output layout.
- [inference.py](vjepa_hand/inference.py): streaming inference and MP4 export.
- [geometry.py](vjepa_hand/geometry.py) and [rendering.py](vjepa_hand/rendering.py):
  MANO decoding and mesh overlay.
