# V-JEPA Hand

Model architecture for **V-JEPA Hand**, a video-based model for egocentric
two-hand reconstruction and camera-motion estimation.

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

- [encoder.py](vjepa_hand/encoder.py): V-JEPA 2.1 encoder interface.
- [predictor.py](vjepa_hand/predictor.py): self-attention trajectory predictor.
- [model.py](vjepa_hand/model.py): model configuration, assembly, and output layout.

## Installation

Requires Python 3.10+ and PyTorch 2.1+.

```bash
python -m pip install -e .
```

Obtain the [official V-JEPA 2.1 encoder](https://github.com/facebookresearch/vjepa2)
separately and pass its encoder module to `vjepa_hand.build_model`.
