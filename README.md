<div align="center">

# V-JEPA Hand
### Simple, Accurate and Fast 3D Hand and Camera Trajectory Recovery from Egocentric Videos

**Ruicheng Li · Qixiu Li · Yu Deng · Sicheng Xu · Jiaolong Yang · Baining Guo**

Tsinghua University · Microsoft Research

[**Paper**](https://arxiv.org/abs/2610.26780) · [**Project page**](https://qft-333.github.io/V-JEPA-Hand-Page/) · [**Video results**](https://qft-333.github.io/V-JEPA-Hand-Page/#results) · [**Early-access inquiries**](mailto:lirc23@mails.tsinghua.edu.cn)

</div>

## Code release paused — Microsoft review in progress

We apologize for the temporary interruption. The code release is paused while undergoing Microsoft's internal review process. This default branch currently provides project information rather than the implementation. Model weights and the interactive demo are temporarily unavailable.

For early-access inquiries, please contact **[lirc23@mails.tsinghua.edu.cn](mailto:lirc23@mails.tsinghua.edu.cn)**. Access is subject to applicable approvals.

## At a glance

**One video encoder. One predictor. Joint reconstruction.**

V-JEPA Hand recovers bilateral 3D hand motion and camera trajectories from egocentric video in a single forward pass. Full-frame features from a V-JEPA 2.1 encoder are decoded into hand pose, hand presence, camera motion, and focal lengths.

| Joint reconstruction | Temporal context | Fast inference |
| :--- | :--- | :--- |
| Recover both hands and camera motion together. | Learn from video rather than isolated frames. | 93.4 FPS forward-pass throughput on a single RTX A6000. |

## Project video

[![V-JEPA Hand project video](https://qft-333.github.io/V-JEPA-Hand-Page/static/poster.jpg)](https://qft-333.github.io/V-JEPA-Hand-Page/#video)

**[Watch the project video](https://qft-333.github.io/V-JEPA-Hand-Page/static/project-video.mp4)** · **[Explore nine interactive video examples](https://qft-333.github.io/V-JEPA-Hand-Page/#results)**

## Method overview

![V-JEPA Hand architecture](https://qft-333.github.io/V-JEPA-Hand-Page/static/architecture.png)

A shared video encoder and query-based predictor jointly estimate hand and camera motion. Predicted cameras place the hand trajectories in a common world frame. The model accepts known focal lengths or predicts them when unavailable.

See the [paper](https://arxiv.org/abs/2610.26780) and [project page](https://qft-333.github.io/V-JEPA-Hand-Page/) for the method, benchmark results, and visualizations.

## Citation

```bibtex
@misc{li2026vjepahandsimpleaccurate,
  title={V-JEPA Hand: Simple, Accurate and Fast 3D Hand and Camera Trajectory Recovery from Egocentric Videos},
  author={Ruicheng Li and Qixiu Li and Yu Deng and Sicheng Xu and Jiaolong Yang and Baining Guo},
  year={2026},
  eprint={2610.26780},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.26780},
}
```
