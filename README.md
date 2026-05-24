# Vision-Panda: Controlled Visual Diversity for Pixel Behavior Cloning

Vision-Panda is a controlled robot imitation-learning project for studying how visual distribution shift affects closed-loop behavior cloning from pixels. The active project lives in [`research2/`](research2/) and uses a PyBullet Franka Panda reaching environment with dual RGB observations.

The core question is:

> Under a fixed demonstration budget, which visual diversity axes actually improve out-of-distribution robot behavior cloning, and when do pretrained visual encoders help or fail?

## Highlights

- Built a 128x128 dual-camera PyBullet benchmark with **176 training datasets**, **947,429 expert timesteps**, and **1,894,858 RGB images**.
- Evaluated **4 visual shift axes**: target color, target spatial distribution, external camera viewpoint, and lighting direction/intensity.
- Compared **Scratch CNN**, **frozen ImageNet ResNet-18**, and **partially fine-tuned ResNet-18** policies across **5 demo budgets**.
- Aggregated **1,056 metric files** and **158,400 closed-loop policy rollouts**.
- Diagnosed obstacle-aware spatial failures with a phase-and-geometry policy, improving hardest-case OOD `success@1cm` from **1.3% to 53.3%**.

## Repository Layout

```text
research2/
  configs/
    dataset_128px_v1.yaml        # source of truth for tasks, axes, budgets, eval configs
  code/
    collect_dataset.py           # scripted expert dataset collection
    train_bc.py                  # behavior cloning training
    evaluate_bc.py               # closed-loop rollout evaluation
    run_bc_matrix.py             # training matrix launcher
    run_bc_eval_matrix.py        # evaluation matrix launcher
    build_results_presentation.py
  results/
    analysis_id_ood_all_budgets/ # public aggregate metrics
    analysis_scratch_cnn/        # scratch baseline summary tables
    slide_charts/                # publication/presentation figures
    structured_analysis/         # phase+geometry diagnostic tables/figures
  paper/
    vision_panda_paper.tex
    vision_panda_paper.pdf
```

Large generated artifacts such as raw `.pkl` datasets, `.pt` model checkpoints, logs, and rollout videos are intentionally excluded from Git. The public repository keeps the code, configs, summaries, plots, presentation HTML, and paper.

## Benchmark

The benchmark studies two tasks:

- `reach`: move the end effector to a target cube.
- `avoid_reach`: route around a static obstacle before reaching the target.

Each policy observes:

- an external RGB camera,
- a wrist/end-effector RGB camera,
- the current 3D end-effector position.

The policy predicts a 3D Cartesian delta action and is trained with MSE against a scripted expert.

## Visual Diversity Axes

| Axis | Training contrast | OOD evaluation |
| --- | --- | --- |
| Color | red-only vs multi-color target | held-out target colors |
| Spatial distribution | central target region vs wider workspace | edge/corner targets |
| Camera viewpoint | fixed external camera vs yaw/pitch/distance variation | more extreme viewpoints |
| Lighting | fixed neutral lighting vs direction/intensity variation | extreme direction/intensity |

## Key Results

At 50 demonstrations, Scratch CNN showed strong differences across OOD axes:

| OOD axis | Scratch CNN success@1cm |
| --- | ---: |
| Camera viewpoint | 57.9% |
| Color | 41.1% |
| Lighting | 20.6% |
| Spatial distribution | 2.9% |

The main finding is that visual diversity is not interchangeable. Appearance and viewpoint shifts were often manageable at loose precision thresholds, but spatial distribution shift remained the dominant failure mode.

For the hardest obstacle-aware spatial setting, a diagnostic policy with explicit phase and geometry features improved:

| Setup | OOD success@1cm | Final distance |
| --- | ---: | ---: |
| Edge-balanced visual-only Scratch CNN | 1.3% | 19.25 cm |
| Edge-balanced phase+geometry Scratch CNN | 53.3% | 4.23 cm |

This suggests that the hardest failures are not only about pixel diversity; image-only policies also struggle to infer task stage and obstacle-target geometry.

## Figures

Selected public figures are in:

- [`research2/results/slide_charts/`](research2/results/slide_charts/)
- [`research2/results/structured_analysis/`](research2/results/structured_analysis/)

## Setup

Install the core dependencies in a Python environment with PyTorch:

```bash
pip install -r requirements.txt
```

For CUDA training, install the PyTorch build appropriate for your system from the official PyTorch instructions.

## Quick Start

Preview the simulator and visual axes:

```bash
cd research2
python3 code/preview_simulator.py \
  --config configs/dataset_128px_v1.yaml \
  --samples-per-config 1
```

Collect a small dataset:

```bash
python3 code/collect_dataset.py \
  --config configs/dataset_128px_v1.yaml \
  --train-configs color_multi \
  --budgets 5 \
  --seeds 0
```

Train one Scratch CNN behavior cloning policy:

```bash
python3 code/train_bc.py \
  --train-config color_multi \
  --budget 5 \
  --seed 0 \
  --model-family scratch_bc_128 \
  --epochs 10 \
  --output-dir results/bc_128px_v1
```

## Paper

The 7-page paper draft is available at:

- [`research2/paper/vision_panda_paper.pdf`](research2/paper/vision_panda_paper.pdf)
- [`research2/paper/vision_panda_paper.tex`](research2/paper/vision_panda_paper.tex)

## Artifact Policy

The GitHub repository is intended to be public and lightweight. It tracks:

- source code,
- configs,
- paper materials,
- aggregate CSV/JSON summaries,
- selected figures.

It does not track:

- raw `.pkl` datasets,
- trained `.pt` checkpoints,
- generated logs,
- generated analysis HTML/assets,
- rollout videos,
- course/private materials.

Those large artifacts should be hosted separately, for example on Google Drive, Hugging Face Datasets, Zenodo, or GitHub Releases.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
