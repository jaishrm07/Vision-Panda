# Structured Phase + Geometry BC Framing

## Research Questions

1. Which visual diversity axes matter most for OOD generalization?
2. Do pretrained visual encoders help in simulation, or are frozen real-image features weak under sim visuals?
3. Does partial fine-tuning improve over frozen encoders and scratch CNNs?
4. Are loose success thresholds hiding poor precision and unstable final behavior?

## Method Positioning

This result should be framed as **phase- and geometry-conditioned visual behavior cloning**, not pure pixel BC.

The diagnostic adds an 18-D structured state to the two RGB streams and end-effector state:

- avoid phase one-hot: `cube_hover`, `side_align`, `final_descent`
- target position
- obstacle center and half-extents
- end-effector to target delta
- target to obstacle delta

The policy still consumes external RGB and wrist RGB, but the structured signal makes the hidden task stage and obstacle geometry explicit. This directly tests whether the obstacle-aware task is failing because image-only BC cannot infer phase/geometry reliably.

Evaluation reports `success@0.5cm`, `success@1cm`, `success@2cm`, and `success@5cm` using closest distance reached during rollout, plus mean best distance and mean final distance. Rollouts only terminate early at 1 mm, so `final_distance` is not artificially clipped at 5 cm.

## Main Result

Edge-balanced visual-only training did not solve the hard spatial setting.

| Setup | Model | Split | S@1cm | S@5cm | Best cm | Final cm |
| --- | --- | --- | --- | --- | --- | --- |
| Edge-balanced visual-only | Scratch CNN | ID | 0.0% | 4.7% | 17.16 | 23.08 |
| Edge-balanced visual-only | Scratch CNN | OOD | 1.3% | 14.7% | 12.95 | 19.25 |
| Edge-balanced phase+geometry | Scratch CNN | ID | 64.7% | 74.7% | 3.01 | 3.27 |
| Edge-balanced phase+geometry | Scratch CNN | OOD | 53.3% | 58.7% | 4.05 | 4.23 |

The structured scratch model is the strongest precision result in the edge-balanced setting: it sharply improves `success@1cm` and keeps final distance close to best distance. This supports the hidden-stage/geometry bottleneck hypothesis.

## Ablation Result

| Variant | Split | S@1cm | S@2cm | S@5cm | Best cm | Final cm |
| --- | --- | --- | --- | --- | --- | --- |
| Phase only | ID | 0.0% | 0.7% | 6.0% | 19.21 | 22.11 |
| Phase only | OOD | 0.0% | 2.0% | 10.7% | 14.60 | 17.27 |
| Target geometry only | ID | 22.7% | 99.3% | 100.0% | 1.25 | 3.52 |
| Target geometry only | OOD | 6.7% | 94.0% | 100.0% | 1.46 | 3.17 |
| Full geometry only | ID | 31.3% | 82.7% | 100.0% | 1.27 | 4.30 |
| Full geometry only | OOD | 21.3% | 82.0% | 100.0% | 1.45 | 3.80 |
| Phase + geometry | ID | 64.7% | 74.0% | 74.7% | 3.01 | 3.27 |
| Phase + geometry | OOD | 53.3% | 55.3% | 58.7% | 4.05 | 4.23 |

Interpretation:

- Phase alone is not useful; it cannot locate the target or obstacle.
- Geometry alone gives excellent loose success and final stability, but weaker sub-centimeter precision.
- Phase + geometry gives the best `success@1cm`, but it is more bimodal: many rollouts become very precise, while a minority fail badly enough to reduce `success@5cm`.
- This suggests a tradeoff between **robust coarse reaching** and **precise phase-conditioned behavior**.

## Spatial Bucket Result

The bucket breakdown shows that structured scratch improves the hardest spatial groups, but not uniformly.

| Setup | Split | Bucket group | S@1cm | S@5cm | Best cm | Final cm |
| --- | --- | --- | --- | --- | --- | --- |
| Visual-only scratch | OOD | corners | 0.0% | 0.0% | 22.29 | 30.40 |
| Visual-only scratch | OOD | edges | 1.6% | 9.4% | 14.23 | 20.64 |
| Visual-only scratch | OOD | interior | 1.4% | 23.2% | 9.47 | 15.22 |
| Structured scratch | OOD | corners | 82.4% | 100.0% | 0.58 | 1.23 |
| Structured scratch | OOD | edges | 62.5% | 64.1% | 4.36 | 4.53 |
| Structured scratch | OOD | interior | 37.7% | 43.5% | 4.61 | 4.70 |

The remaining weakness is not generic spatial OOD anymore. It is concentrated in specific edge/interior cases, especially where the phase-conditioned policy chooses a bad mode and never recovers.

## Encoder Takeaway

Frozen ResNet18 is more robust than visual-only scratch at loose thresholds, but it does not match structured scratch precision. Partial ResNet18 remains unstable in this setting. For this simulated task, pretrained visual features are not enough by themselves; explicit geometry and phase diagnostics are more informative.

## Figures And Videos

Tables:

- `results/structured_analysis/main_comparison.png`
- `results/structured_analysis/scratch_structured_ablation.png`
- `results/structured_analysis/spatial_bucket_groups.png`

Videos:

- `results/structured_analysis/videos/visual_only_scratch_ood_failure.mp4`
- `results/structured_analysis/videos/structured_scratch_ood_success.mp4`
- `results/structured_analysis/videos/structured_scratch_ood_failure.mp4`
- `results/structured_analysis/videos/frozen_structured_ood_comparison.mp4`

## Paper Framing

A defensible workshop framing is:

> Pixel BC on simulated visual obstacle-aware reaching appears to fail not only from visual diversity limits, but from a hidden task-stage and geometry inference bottleneck. Explicit phase and geometry conditioning reveals that the task can be cloned precisely, while ablations show a tradeoff between coarse robustness and precise phase-conditioned behavior.

This is strongest as a diagnostic paper: visual diversity alone is insufficient, pretrained features are mixed, and evaluation at multiple precision thresholds changes the conclusion.
