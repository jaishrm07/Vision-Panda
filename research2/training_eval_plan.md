# Research2 BC Training and ID/OOD Evaluation Plan

## What the old research pipeline did

The old `research` pipeline used `run_benchmark.py` as the orchestrator.

Training:
- `run_benchmark.py` loaded a benchmark config JSON.
- With `--skip-collection`, it skipped dataset collection and reused existing datasets.
- It called `train_policy.py::train_model`.
- Input was two RGB images plus end-effector position:
  - static external camera image
  - end-effector camera image
  - EE position
- Target was the expert 3D delta-position action.
- Loss was action MSE.
- Main settings:
  - model family: `scratch_bc`
  - image size: `64x64`
  - budgets: `5, 20, 50`
  - seeds: `0, 1, 2`
  - epochs: `10`
  - batch size: `64`
  - learning rate: `0.001`

Evaluation:
- The old pipeline evaluated with closed-loop PyBullet rollouts, not by replaying a saved eval pickle.
- `run_benchmark.py` called `test_policy.py::evaluate_policy`.
- `evaluate_policy` reset a `BenchmarkEnv` using an eval scene config.
- At each step it rendered current images, read current EE position, predicted an action, stepped the robot, and repeated until success or step cap.
- Default eval settings:
  - `num_tests: 100`
  - `steps_per_test: 400`
  - success threshold: `0.05`
  - headless PyBullet
- Metrics:
  - success rate
  - mean best distance
  - mean steps
  - mean success steps
  - 95% CI across seeds for aggregate results

## What changes for research2

We cannot reuse the old code unchanged.

Reasons:
- old dataset format was a list: `[static_img, ee_img, state_plus_action]`
- new dataset format is a dict with `samples`
- new image fields are `external_rgb` and `eef_rgb`
- new state field is `robot_state.ee_position`
- new action field is `expert_action.delta_position`
- old scratch CNN was sized for `64x64`; research2 images are `128x128`

## BC model for research2

Use a simple scratch behavior cloning policy first.

Inputs:
- `external_rgb`: `3 x 128 x 128`
- `eef_rgb`: `3 x 128 x 128`
- `ee_position`: `3`

Target:
- `delta_position`: `3`

Image encoder, separate weights for each camera:

```text
Conv 3 -> 32, kernel 5, stride 2
ReLU
Conv 32 -> 64, kernel 5, stride 2
ReLU
Conv 64 -> 128, kernel 3, stride 2
ReLU
Conv 128 -> 128, kernel 3, stride 2
ReLU
AdaptiveAvgPool2d(1)
Flatten
Linear 128 -> 64
ReLU
```

Policy head:

```text
concat(z_external, z_eef, ee_position)  # 64 + 64 + 3 = 131
Linear 131 -> 256
ReLU
Linear 256 -> 128
ReLU
Linear 128 -> 3
Tanh
```

Training loss:

```text
MSE(predicted_delta_position, expert_delta_position)
```

Initial hyperparameters:
- model: `scratch_bc_128`
- epochs: `10`
- batch size: `64`
- optimizer: Adam
- learning rate: `0.001`
- checkpoint every epoch
- save final model and training history JSON

## Training matrix

Train one model per training dataset:

- 16 train configs
- 3 budgets: `5, 20, 50`
- 3 train seeds: `0, 1, 2`

Total:

```text
16 * 3 * 3 = 144 BC models
```

Training data:

```text
/home/jaisharma/HW8/research2/results/datasets_128px_v1
```

Output:

```text
/home/jaisharma/HW8/research2/results/bc_128px_v1
```

Expected subdirs:
- `models/`
- `histories/`
- `metrics/`
- `aggregates/`
- `plots/`
- `logs/`

## ID eval datasets

We still need ID eval scene banks.

Collect:
- same 16 train distributions
- budget `50`
- eval seeds `200, 201, 202`

Output:

```text
/home/jaisharma/HW8/research2/results/eval_id_128px_v1
```

Expected:

```text
16 configs * 1 budget * 3 seeds = 48 ID eval datasets
```

Purpose:
- not for training
- used as deterministic scene banks for closed-loop ID evaluation

## OOD eval datasets

Already collected:

```text
/home/jaisharma/HW8/research2/results/eval_ood_128px_v1
```

Expected and validated:

```text
8 OOD configs * 1 budget * 3 seeds = 24 OOD eval datasets
```

These are also used as deterministic scene banks for closed-loop OOD evaluation.

## Eval mapping

Each trained model gets one ID eval group and one OOD eval group.

| Train config | ID eval | OOD eval |
|---|---|---|
| `color_red_only` | `color_red_only` | `color_ood_eval` |
| `color_multi` | `color_multi` | `color_ood_eval` |
| `avoid_color_red_only` | `avoid_color_red_only` | `avoid_color_ood_eval` |
| `avoid_color_multi` | `avoid_color_multi` | `avoid_color_ood_eval` |
| `spatial_narrow` | `spatial_narrow` | `spatial_edge_ood_eval` |
| `spatial_wide` | `spatial_wide` | `spatial_edge_ood_eval` |
| `avoid_spatial_narrow` | `avoid_spatial_narrow` | `avoid_spatial_edge_ood_eval` |
| `avoid_spatial_wide` | `avoid_spatial_wide` | `avoid_spatial_edge_ood_eval` |
| `camera_fixed` | `camera_fixed` | `camera_extreme_ood_eval` |
| `camera_multi_pose` | `camera_multi_pose` | `camera_extreme_ood_eval` |
| `avoid_camera_fixed` | `avoid_camera_fixed` | `avoid_camera_extreme_ood_eval` |
| `avoid_camera_multi_pose` | `avoid_camera_multi_pose` | `avoid_camera_extreme_ood_eval` |
| `lighting_fixed` | `lighting_fixed` | `lighting_extreme_ood_eval` |
| `lighting_diverse` | `lighting_diverse` | `lighting_extreme_ood_eval` |
| `avoid_lighting_fixed` | `avoid_lighting_fixed` | `avoid_lighting_extreme_ood_eval` |
| `avoid_lighting_diverse` | `avoid_lighting_diverse` | `avoid_lighting_extreme_ood_eval` |

## Closed-loop research2 evaluation

Evaluation should be closed-loop, matching old `research`.

For each model:

1. Load model checkpoint.
2. Load eval scene bank metadata from ID or OOD eval datasets.
3. For each eval episode:
   - reset PyBullet scene to that episode's `initial_scene`
   - reset Panda to the standard start pose
   - render current `external_rgb` and `eef_rgb`
   - read current EE position
   - predict `delta_position`
   - clip action magnitude to `1.0`
   - step robot with position gain `0.03`
   - stop at success or `400` steps
4. Save rollout metrics.

Success thresholds:
- `reach`: EE distance to cube <= `0.05`
- `avoid_reach`: EE distance to cube offset target <= `0.05`

Metrics per model and eval group:
- success rate
- success count
- mean best distance
- mean final distance
- mean steps
- mean success steps
- rollout count

Aggregate metrics:
- mean and 95% CI across train seeds
- separate ID and OOD aggregates

Derived metrics:

```text
generalization_gap = ID_success_rate - OOD_success_rate
diversity_gain = OOD_success_rate(diverse_or_wide) - OOD_success_rate(fixed_or_narrow)
```

## Plots for project video

Required plots:
- ID success vs budget
- OOD success vs budget
- generalization gap vs budget
- diversity gain at budget 50

Make plots split by:
- task: `reach`, `avoid_reach`
- axis: color, spatial, camera, lighting

## Execution order

1. Implement `research2` BC model, dataset loader, trainer.
2. Run a smoke train:
   - `color_red_only`
   - budget `5`
   - seed `0`
   - one or two epochs
3. Implement closed-loop evaluator.
4. Collect ID eval scene banks.
5. Run smoke eval:
   - one trained model
   - small ID subset
   - small OOD subset
6. Launch full 144-model training on role-lab.
7. Launch ID/OOD evaluation.
8. Aggregate results and make plots.

