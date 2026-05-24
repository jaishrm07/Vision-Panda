# 128px Dataset Specification

## Purpose

This document defines the dataset we will collect in `research2` before running experiments.

The goal is to create a clean 128x128 V-BCOOD dataset bank that measures which visual diversity axes matter for closed-loop behavior cloning.

We will not start full collection until this spec is locked.

## Workspace

All 128px experiment work should live under:

```text
research2/
```

Planned dataset output:

```text
research2/results/datasets_128px_v1/
```

Planned preview output:

```text
research2/results/previews_128px_v1/
```

## Global Collection Settings

- image resolution: `128x128`
- cameras: external/static RGB camera + end-effector RGB camera
- tasks:
  - `reach`
  - `avoid_reach`
- demo budgets:
  - `5`
  - `20`
  - `50`
- seeds:
  - `0`
  - `1`
  - `2`
- step cap per demo: `400`
- collection mode: headless
- policy target: scripted expert action
- metadata: required for every dataset

The step cap is a maximum. Current collectors stop early on success, so this is a fixed-demo-budget dataset, not a perfectly fixed-transition-budget dataset.

## Locked Visual Diversity Axes

We will collect one unified dataset for exactly four visual diversity axes:

```text
color
spatial distribution
camera location / viewpoint
lighting direction + intensity
```

There is no Phase A or Phase B split.

### Axis 1: Object Color

Question:

Does target appearance diversity improve OOD robustness?

Variation design:

- what varies: target object color
- fixed train condition: one target color, initially red
- diverse train condition: multiple target colors, initially red, green, blue, and yellow
- ID evaluation: same color or same color set as training
- OOD evaluation: held-out target colors, initially cyan, magenta, purple, and orange
- held fixed while measuring color: target shape, target size, spatial distribution, camera, lighting, and obstacle geometry

Train configs:

- `color_red_only`
- `color_multi`
- `avoid_color_red_only`
- `avoid_color_multi`

Eval configs:

- `red_id`
- `blue_ood`
- `color_multi_id`
- `color_heldout_ood`
- `avoid_red_id`
- `avoid_blue_ood`
- `avoid_color_multi_id`
- `avoid_color_heldout_ood`

Expected dataset count:

```text
4 train configs * 3 budgets * 3 seeds = 36 datasets
```

### Axis 2: Object Spatial Distribution

Question:

Does broader object-position support improve closed-loop robustness under spatial shift?

Variation design:

- what varies: target object x/y position on the table
- narrow train condition: target sampled from a small central reachable region
- wide train condition: target sampled from a larger reachable workspace region
- ID evaluation: target positions sampled from the same train region
- OOD evaluation: target positions near workspace edges or just outside the wide train region while still physically reachable
- held fixed while measuring spatial distribution: target color, target shape, target size, camera, lighting, and obstacle geometry

Train configs:

- `spatial_narrow`
- `spatial_wide`
- `avoid_spatial_narrow`
- `avoid_spatial_wide`

Eval configs:

- `spatial_narrow_id`
- `spatial_wide_id`
- `spatial_edge_ood`
- `spatial_outside_wide_ood`
- `avoid_spatial_narrow_id`
- `avoid_spatial_wide_id`
- `avoid_spatial_edge_ood`
- `avoid_spatial_outside_wide_ood`

Expected dataset count:

```text
4 train configs * 3 budgets * 3 seeds = 36 datasets
```

### Axis 3: Camera Location / Viewpoint

Question:

Does varying camera location/viewpoint improve robustness to viewpoint shift?

This includes the professor-suggested axis:

```text
the camera location would be good to vary
```

Variation design:

- what varies: external camera location/viewpoint
- fixed train condition: one camera pose looking at the workspace
- diverse train condition: multiple camera poses sampled around the workspace
- variation dimensions: camera azimuth/yaw, elevation/pitch, and distance from the workspace
- ID evaluation: camera poses sampled from the same train pose range
- OOD evaluation: more extreme viewpoints that still keep the robot, target, obstacle, and table visible
- held fixed while measuring camera location/viewpoint: target color, spatial distribution, lighting, and obstacle geometry

Train configs:

- `camera_fixed`
- `camera_multi_pose`
- `avoid_camera_fixed`
- `avoid_camera_multi_pose`

Eval configs:

- `camera_fixed_id`
- `camera_multi_id`
- `camera_extreme_ood`
- `avoid_camera_fixed_id`
- `avoid_camera_multi_id`
- `avoid_camera_extreme_ood`

Expected dataset count:

```text
4 train configs * 3 budgets * 3 seeds = 36 datasets
```

Not varied in this dataset:

- camera field of view
- end-effector camera placement
- obstacle geometry

The locked camera axis for this dataset is external-camera viewpoint variation through camera yaw, pitch, and distance.

### Axis 4: Lighting Direction And Intensity

Question:

Does lighting diversity improve robustness to illumination shift?

This includes the professor-suggested axis:

```text
you can also vary the direction and intensity of the light
```

Variation design:

- what varies: light direction and light intensity
- fixed train condition: one neutral light setup
- diverse train condition: multiple light directions and brightness levels
- direction examples: front, back, left, right, and top-biased lighting
- intensity examples: dim, medium, and bright lighting
- ID evaluation: lighting sampled from the same train range
- OOD evaluation: unseen light directions or stronger dim/bright settings that still preserve task visibility
- held fixed while measuring lighting: target color, spatial distribution, camera, and obstacle geometry

Train configs:

- `lighting_fixed`
- `lighting_diverse`
- `avoid_lighting_fixed`
- `avoid_lighting_diverse`

Eval configs:

- `lighting_fixed_id`
- `lighting_diverse_id`
- `lighting_extreme_ood`
- `avoid_lighting_fixed_id`
- `avoid_lighting_diverse_id`
- `avoid_lighting_extreme_ood`

Required before full collection:

- implement train/eval configs for both `reach` and `avoid_reach`
- implement default eval mappings
- implement train/eval relation entries
- preview fixed/diverse/extreme lighting at 128px

Expected dataset count:

```text
4 train configs * 3 budgets * 3 seeds = 36 datasets
```

## Locked Dataset Matrix

Full dataset:

```text
4 axes * 4 train configs per axis * 3 budgets * 3 seeds = 144 datasets
```

Breakdown:

- color: `36`
- spatial: `36`
- camera location/viewpoint: `36`
- lighting direction/intensity: `36`

This is the cleanest matrix because every axis has both `reach` and `avoid_reach`.

## Locked Train Config List

The full collection will include these 16 train configs:

- `color_red_only`
- `color_multi`
- `avoid_color_red_only`
- `avoid_color_multi`
- `spatial_narrow`
- `spatial_wide`
- `avoid_spatial_narrow`
- `avoid_spatial_wide`
- `camera_fixed`
- `camera_multi_pose`
- `avoid_camera_fixed`
- `avoid_camera_multi_pose`
- `lighting_fixed`
- `lighting_diverse`
- `avoid_lighting_fixed`
- `avoid_lighting_diverse`

Each train config will be collected at:

- budgets: `5`, `20`, `50`
- seeds: `0`, `1`, `2`

This gives:

```text
16 train configs * 3 budgets * 3 seeds = 144 datasets
```

## Storage Estimate

The existing 64px 108-dataset bank is about `19 GB`.

At 128px:

- pixel count is 4x larger
- 108 datasets are expected to be about `75 GB`
- 144 datasets are expected to be about `100 GB`

Recommended free space before full collection:

```text
150 GB minimum
180 GB safer
```

## Dataset Naming

Use this naming pattern:

```text
dataset__<train_config>__budget<DDD>__seed<SSS>.pkl
dataset__<train_config>__budget<DDD>__seed<SSS>.json
```

Examples:

```text
dataset__camera_multi_pose__budget020__seed001.pkl
dataset__lighting_diverse__budget050__seed002.pkl
dataset__avoid_lighting_fixed__budget005__seed000.pkl
```

## Metadata Requirements

Every metadata JSON must include:

- benchmark version
- output path
- train config name
- task name
- factor name
- image height and width
- number of demos
- step cap per demo
- number of collected samples
- seed
- full scene config
- per-demo summaries
- cube color and RGBA
- cube position
- camera parameters
- lighting parameters when present
- obstacle metadata when present
- distractor metadata when present

For lighting datasets, metadata must include:

- `lightDirection`
- `lightColor`
- `lightAmbientCoeff`
- `lightDiffuseCoeff`
- `lightSpecularCoeff`

## Quality Gates Before Full Collection

### Gate 1: Code Completeness

Required:

- collection runner exists under `research2/code/`
- all locked train configs exist
- all locked eval configs exist
- default eval mappings exist
- train/eval relation mappings exist
- metadata records resolution, camera, and lighting

Implementation work:

- write or adapt a full dataset collection runner under `research2/code/`
- implement all locked train/eval configs for `reach` and obstacle-avoidance reach
- generate all preview scenes at `128x128`

### Gate 2: 128px Preview

Generate preview images for all train configs at 128px before collecting datasets.

Pass criteria:

- external camera is not blank
- end-effector camera is not blank
- cube is visible
- obstacle is visible for `avoid_reach`
- camera-diverse views keep the workspace in frame
- lighting-diverse images visibly vary but do not hide the task
- extreme lighting is still usable for evaluation

### Gate 3: Smoke Collection

Before full collection, collect:

- one color config
- one spatial config
- one camera config
- one lighting config
- one avoid config

Use:

- budget `1`
- seed `0`
- step cap `40`
- resolution `128x128`

Pass criteria:

- all `.pkl` files are created
- all `.json` files are created
- images are `128x128`
- metadata is complete
- sample counts are nonzero
- no renderer errors

### Gate 4: Full Collection

Collect the locked `144` dataset matrix only after Gates 1-3 pass.

Expected outputs:

- `144` `.pkl` files
- `144` `.json` files
- preview index
- collection summary JSON

## Collection Order

Collect in this order:

1. color
2. spatial
3. camera location/viewpoint
4. lighting direction/intensity

Reason:

- color/spatial/camera are closest to the existing working benchmark
- lighting has higher renderer risk and should run last

## Locked Decision

The locked dataset to collect is:

```text
128x128, reach + avoid_reach, 4 axes, 3 budgets, 3 seeds, 144 total datasets
```

We should not start full collection until:

- avoid-lighting configs are added
- a `research2` collection runner is ready
- 128px previews pass inspection
- smoke collection passes
