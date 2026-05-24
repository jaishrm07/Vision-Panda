# 128px Visual Diversity Dataset Plan

## Tracking Policy

Before any implementation change, add a move entry to `WORK_LOG.md` describing the intended action.
After the change, add another move entry describing what changed and how it was verified.

This project is starting from scratch on `role-lab` at:

```text
/home/jaisharma/HW8/research2
```

## Dataset Goal

Collect one unified 128x128 visual-diversity dataset for behavior cloning.

The locked visual diversity axes are:

```text
color
spatial distribution
camera location / viewpoint
lighting direction + intensity
```

There is no phase split. The full dataset includes all four axes.

## Lighting Baseline Rule

For the color, spatial distribution, and camera location/viewpoint axes, use PyBullet default renderer lighting. This matches the original `homework_archive` setup, which does not pass explicit light parameters to `p.getCameraImage`.

For the lighting direction/intensity axis only, pass explicit light parameters so that fixed-lighting and diverse-lighting conditions are controlled by the dataset config.

## Tasks

### Reach

The robot moves its end-effector to the target cube.

### Obstacle-Avoidance Reach

The robot reaches the target cube while avoiding a fixed static obstacle placed between the robot and the target.

Presentation name:

```text
Obstacle-Avoidance Reach
```

Internal short name:

```text
avoid_reach
```

The obstacle is part of the task, not a visual diversity axis. Its geometry should stay fixed while measuring color, spatial distribution, camera viewpoint, and lighting.

Recommended obstacle:

```text
shape: rectangular wall/block
material/color: matte neutral gray
behavior: static
placement: between robot start region and target region
```

## Global Collection Settings

```text
resolution: 128x128
tasks: reach, obstacle-avoidance reach
cameras: external RGB camera + end-effector RGB camera
budgets: 5, 20, 50 demonstrations
seeds: 0, 1, 2
step cap per demo: 400
collection mode: headless
```

Each dataset sample should include:

```text
external_rgb
eef_rgb
robot_state
expert_action
task distance / success signal
metadata
```

## Axis 1: Color

Question:

Does target color diversity improve robustness to appearance shift?

Variation design:

- what varies: target object color
- fixed train condition: one target color, initially red
- diverse train condition: multiple target colors, initially red, green, blue, and yellow
- ID evaluation: same color or same color set as training
- OOD evaluation: held-out colors, initially cyan, magenta, purple, and orange
- held fixed: target shape, target size, spatial distribution, camera, lighting, and obstacle geometry

Train configs:

```text
color_red_only
color_multi
avoid_color_red_only
avoid_color_multi
```

## Axis 2: Spatial Distribution

Question:

Does broader target-position support improve robustness under spatial shift?

Variation design:

- what varies: target object x/y position on the table
- narrow train condition: target sampled from a small central reachable region
- wide train condition: target sampled from a larger reachable workspace region
- ID evaluation: positions sampled from the same train region
- OOD evaluation: positions near workspace edges or just outside the wide train region while still reachable
- held fixed: target color, target shape, target size, camera, lighting, and obstacle geometry

Train configs:

```text
spatial_narrow
spatial_wide
avoid_spatial_narrow
avoid_spatial_wide
```

## Axis 3: Camera Location / Viewpoint

Question:

Does varying camera location/viewpoint improve robustness to viewpoint shift?

Variation design:

- what varies: external camera location/viewpoint
- fixed train condition: one camera pose looking at the workspace
- diverse train condition: multiple camera poses sampled around the workspace
- variation dimensions: azimuth/yaw, elevation/pitch, distance from workspace
- ID evaluation: camera poses sampled from the same train pose range
- OOD evaluation: more extreme viewpoints that still keep robot, target, obstacle, and table visible
- held fixed: target color, spatial distribution, lighting, and obstacle geometry

Train configs:

```text
camera_fixed
camera_multi_pose
avoid_camera_fixed
avoid_camera_multi_pose
```

## Axis 4: Lighting Direction And Intensity

Question:

Does lighting diversity improve robustness to illumination shift?

Variation design:

- what varies: light direction and light intensity
- fixed train condition: one neutral light setup
- diverse train condition: multiple light directions and brightness levels
- direction examples: front, back, left, right, top-biased
- intensity examples: dim, medium, bright
- ID evaluation: lighting sampled from the same train range
- OOD evaluation: unseen directions or stronger dim/bright settings that still preserve visibility
- held fixed: target color, spatial distribution, camera, and obstacle geometry
- implementation rule: explicit light parameters are passed only for this axis; other axes use PyBullet default lighting

Train configs:

```text
lighting_fixed
lighting_diverse
avoid_lighting_fixed
avoid_lighting_diverse
```

## Locked Dataset Matrix

```text
16 train configs * 3 budgets * 3 seeds = 144 datasets
```

Breakdown:

```text
color: 36 datasets
spatial distribution: 36 datasets
camera location / viewpoint: 36 datasets
lighting direction + intensity: 36 datasets
```

## Quality Gates

### Gate 1: Preview

Generate 128x128 preview images for all train conditions.

Pass criteria:

- external camera image is not blank
- end-effector camera image is not blank
- target is visible
- obstacle is visible for obstacle-avoidance reach
- camera viewpoints keep the task in frame
- lighting variation is visible but not destructive

### Gate 2: Smoke Collection

Collect a small mixed subset before the full run.

```text
budget: 1
seed: 0
step cap: 40
resolution: 128x128
```

Pass criteria:

- dataset files are created
- metadata files are created
- images are 128x128
- sample count is nonzero
- no renderer errors

### Gate 3: Full Collection

Run full collection only after preview and smoke collection pass.

Expected output:

```text
144 dataset files
144 metadata files
collection summary
preview index
```
