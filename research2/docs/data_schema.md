# Dataset Data Schema

This document defines the expected dataset file format for the 128px visual-diversity collection.

## File Naming

Each collected dataset should produce one data file and one metadata file.

```text
results/datasets_128px_v1/dataset__<train_config>__budget<DDD>__seed<SSS>.pkl
results/datasets_128px_v1/dataset__<train_config>__budget<DDD>__seed<SSS>.json
```

Examples:

```text
dataset__color_multi__budget020__seed001.pkl
dataset__avoid_camera_multi_pose__budget050__seed002.json
dataset__lighting_diverse__budget005__seed000.pkl
```

## Dataset File

The `.pkl` file should contain a dictionary with these top-level keys:

```text
version
train_config
task
axis
variant
budget
seed
resolution
samples
episodes
```

## Sample Schema

Each entry in `samples` should be a dictionary.

Required keys:

```text
episode_index
step_index
external_rgb
eef_rgb
robot_state
expert_action
task_distance
success
scene
```

Image fields:

```text
external_rgb: uint8 array, shape [128, 128, 3]
eef_rgb: uint8 array, shape [128, 128, 3]
```

Robot state should include at least:

```text
ee_position
joint_positions
joint_velocities
```

Expert action should include:

```text
delta_position
```

The first implementation can use Cartesian delta-position actions. If we later use joint actions, record the action type explicitly in metadata.

## Scene Schema

Each sample and each episode summary should be able to resolve the full visual scene.

Required scene fields:

```text
task_name
target_position
target_color_name
target_rgba
camera
lighting
obstacle
```

For `reach`, `obstacle` should be null.

For `avoid_reach`, `obstacle` should include:

```text
shape
center
half_extents
rgba
```

## Camera Schema

External camera fields:

```text
yaw
pitch
distance
target_position
fov
near
far
view_matrix
projection_matrix
```

End-effector camera fields:

```text
mount_name
width
height
fov
near
far
view_matrix
projection_matrix
```

## Lighting Schema

Lighting fields:

```text
mode
light_direction
light_color
light_ambient_coeff
light_diffuse_coeff
light_specular_coeff
```

For non-lighting visual diversity axes, lighting should be recorded as:

```text
mode: pybullet_default
```

In that case, do not pass explicit light parameters to `p.getCameraImage`; use the same PyBullet renderer defaults as `homework_archive`.

For the lighting direction/intensity axis, lighting should be recorded as:

```text
mode: explicit
```

and should include `light_direction`, `light_color`, `light_ambient_coeff`, `light_diffuse_coeff`, and `light_specular_coeff`.

## Episode Summary Schema

Each entry in `episodes` should contain:

```text
episode_index
num_steps
success
final_task_distance
initial_scene
final_scene
termination_reason
```

Valid termination reasons:

```text
success
step_cap
simulation_error
```

## Metadata JSON Schema

The `.json` metadata file should be human-readable and should not duplicate full image arrays.

Required top-level keys:

```text
version
created_at
workspace
python_executable
train_config
task
axis
variant
budget
seed
max_steps_per_demo
resolution
num_samples
num_episodes
success_count
success_rate
sample_file
config_file
scene_distribution
quality_checks
```

`scene_distribution` should summarize the exact ranges or presets used for the collected dataset.

`quality_checks` should include:

```text
external_rgb_shape_ok
eef_rgb_shape_ok
external_rgb_nonblank
eef_rgb_nonblank
metadata_complete
sample_count_nonzero
```

## Preview Output Schema

Preview files should be written under:

```text
results/previews_128px_v1/
```

Preview naming:

```text
preview__<train_config>__seed<SSS>__sample<NNN>__external.png
preview__<train_config>__seed<SSS>__sample<NNN>__eef.png
```

Preview index:

```text
results/previews_128px_v1/preview_index.json
```

The preview index should record:

```text
train_config
seed
axis
variant
task
external_preview_path
eef_preview_path
scene
```

## Validation Rules

A dataset is valid only if:

```text
all external_rgb images are 128x128x3
all eef_rgb images are 128x128x3
image dtype is uint8
sample count is greater than zero
metadata JSON exists
metadata references the correct sample file
train_config, budget, and seed match the filename
lighting fields are present
camera fields are present
obstacle fields are present for avoid_reach
obstacle is null for reach
```

## Collection Matrix Check

Full collection should contain:

```text
16 train configs
3 budgets
3 seeds
144 pkl files
144 json files
```
