# Scratch CNN Results

Scope: completed Scratch CNN precision eval metrics for budgets 5, 20, and 50. Each metric file evaluates 150 rollouts. High-budget 100/200 evals are excluded until complete.


Distance definitions:
- `nearest cm`: mean of the closest target distance reached at any timestep in each rollout. Success@0.5/1/2/5cm is computed from this value per rollout.
- `end cm`: mean target distance at the final rollout timestep. This measures whether the policy stays near the target or drifts away.

## Overall

| split | budget | n | succ@0.5cm % | succ@1cm % | succ@2cm % | succ@5cm % | nearest cm | end cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID | 5 | 48 | 4.3 | 19.4 | 54.8 | 84.4 | 3.2 | 38.7 |
| ID | 20 | 48 | 4.4 | 27.2 | 66.7 | 87.7 | 2.6 | 47.0 |
| ID | 50 | 48 | 25.8 | 39.4 | 64.3 | 87.1 | 2.7 | 39.5 |
| OOD | 5 | 48 | 4.8 | 21.6 | 51.3 | 79.4 | 4.5 | 39.5 |
| OOD | 20 | 48 | 4.8 | 26.1 | 58.0 | 80.0 | 4.2 | 46.4 |
| OOD | 50 | 48 | 16.1 | 30.6 | 54.7 | 78.5 | 4.4 | 40.4 |

## OOD By Axis

| budget | axis | n | succ@1cm % | succ@2cm % | succ@5cm % | nearest cm | end cm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | color | 12 | 33.3 | 66.7 | 100.0 | 1.7 | 41.2 |
| 5 | spatial_distribution | 12 | 1.7 | 5.1 | 17.5 | 13.0 | 37.1 |
| 5 | camera_location_viewpoint | 12 | 33.3 | 75.0 | 100.0 | 1.6 | 41.7 |
| 5 | lighting_direction_intensity | 12 | 18.1 | 58.3 | 100.0 | 1.9 | 38.0 |
| 20 | color | 12 | 25.0 | 85.5 | 100.0 | 1.5 | 49.2 |
| 20 | spatial_distribution | 12 | 2.3 | 9.4 | 25.1 | 11.6 | 35.9 |
| 20 | camera_location_viewpoint | 12 | 55.7 | 83.2 | 100.0 | 1.2 | 57.5 |
| 20 | lighting_direction_intensity | 12 | 21.6 | 53.8 | 94.8 | 2.4 | 42.8 |
| 50 | color | 12 | 41.1 | 83.3 | 100.0 | 1.2 | 48.0 |
| 50 | spatial_distribution | 12 | 2.9 | 12.7 | 31.4 | 10.8 | 27.5 |
| 50 | camera_location_viewpoint | 12 | 57.9 | 73.4 | 91.5 | 2.9 | 44.3 |
| 50 | lighting_direction_intensity | 12 | 20.6 | 49.3 | 91.2 | 2.9 | 41.8 |

## OOD By Task

| budget | task | n | succ@1cm % | succ@2cm % | succ@5cm % | nearest cm |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | reach | 24 | 30.0 | 59.1 | 78.2 | 4.5 |
| 5 | avoid_reach | 24 | 13.2 | 43.5 | 80.5 | 4.6 |
| 20 | reach | 24 | 43.4 | 69.9 | 79.8 | 4.0 |
| 20 | avoid_reach | 24 | 8.8 | 46.0 | 80.2 | 4.4 |
| 50 | reach | 24 | 57.0 | 69.9 | 80.8 | 3.6 |
| 50 | avoid_reach | 24 | 4.2 | 39.4 | 76.3 | 5.2 |

## OOD Diversity Pair Gains

| task | axis | pair @ budget50 | gain @1cm pp | gain @2cm pp | gain @5cm pp |
| --- | --- | --- | --- | --- | --- |
| reach | color | color_red_only -> color_multi | 35.6 | 0.0 | 0.0 |
| avoid_reach | color | avoid_color_red_only -> avoid_color_multi | 0.0 | 0.0 | 0.0 |
| reach | spatial_distribution | spatial_narrow -> spatial_wide | 0.4 | 14.2 | 33.6 |
| avoid_reach | spatial_distribution | avoid_spatial_narrow -> avoid_spatial_wide | 0.9 | 7.6 | 13.8 |
| reach | camera_location_viewpoint | camera_fixed -> camera_multi_pose | 0.0 | 0.0 | 0.0 |
| avoid_reach | camera_location_viewpoint | avoid_camera_fixed -> avoid_camera_multi_pose | -31.8 | -26.9 | -32.7 |
| reach | lighting_direction_intensity | lighting_fixed -> lighting_diverse | -12.0 | 0.0 | -10.7 |
| avoid_reach | lighting_direction_intensity | avoid_lighting_fixed -> avoid_lighting_diverse | 0.0 | -11.8 | 6.9 |
