# Scratch CNN Task-Basis Results

Task labels:
- `Reach`: plain target reaching.
- `Obstacle-aware reach`: reaching with a static obstacle/wall between start and target region.

Distance definitions:
- `nearest cm`: closest target distance reached during rollout; success thresholds are computed from this.
- `end cm`: target distance at final timestep.

## OOD Task Results

| Budget | Task | n | succ@1cm | succ@2cm | succ@5cm | nearest cm | end cm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | Reach | 24 | 30.0% | 59.1% | 78.2% | 4.5 | 27.4 |
| 5 | Obstacle-aware reach | 24 | 13.2% | 43.5% | 80.5% | 4.6 | 51.6 |
| 20 | Reach | 24 | 43.4% | 69.9% | 79.8% | 4.0 | 16.6 |
| 20 | Obstacle-aware reach | 24 | 8.8% | 46.0% | 80.2% | 4.4 | 76.1 |
| 50 | Reach | 24 | 57.0% | 69.9% | 80.8% | 3.6 | 9.8 |
| 50 | Obstacle-aware reach | 24 | 4.2% | 39.4% | 76.3% | 5.2 | 71.1 |

## ID Task Results

| Budget | Task | n | succ@1cm | succ@2cm | succ@5cm | nearest cm | end cm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | Reach | 24 | 25.6% | 65.4% | 84.1% | 3.0 | 26.5 |
| 5 | Obstacle-aware reach | 24 | 13.2% | 44.2% | 84.8% | 3.4 | 50.9 |
| 20 | Reach | 24 | 46.0% | 81.1% | 88.6% | 2.3 | 13.8 |
| 20 | Obstacle-aware reach | 24 | 8.5% | 52.2% | 86.9% | 3.0 | 80.3 |
| 50 | Reach | 24 | 78.3% | 86.5% | 92.0% | 1.3 | 6.1 |
| 50 | Obstacle-aware reach | 24 | 0.5% | 42.1% | 82.2% | 4.0 | 72.9 |
