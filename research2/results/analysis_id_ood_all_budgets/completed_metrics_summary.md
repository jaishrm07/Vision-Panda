# Completed ID/OOD Metrics Summary

Generated: 2026-05-07T18:45:37.307429+00:00

Scope: precision evaluation metrics for ID and OOD splits only. The old Scratch CNN coarse 5cm early-stop split is excluded.

Budgets 5/20/50 use seeds 0/1/2. Budgets 100/200 use seed 0 only, so treat high-budget trends as directional until repeated with more seeds.

## Scratch CNN Budget Scaling

| Split | Budget | success@1cm | success@2cm | success@5cm | nearest | end | metric files | rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID | 5 | 19.4% | 54.8% | 84.4% | 3.2 cm | 38.7 cm | 48 | 7200 |
| ID | 20 | 27.2% | 66.7% | 87.7% | 2.6 cm | 47.0 cm | 48 | 7200 |
| ID | 50 | 39.4% | 64.3% | 87.1% | 2.7 cm | 39.5 cm | 48 | 7200 |
| ID | 100 | 44.5% | 73.7% | 94.7% | 1.8 cm | 44.1 cm | 16 | 2400 |
| ID | 200 | 43.0% | 77.0% | 96.2% | 1.5 cm | 35.6 cm | 16 | 2400 |
| OOD | 5 | 21.6% | 51.3% | 79.4% | 4.5 cm | 39.5 cm | 48 | 7200 |
| OOD | 20 | 26.1% | 58.0% | 80.0% | 4.2 cm | 46.4 cm | 48 | 7200 |
| OOD | 50 | 30.6% | 54.7% | 78.5% | 4.4 cm | 40.4 cm | 48 | 7200 |
| OOD | 100 | 31.5% | 59.0% | 77.8% | 4.1 cm | 41.0 cm | 16 | 2400 |
| OOD | 200 | 43.5% | 63.7% | 86.6% | 3.0 cm | 37.3 cm | 16 | 2400 |

## OOD Axis, Budget 50

| Model | Axis | success@1cm | success@2cm | success@5cm | nearest | end |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen ResNet-18 | Camera viewpoint | 9.0% | 29.7% | 85.6% | 4.0 cm | 25.0 cm |
| Frozen ResNet-18 | Color | 8.3% | 37.3% | 97.7% | 3.2 cm | 22.6 cm |
| Frozen ResNet-18 | Lighting | 9.2% | 18.3% | 66.2% | 4.0 cm | 26.1 cm |
| Frozen ResNet-18 | Spatial distribution | 7.4% | 21.4% | 44.8% | 8.0 cm | 21.4 cm |
| Partial ResNet-18 | Camera viewpoint | 4.7% | 14.4% | 56.0% | 5.5 cm | 23.6 cm |
| Partial ResNet-18 | Color | 8.6% | 22.4% | 58.9% | 4.4 cm | 27.8 cm |
| Partial ResNet-18 | Lighting | 1.9% | 10.6% | 61.0% | 6.1 cm | 28.0 cm |
| Partial ResNet-18 | Spatial distribution | 3.4% | 11.7% | 31.8% | 8.9 cm | 26.6 cm |
| Scratch CNN | Camera viewpoint | 57.9% | 73.4% | 91.5% | 2.9 cm | 44.3 cm |
| Scratch CNN | Color | 41.1% | 83.3% | 100.0% | 1.2 cm | 48.0 cm |
| Scratch CNN | Lighting | 20.6% | 49.3% | 91.2% | 2.9 cm | 41.8 cm |
| Scratch CNN | Spatial distribution | 2.9% | 12.7% | 31.4% | 10.8 cm | 27.5 cm |

## OOD Axis, Budget 200

| Model | Axis | success@1cm | success@2cm | success@5cm | nearest | end |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Frozen ResNet-18 | Camera viewpoint | 26.7% | 34.2% | 95.8% | 3.0 cm | 22.1 cm |
| Frozen ResNet-18 | Color | 13.2% | 50.0% | 100.0% | 2.5 cm | 22.4 cm |
| Frozen ResNet-18 | Lighting | 0.0% | 14.7% | 78.0% | 4.4 cm | 26.0 cm |
| Frozen ResNet-18 | Spatial distribution | 15.2% | 27.2% | 47.7% | 7.0 cm | 21.7 cm |
| Partial ResNet-18 | Camera viewpoint | 1.0% | 42.8% | 76.2% | 4.6 cm | 21.8 cm |
| Partial ResNet-18 | Color | 43.7% | 43.7% | 50.0% | 4.7 cm | 29.9 cm |
| Partial ResNet-18 | Lighting | 25.0% | 37.3% | 62.2% | 5.9 cm | 16.5 cm |
| Partial ResNet-18 | Spatial distribution | 6.3% | 16.2% | 34.7% | 15.0 cm | 37.7 cm |
| Scratch CNN | Camera viewpoint | 61.7% | 68.8% | 99.8% | 1.1 cm | 49.7 cm |
| Scratch CNN | Color | 57.0% | 100.0% | 100.0% | 0.8 cm | 35.9 cm |
| Scratch CNN | Lighting | 50.0% | 62.8% | 100.0% | 1.3 cm | 41.8 cm |
| Scratch CNN | Spatial distribution | 5.2% | 23.0% | 46.5% | 8.9 cm | 21.8 cm |
