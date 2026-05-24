# Main Comparison

| Training setup | Model | Split | N | S@1cm | S@2cm | S@5cm | Best cm | Final cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prior spatial-wide visual-only | Scratch CNN | ID | 150 | 30.7% | 56.7% | 78.0% | 3.20 | 5.27 |
| Prior spatial-wide visual-only | Scratch CNN | OOD | 150 | 15.3% | 38.0% | 71.3% | 5.11 | 8.73 |
| Prior spatial-wide visual-only | Frozen ResNet18 | ID | 150 | 0.7% | 22.7% | 82.7% | 3.77 | 7.32 |
| Prior spatial-wide visual-only | Frozen ResNet18 | OOD | 150 | 2.7% | 18.0% | 55.3% | 6.18 | 14.86 |
| Prior spatial-wide visual-only | Partial ResNet18 | ID | 150 | 43.3% | 74.0% | 94.7% | 1.79 | 6.85 |
| Prior spatial-wide visual-only | Partial ResNet18 | OOD | 150 | 28.0% | 52.0% | 74.7% | 4.13 | 14.52 |
| Edge-balanced visual-only | Scratch CNN | ID | 150 | 0.0% | 1.3% | 4.7% | 17.16 | 23.08 |
| Edge-balanced visual-only | Scratch CNN | OOD | 150 | 1.3% | 3.3% | 14.7% | 12.95 | 19.25 |
| Edge-balanced visual-only | Frozen ResNet18 | ID | 150 | 10.7% | 32.7% | 62.7% | 7.62 | 15.22 |
| Edge-balanced visual-only | Frozen ResNet18 | OOD | 150 | 8.7% | 21.3% | 51.3% | 7.68 | 14.98 |
| Edge-balanced visual-only | Partial ResNet18 | ID | 150 | 0.0% | 0.0% | 4.0% | 15.10 | 68.16 |
| Edge-balanced visual-only | Partial ResNet18 | OOD | 150 | 0.0% | 0.0% | 8.0% | 13.73 | 65.61 |
| Edge-balanced phase+geometry | Scratch CNN | ID | 150 | 64.7% | 74.0% | 74.7% | 3.01 | 3.27 |
| Edge-balanced phase+geometry | Scratch CNN | OOD | 150 | 53.3% | 55.3% | 58.7% | 4.05 | 4.23 |
| Edge-balanced phase+geometry | Frozen ResNet18 | ID | 150 | 10.0% | 40.7% | 71.3% | 3.46 | 6.94 |
| Edge-balanced phase+geometry | Frozen ResNet18 | OOD | 150 | 3.3% | 22.7% | 52.7% | 4.66 | 7.02 |
| Edge-balanced phase+geometry | Partial ResNet18 | ID | 150 | 8.7% | 23.3% | 52.0% | 5.22 | 11.08 |
| Edge-balanced phase+geometry | Partial ResNet18 | OOD | 150 | 10.0% | 26.7% | 49.3% | 4.90 | 11.66 |
