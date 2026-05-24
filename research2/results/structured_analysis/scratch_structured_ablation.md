# Scratch Structured Ablation

| Training setup | Model | Split | N | S@1cm | S@2cm | S@5cm | Best cm | Final cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scratch ablation | Phase only | ID | 150 | 0.0% | 0.7% | 6.0% | 19.21 | 22.11 |
| Scratch ablation | Phase only | OOD | 150 | 0.0% | 2.0% | 10.7% | 14.60 | 17.27 |
| Scratch ablation | Target geometry only | ID | 150 | 22.7% | 99.3% | 100.0% | 1.25 | 3.52 |
| Scratch ablation | Target geometry only | OOD | 150 | 6.7% | 94.0% | 100.0% | 1.46 | 3.17 |
| Scratch ablation | Full geometry only | ID | 150 | 31.3% | 82.7% | 100.0% | 1.27 | 4.30 |
| Scratch ablation | Full geometry only | OOD | 150 | 21.3% | 82.0% | 100.0% | 1.45 | 3.80 |
| Scratch ablation | Phase + geometry | ID | 150 | 64.7% | 74.0% | 74.7% | 3.01 | 3.27 |
| Scratch ablation | Phase + geometry | OOD | 150 | 53.3% | 55.3% | 58.7% | 4.05 | 4.23 |
