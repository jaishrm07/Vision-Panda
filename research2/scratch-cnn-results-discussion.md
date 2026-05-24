# Completed Results Discussion

## Scope

This discussion summarizes the completed precision evals for:

- Scratch CNN
- Frozen ResNet-18 encoder
- Partially fine-tuned ResNet-18 encoder

Included splits:

- ID eval
- OOD eval

Included budgets:

- 5, 20, 50 demonstrations with seeds 0, 1, 2
- 100, 200 demonstrations with seed 0 only

Important caveat:

- Budgets 100 and 200 are directional high-budget results because they currently use one seed. The 5/20/50 results are stronger statistically because they use three seeds.
- The old Scratch CNN coarse 5cm early-stop eval is excluded. These tables use the corrected precision evaluator: success thresholds are computed from nearest distance, and early stop only happens at 0.1 cm.

Primary analysis artifacts:

- `results/analysis_id_ood_all_budgets/per_model_metrics.csv`
- `results/analysis_id_ood_all_budgets/overall_by_family_split_budget.csv`
- `results/analysis_id_ood_all_budgets/ood_axis_by_family_budget.csv`
- `results/analysis_id_ood_all_budgets/ood_task_by_family_budget.csv`
- `results/analysis_id_ood_all_budgets/train_config_by_family_split_budget.csv`
- `results/analysis_id_ood_all_budgets/diversity_pair_gains_ood_by_family_budget.csv`
- `results/analysis_id_ood_all_budgets/completed_metrics_summary.md`

## Core Metrics

We should consistently report:

- success@1cm: precision success, computed from nearest distance reached during rollout.
- success@2cm: intermediate precision success.
- success@5cm: loose success, useful for comparison but can hide poor precision.
- nearest distance: closest target distance reached at any timestep.
- end distance: target distance at the final timestep.

Important distinction:

- Nearest distance tells us whether the policy ever got close.
- End distance tells us whether the policy stayed near the target or drifted away.

## Main Result Story

The cleanest story is:

1. Scratch CNN is the strongest model family for 1cm OOD precision overall.
2. More demos help Scratch CNN, especially at budget 200, but the improvement is not uniform across visual axes.
3. Spatial distribution shift remains the hardest visual OOD axis, even with 200 demos.
4. Frozen ResNet-18 is more stable in end distance, but its 1cm OOD precision is much weaker than Scratch CNN.
5. Partial ResNet-18 improves with more data but is inconsistent; it does not cleanly dominate frozen ResNet or Scratch CNN.
6. Obstacle-aware reach remains much harder than plain reach, especially in end stability.
7. success@5cm can substantially overstate policy quality compared with success@1cm.

## Main Comparisons

### Comparison 1: ID vs OOD

Question:

Does each policy degrade when visual conditions shift?

Compare:

- ID success@1cm vs OOD success@1cm
- ID success@5cm vs OOD success@5cm
- ID nearest distance vs OOD nearest distance
- ID end distance vs OOD end distance

Use budgets:

- 5
- 20
- 50
- 100
- 200

Why it matters:

This is the basic generalization comparison. It tells us whether the policy only fits the training visual distribution or transfers to held-out visual conditions.

### Comparison 2: Budget Scaling

Question:

Does more demonstration data improve performance?

Compare:

- 5 demonstrations
- 20 demonstrations
- 50 demonstrations
- 100 demonstrations
- 200 demonstrations

Primary metrics:

- success@1cm
- success@5cm
- nearest distance
- end distance

Why it matters:

This shows whether collecting more demonstrations improves precision and OOD robustness, or whether performance saturates.

### Comparison 3: Model Family

Question:

Do pretrained visual encoders help in this simulated pixel-control setting?

Compare:

- Scratch CNN
- Frozen ResNet-18
- Partial ResNet-18

Why it matters:

This directly addresses the professor's feedback. The result is not simply "pretraining helps." Frozen ResNet is more stable by end distance but less precise at 1cm. Scratch CNN has stronger OOD precision, especially for color, camera, and lighting axes.

### Comparison 4: Visual Diversity Axes

Question:

Which visual shift is hardest?

Compare these OOD axes:

- color
- spatial distribution
- camera location / viewpoint
- lighting direction + intensity

Recommended headline settings:

- budget 50 because it has three seeds
- budget 200 as a directional high-budget extension

Why it matters:

This directly answers which visual diversity axes matter most. Current results show that spatial distribution shift is the clearest persistent failure axis.

### Comparison 5: Task Basis

Question:

Does obstacle-aware reaching expose harder behavior than plain reaching?

Compare:

- Reach
- Obstacle-aware reach

Why it matters:

Obstacle-aware reach tests harder interaction geometry, not just visual appearance. It produces much worse precision and much worse end distance for Scratch CNN.

### Comparison 6: Precision vs Loose Success

Question:

Are loose success thresholds hiding poor control quality?

Compare:

- success@1cm
- success@2cm
- success@5cm

Why it matters:

Success@5cm can look high even when success@1cm is low. This is one of the strongest methodological points in the results.

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

Interpretation:

- Scratch CNN OOD success@1cm rises from 21.6% at budget 5 to 43.5% at budget 200.
- OOD nearest distance improves from 4.5 cm to 3.0 cm.
- End distance remains large, so higher budget improves reaching precision more than final stability.
- Budget 100/200 trends should be repeated with more seeds before making a strong statistical claim.

## Model Family OOD Scaling

| Model | Budget | success@1cm | success@5cm | nearest | end | metric files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch CNN | 5 | 21.6% | 79.4% | 4.5 cm | 39.5 cm | 48 |
| Scratch CNN | 20 | 26.1% | 80.0% | 4.2 cm | 46.4 cm | 48 |
| Scratch CNN | 50 | 30.6% | 78.5% | 4.4 cm | 40.4 cm | 48 |
| Scratch CNN | 100 | 31.5% | 77.8% | 4.1 cm | 41.0 cm | 16 |
| Scratch CNN | 200 | 43.5% | 86.6% | 3.0 cm | 37.3 cm | 16 |
| Frozen ResNet-18 | 5 | 7.7% | 58.5% | 6.5 cm | 25.7 cm | 48 |
| Frozen ResNet-18 | 20 | 4.8% | 64.8% | 6.3 cm | 26.4 cm | 48 |
| Frozen ResNet-18 | 50 | 8.5% | 73.6% | 4.8 cm | 23.8 cm | 48 |
| Frozen ResNet-18 | 100 | 12.9% | 78.1% | 4.8 cm | 23.1 cm | 16 |
| Frozen ResNet-18 | 200 | 13.8% | 80.4% | 4.2 cm | 23.0 cm | 16 |
| Partial ResNet-18 | 5 | 0.4% | 22.5% | 10.8 cm | 30.5 cm | 48 |
| Partial ResNet-18 | 20 | 1.3% | 33.0% | 9.3 cm | 27.9 cm | 48 |
| Partial ResNet-18 | 50 | 4.6% | 51.9% | 6.2 cm | 26.5 cm | 48 |
| Partial ResNet-18 | 100 | 8.7% | 61.5% | 5.7 cm | 28.0 cm | 16 |
| Partial ResNet-18 | 200 | 19.0% | 55.8% | 7.6 cm | 26.5 cm | 16 |

Interpretation:

- Scratch CNN is best for OOD success@1cm.
- Frozen ResNet-18 has lower end distance but lower precision.
- Partial ResNet-18 improves with more data but remains inconsistent.
- These results support a careful answer to the pretrained-encoder question: pretrained features do not automatically solve simulation visual imitation learning.

## OOD Visual Axis Breakdown

### Budget 50

Budget 50 is the strongest axis comparison because it uses three seeds.

| Model | Axis | success@1cm | success@2cm | success@5cm | nearest | end |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scratch CNN | Camera viewpoint | 57.9% | 73.4% | 91.5% | 2.9 cm | 44.3 cm |
| Scratch CNN | Color | 41.1% | 83.3% | 100.0% | 1.2 cm | 48.0 cm |
| Scratch CNN | Lighting | 20.6% | 49.3% | 91.2% | 2.9 cm | 41.8 cm |
| Scratch CNN | Spatial distribution | 2.9% | 12.7% | 31.4% | 10.8 cm | 27.5 cm |
| Frozen ResNet-18 | Camera viewpoint | 9.0% | 29.7% | 85.6% | 4.0 cm | 25.0 cm |
| Frozen ResNet-18 | Color | 8.3% | 37.3% | 97.7% | 3.2 cm | 22.6 cm |
| Frozen ResNet-18 | Lighting | 9.2% | 18.3% | 66.2% | 4.0 cm | 26.1 cm |
| Frozen ResNet-18 | Spatial distribution | 7.4% | 21.4% | 44.8% | 8.0 cm | 21.4 cm |
| Partial ResNet-18 | Camera viewpoint | 4.7% | 14.4% | 56.0% | 5.5 cm | 23.6 cm |
| Partial ResNet-18 | Color | 8.6% | 22.4% | 58.9% | 4.4 cm | 27.8 cm |
| Partial ResNet-18 | Lighting | 1.9% | 10.6% | 61.0% | 6.1 cm | 28.0 cm |
| Partial ResNet-18 | Spatial distribution | 3.4% | 11.7% | 31.8% | 8.9 cm | 26.6 cm |

### Budget 200

Budget 200 is a directional high-budget extension because it uses seed 0 only.

| Model | Axis | success@1cm | success@2cm | success@5cm | nearest | end |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scratch CNN | Camera viewpoint | 61.7% | 68.8% | 99.8% | 1.1 cm | 49.7 cm |
| Scratch CNN | Color | 57.0% | 100.0% | 100.0% | 0.8 cm | 35.9 cm |
| Scratch CNN | Lighting | 50.0% | 62.8% | 100.0% | 1.3 cm | 41.8 cm |
| Scratch CNN | Spatial distribution | 5.2% | 23.0% | 46.5% | 8.9 cm | 21.8 cm |
| Frozen ResNet-18 | Camera viewpoint | 26.7% | 34.2% | 95.8% | 3.0 cm | 22.1 cm |
| Frozen ResNet-18 | Color | 13.2% | 50.0% | 100.0% | 2.5 cm | 22.4 cm |
| Frozen ResNet-18 | Lighting | 0.0% | 14.7% | 78.0% | 4.4 cm | 26.0 cm |
| Frozen ResNet-18 | Spatial distribution | 15.2% | 27.2% | 47.7% | 7.0 cm | 21.7 cm |
| Partial ResNet-18 | Camera viewpoint | 1.0% | 42.8% | 76.2% | 4.6 cm | 21.8 cm |
| Partial ResNet-18 | Color | 43.7% | 43.7% | 50.0% | 4.7 cm | 29.9 cm |
| Partial ResNet-18 | Lighting | 25.0% | 37.3% | 62.2% | 5.9 cm | 16.5 cm |
| Partial ResNet-18 | Spatial distribution | 6.3% | 16.2% | 34.7% | 15.0 cm | 37.7 cm |

Interpretation:

- Spatial distribution remains difficult at both 50 and 200 demos.
- Scratch CNN improves substantially on lighting by budget 200, but spatial remains weak.
- Frozen ResNet-18 handles end distance better but does not reach high 1cm precision.
- Partial ResNet-18 has unstable high-budget behavior across axes.

## OOD Task Basis

### Budget 50

| Model | Task | success@1cm | success@2cm | success@5cm | nearest | end |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scratch CNN | Reach | 57.0% | 69.9% | 80.8% | 3.6 cm | 9.8 cm |
| Scratch CNN | Obstacle-aware reach | 4.2% | 39.4% | 76.3% | 5.2 cm | 71.1 cm |
| Frozen ResNet-18 | Reach | 12.6% | 45.6% | 80.8% | 3.4 cm | 28.7 cm |
| Frozen ResNet-18 | Obstacle-aware reach | 4.4% | 7.8% | 66.3% | 6.2 cm | 18.8 cm |
| Partial ResNet-18 | Reach | 4.6% | 17.6% | 50.2% | 5.9 cm | 22.3 cm |
| Partial ResNet-18 | Obstacle-aware reach | 4.6% | 11.9% | 53.7% | 6.5 cm | 30.6 cm |

### Budget 200

| Model | Task | success@1cm | success@2cm | success@5cm | nearest | end |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scratch CNN | Reach | 77.4% | 81.4% | 86.2% | 2.7 cm | 6.9 cm |
| Scratch CNN | Obstacle-aware reach | 9.5% | 45.9% | 86.9% | 3.4 cm | 67.7 cm |
| Frozen ResNet-18 | Reach | 26.6% | 60.1% | 87.2% | 2.6 cm | 26.5 cm |
| Frozen ResNet-18 | Obstacle-aware reach | 0.9% | 2.9% | 73.5% | 5.8 cm | 19.6 cm |
| Partial ResNet-18 | Reach | 37.2% | 57.0% | 89.2% | 2.6 cm | 10.3 cm |
| Partial ResNet-18 | Obstacle-aware reach | 0.8% | 13.0% | 22.3% | 12.5 cm | 42.6 cm |

Interpretation:

- Reach is much easier than obstacle-aware reach for precision.
- Scratch CNN budget 200 reach is strong at success@1cm, but obstacle-aware reach is still only 9.5% at success@1cm.
- Obstacle-aware reach can look acceptable at success@5cm while still failing precision and end stability.

## What Should Go In Main Paper

Main paper should include:

- Overall ID/OOD precision over budgets for all three model families.
- OOD visual-axis comparison at budget 50.
- Budget 200 as a directional high-budget extension.
- OOD reach vs obstacle-aware reach, with nearest and end distance.
- success@1cm vs success@5cm as the precision argument.

Appendix should include:

- Full per-config tables.
- Full diversity-pair gain tables.
- Seed-level results.
- success@0.5cm details.
- Full nearest/end distance tables.

## Possible Claims

Strong claims:

- Loose 5cm success substantially overestimates policy quality.
- Spatial distribution shift is the most persistent visual OOD failure axis.
- Obstacle-aware reach exposes precision and final-stability failures.
- Scratch CNN outperforms frozen and partial ResNet on OOD success@1cm overall in this setup.
- Pretrained encoders do not automatically solve simulated visual imitation learning.

Claims to be careful with:

- More demonstrations always help.
- Partial fine-tuning is better than freezing.
- Frozen ResNet is worse in every sense.
- Color or camera generalization is solved.

The safer framing is: visual axes differ sharply in difficulty; pretrained visual encoders are not a guaranteed win in simulation; and precision metrics reveal failures hidden by loose success.

## Open Decisions

- Should the headline precision threshold be success@1cm or success@2cm?
- Should budget 200 be shown in the main figure or marked as a one-seed extension?
- Should end distance be a main metric or a secondary stability metric?
- Should obstacle-aware reach be presented as a separate task or a harder task variant?
- Do we need seeds 1 and 2 for budgets 100/200 before making the final paper claim?
