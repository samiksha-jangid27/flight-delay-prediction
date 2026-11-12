# Deep learning hyperparameter search and ablation

The six-trial search in [tuning_results.csv](../reports/dl/tuning_results.csv)
uses a different seed for each configuration. It selected the published
checkpoint with dropout 0.3, seed 46, and validation AP 0.206583. The search
scores combine the effects of hyperparameter choice and random initialization
and batch order.

## Matched-seed ablation

The controlled experiment runs all six configurations at each of seeds 42, 46,
and 50. Each variant changes one hyperparameter from the baseline. The table
subtracts the baseline validation AP at the same seed, then summarizes the
three paired differences. All trials use the same chronological training and
validation windows and the same early-stopping rule.

| Change from baseline | Mean AP change | Min | Max |
|---|---:|---:|---:|
| Learning rate 0.001 → 0.0005 | +0.004759 | +0.004466 | +0.005189 |
| Batch size 1024 → 512 | −0.004602 | −0.006085 | −0.003278 |
| Hidden widths [128, 64] → [64, 32] | +0.001489 | +0.000046 | +0.002479 |
| Dropout 0.1 → 0.3 | +0.002135 | +0.001270 | +0.002942 |
| Weight decay 0.0001 → 0.001 | +0.000027 | −0.000022 | +0.000092 |

The lower learning rate had the largest mean gain among the tested changes.
Dropout improved AP at all three tested seeds. The weight-decay differences
were small and changed sign. These results describe the tested settings and
seeds on the validation window. The published checkpoint remains the one
selected by the original search, and the ablation is reported separately.

The MLP uses dropout, AdamW weight decay, and early stopping on validation AP.
The unseen embedding index maps to a frozen zero vector. The per-run data are
in [controlled_ablation.csv](../reports/dl/controlled_ablation.csv).

```bash
python scripts/run_dl_ablation.py
```
