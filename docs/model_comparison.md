# Logistic Regression vs XGBoost — modeling comparison (AML Phase 1)

This document compares **modeling candidates only**. Probability calibration,
operating thresholds, decision utility, final test evaluation and rescheduling
remain outside this change (Lakshya's responsibility).

## Historical chronological validation results (superseded)

The scores below are retained only to document earlier component experiments.
They predate the direct comparison described below and are not the current
canonical model comparison or the frozen calibrated evaluation. The earlier
notebook's balanced Logistic Regression AP of `0.206450` is superseded by the
direct shared-pipeline value `0.206047`.

| Model | Average Precision (AP) | ROC-AUC |
|---|---:|---:|
| Logistic Regression, unweighted (`C=1`) | 0.203866 | 0.601489 |
| Logistic Regression, balanced (`C=1`) | 0.206450 | 0.605049 |
| XGBoost, tuned `full` features | **0.239206** | **0.629462** |

Validation population: **50,466 flights** (September–early November 2013),
positive arrival-delay prevalence **15.64%**. These component results used
their recorded experiment configurations; they should not be assumed to have
the same fitted estimators, calibration, or exact feature contract as the
canonical frozen evaluation.

Against balanced Logistic Regression, tuned XGBoost shows an **absolute AP
advantage of 0.032756** (about **15.9% relative**), and an **absolute ROC-AUC
advantage of 0.024413**. These are validation/model-selection metrics, not
independent test results; the winner is **not proof of production performance**.
The two models use model-specific preprocessing, as appropriate.

## Verified direct comparison — 22 October 2026

Executed with `python scripts/compare_models.py --n-jobs 4`.

Both classifiers used 228,299 training flights and the same 50,466
chronological validation flights (delay prevalence: 15.6402%).

| Model | Average Precision | ROC-AUC |
|---|---:|---:|
| Logistic Regression, unweighted | 0.203867 | 0.601490 |
| Logistic Regression, balanced | 0.206047 | 0.604281 |
| Tuned XGBoost | **0.239206** | **0.629462** |

XGBoost achieved approximately 16.1% higher Average Precision
than balanced Logistic Regression.

**Discrepancy resolved.** The earlier Logistic Regression notebook reported a
balanced-model AP of `0.206450`, whereas the direct comparison here reports
`0.206047`. The difference is not a modelling error: the notebook predates the
shared-pipeline refactor, in which Logistic Regression previously built the data
twice and read an unused raw CSV, slightly perturbing the fitted pipeline. The
**direct-comparison value is `0.206047`**, produced by `scripts/compare_models.py`,
which builds the dataset once and fits all candidates on identical rows. This is
an uncalibrated validation comparison, not the canonical frozen model evaluation;
see [`canonical_evaluation.json`](canonical_evaluation.json) for the latter.

These are model-comparison results on validation data, not final
held-out test results.

---

## Reproduce the direct comparison on the current main branch

```bash
python -m pytest -q
python scripts/compare_models.py --n-jobs 4
```

The command builds the dataset **once**, fits three frozen candidate
configurations on the *same* chronological training rows, and scores each on
identical validation flights. It writes a single git-ignored CSV:

```
reports/results/model_comparison_validation.csv
```

This comparison does **not** use the held-out test set, recalibrate probabilities,
select thresholds, or change historical-feature construction. The historical
features have the team's known prior-day outcome-availability qualification in
`docs/data_contract.md`; comparing models does not resolve that limitation.

## Integration fixes

- A shared `features/model_features.py` owns `FEATURE_SETS`, categoricals, and the
  approved-feature selector. Logistic Regression can now be imported without
  importing the XGBoost implementation. The `select_xgboost_features` import is
  retained as a compatibility alias for existing notebooks/tests.
- Logistic Regression now invokes the shared data pipeline once rather than
  processing it twice and reading an unused raw CSV.
- Logistic Regression saves its trained sklearn Pipeline under a
  configuration-specific filename in `models/artifacts/`. It still exports
  `logistic_validation_predictions.csv` with its original column schema.
- No existing XGBoost hyperparameters or feature-selection semantics changed.

**Note:** the comparison script reruns the chosen XGBoost parameters rather than
using a potentially overwritten `xgboost_pipeline.joblib`. It does not perform
new hyperparameter search. The calibrated evaluation remains Lakshya's work.
