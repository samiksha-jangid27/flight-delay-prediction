# Hybrid Approach: Stacked XGBoost + MLP Ensemble

**Status: implemented and evaluated. Honest verdict — does not yet demonstrate
a ranking improvement over the AML baseline on this dataset.** This document
reports the method, the leakage controls, the measured result, and the reason,
so the finding can be used in the report either as a negative/ablation result or
as the starting point for a stronger fusion.

## Rationale (the intended "hybrid logic")

The project brief frames the hybrid as *"operational tabular features to learned
temporal and route patterns."* XGBoost splits on engineered operational features
(schedule, congestion, calendar, historical delay rates) and captures sharp
threshold interactions; the embedding MLP learns dense latent vectors for
carrier/origin/dest and smooth combinations of the standardized numeric block.
If the two families made sufficiently independent errors, a small logistic
meta-learner stacked on their probabilities would recover signal neither
expresses alone.

## Method

- **Base models (unchanged):** the frozen Phase 1 XGBoost (`reports/evaluation/frozen.joblib`)
  and the Phase 2 embedding MLP (`reports/dl/*_predictions.csv`).
- **Meta-learner:** `LogisticRegression` on two features — the XGBoost and MLP
  predicted probabilities.
- **Fit set:** the `train_calibration` partition (last 20% of training dates).
- **Evaluation set:** `validation`. The held-out `test` partition is never loaded.

Code: `src/flight_delay/deep_learning/hybrid.py`, driver `scripts/build_hybrid.py`,
tests `tests/test_hybrid.py`.

## Leakage discipline

Both base models are trained **only on the early-training partition**, so their
probabilities on the later `train_calibration` rows are genuinely out-of-sample.
The meta-learner is fitted on those calibration probabilities and evaluated on
`validation`. The `test` partition is never transformed, scored, or used for any
selection — enforced by the module (it calls `build_data_pipeline` and uses only
`train`/`validation`) and asserted in the tests.

## Results (validation, same rows and protocol; all base models early-train)

| Model | Avg Precision ↑ | ROC-AUC ↑ | Brier ↓ | Log loss ↓ |
|---|---|---|---|---|
| Logistic (raw) | **0.2532** | **0.6467** | 0.5219 | 1.3959 |
| Logistic (isotonic) | 0.2233 | 0.6360 | 0.1509 | 0.4822 |
| XGBoost (isotonic) | 0.2129 | 0.6279 | **0.1372** | 0.4428 |
| XGBoost (raw) | 0.2112 | 0.6278 | 0.3059 | 0.8600 |
| MLP (raw) | 0.2066 | 0.6206 | 0.2085 | 0.5976 |
| **Hybrid: XGB + MLP** | 0.2093 | 0.6255 | **0.1371** | **0.4411** |

Meta-learner coefficients for XGB + MLP: `xgb = +3.49`, `mlp = −0.87`,
intercept `−2.87`.

## Error diversity and interpretation

The validation residuals, defined as observed delay minus raw predicted risk,
had Pearson correlation 0.968. The probability Pearson correlation was
0.907 and rank correlation was 0.905. These figures show substantial overlap
in the two models' errors. The fitted coefficient on MLP probability is
conditional on the XGBoost probability and does not establish causation.

The hybrid's validation AP was 0.2093, compared with 0.2129 for calibrated
XGBoost and 0.2233 for calibrated Logistic Regression. Its Brier score was
0.137145, compared with 0.137199 for calibrated XGBoost and 0.137476 for
the calibrated MLP. This is a small descriptive difference on one validation
window. The project's offline rescheduling analysis uses the calibrated AML
models. The hybrid remains an evaluated candidate in the Phase 2 comparison.

## Reproduce

From a fresh checkout, generate the Phase 1 checkpoint before building the
hybrid. The stack fits on calibration predictions and checks matching IDs,
dates, and labels in each window.

```bash
python -m flight_delay.evaluation.evaluate validation --output-dir reports/evaluation
python scripts/build_hybrid.py
python scripts/compare_aml_dl.py --bootstrap 300
```
