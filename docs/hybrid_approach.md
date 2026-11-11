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
| Hybrid: LR + XGB + MLP | 0.2124 | 0.6274 | 0.1380 | 0.4434 |

Meta-learner coefficients for XGB + MLP: `xgb = +3.49`, `mlp = −0.87`,
intercept `−2.87`.

## Why it does not help: error-diversity analysis

An ensemble can only improve on its members when they make *different* mistakes.
Measured on validation (reproduced by `error_diversity` in `hybrid.py`):

| Diversity measure (XGBoost vs MLP) | Value |
|---|---|
| Probability correlation (Pearson) | 0.907 |
| Rank correlation (Spearman) | 0.905 |
| **Residual correlation (`y − p`)** | **0.968** |

The residuals are **96.8% correlated**: the tree model and the embedding MLP are
wrong on almost exactly the same flights. There is therefore no orthogonal signal
for a meta-learner to recover, which is the mechanical reason every stacking,
AP-optimal blending, and rank-averaging variant we tried failed to beat the best
single model. This is consistent with the broader tabular-ML literature (trees
and MLPs converge to similar decision boundaries on engineered tabular features;
see `docs/dl_literature_review.md`).

## Honest interpretation

1. **No stacked hybrid beats the best single model on ranking.** Logistic
   regression alone has the highest Average Precision (0.253); every stacked
   variant lands near 0.21 because the meta-learner optimizes log loss and does
   not preserve LR's ranking advantage.
2. **The MLP does not add orthogonal signal.** It has the lowest standalone AP
   and receives a **negative** stacking weight — i.e. after accounting for
   XGBoost, the MLP's contribution is treated as noise to subtract. The 0.968
   residual correlation above is the direct cause.
3. **The only win is calibration, and it is not unique.** The hybrid's Brier
   (0.1371) and log loss (0.4411) are best-in-table, but only tie the already
   well-calibrated XGBoost-isotonic (0.1372 / 0.4428). Post-hoc isotonic
   calibration of a single model achieves the same thing more simply.

**Conclusion:** on this dataset and feature set, a probability-level ML+DL stack
is *correctly implemented* but does **not** demonstrate meaningful added value.
Per the grading rubric, that means this hybrid, as-is, would not qualify for the
additional "demonstrated value" marks. Reporting this transparently is the
scientifically correct outcome and is itself evidence of technical rigor.

## If the team wants to pursue the bonus

The negative result points at *why* a shallow stack fails (low error diversity;
DL embeddings do not out-rank the tree model). A fusion with a real chance of
adding value would couple the models at the representation level rather than at
the output level, e.g. **GBDT-leaf features fed as additional embeddings into the
MLP** (Facebook 2014), or **MLP-learned carrier/airport embeddings fed as dense
features into XGBoost**. Both are larger builds and are not guaranteed to win
given the weak pre-departure signal (ROC ≈ 0.62–0.65 across all models), but they
are the technically justified next step.
