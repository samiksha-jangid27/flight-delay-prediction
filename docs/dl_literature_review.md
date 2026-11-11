# Deep Learning Literature Review

This review situates the Phase 2 design (an embedding MLP for pre-departure
flight-delay prediction) and interprets our empirical findings against the
relevant literature. Each entry states the idea, how we used it, and what our
results showed.

## 1. Entity embeddings for categorical features
**Guo & Berkhahn (2016), "Entity Embeddings of Categorical Variables",
arXiv:1604.06737.** Categorical variables are mapped into a learned Euclidean
space by a neural network during ordinary supervised training. Compared with
one-hot encoding this reduces memory, speeds training, places similar categories
near each other, and generalizes better on **high-cardinality** features where
other methods overfit. The learned embeddings can also be reused as features for
other models.

*Our use.* This is the core justification for embedding `carrier`, `origin`, and
`dest` rather than one-hot encoding them (`embeddings.py`, `mlp.py`). Origin and
destination are high-cardinality airport codes — exactly the regime the paper
targets.

*Our finding.* The embeddings trained and produced valid representations, but on
this dataset they did **not** out-rank the AML baseline, and their probabilities
are 0.97-correlated with XGBoost's (see `docs/hybrid_approach.md`). The
pre-departure signal is weak enough that the latent structure the embeddings
capture largely overlaps with the engineered route/congestion features.

## 2. Probability calibration of neural networks
**Guo, Pleiss, Sun & Weinberger (2017), "On Calibration of Modern Neural
Networks", ICML.** Modern networks are often poorly calibrated (confidently
wrong), and simple post-hoc methods such as temperature/Platt scaling restore
calibration without harming accuracy.

*Our use.* The project deliverable requires **calibrated** probabilities. We
report Brier score and log loss alongside ranking metrics, and compare raw vs
sigmoid/isotonic-calibrated probabilities using the existing `calibration.py`.
This paper is already cited in the Phase 1 report (`paper/main.tex`).

*Our finding.* The raw MLP (Brier 0.209) is better calibrated than raw XGBoost
(0.306) because BCE training with no class reweighting keeps probabilities near
the base rate; post-hoc isotonic calibration brings every model to Brier ≈ 0.14.

## 3. Trees vs. deep learning on tabular data
**Grinsztajn, Oyallon & Varoquaux (2022), "Why do tree-based models still
outperform deep learning on tabular data?", arXiv:2207.08815**, and
**Shwartz-Ziv & Armon (2021/2022), "Tabular Data: Deep Learning is Not All You
Need", arXiv:2106.03253.** Across dozens of datasets, gradient-boosted trees
remain state-of-the-art on medium-sized tabular problems even after heavy tuning,
because of inductive biases better suited to irregular, feature-engineered
tabular targets. Notably, Shwartz-Ziv & Armon also found that an **ensemble of
deep models and XGBoost beat XGBoost alone** on their datasets.

*Our finding.* We reproduce the headline result — XGBoost is competitive with or
better than the MLP on ranking — but **not** their ensembling result: on flight
delays the DL+XGBoost ensemble does *not* beat XGBoost, because the base models'
residuals are 0.968-correlated. This makes our negative hybrid result a
dataset-specific, literature-consistent contribution rather than a failure.

## 4. Representation-level ML+DL fusion (the hybrid frontier)
**He et al. (2014), "Practical Lessons from Predicting Clicks on Ads at
Facebook" (GBDT + Logistic Regression).** Gradient-boosted tree **leaf indices**
are used as categorical inputs to a downstream linear/neural model, coupling the
feature-interaction strength of trees with a learned combiner. Combined with
Guo & Berkhahn's point that learned embeddings transfer as features, this defines
the two credible representation-level hybrids (GBDT leaves → MLP, or MLP
embeddings → XGBoost).

*Our use.* We implement the simpler output-level stack now and document the
representation-level fusion as the technically justified next step for the bonus
(`docs/hybrid_approach.md`). Given the 0.968 residual correlation we expect
limited upside, but these are the designs the literature supports.

## 5. Problem/domain reference
**US Bureau of Transportation Statistics — Airline On-Time Performance
documentation.** Defines the on-time/delay reporting fields and the
post-departure outcome columns that must be excluded from pre-departure features;
underpins the leakage audit in `docs/dl_data_contract.md`.

## Takeaway for this project
The architecture choices (entity embeddings, BCE + AdamW, calibration-aware
evaluation) are directly grounded in the literature. The central empirical
finding — DL matches but does not beat gradient boosting, and adds no ensemble
value here — is exactly what the tabular-DL literature predicts for a
feature-engineered problem with a weak signal, and we support it with a
quantitative error-diversity analysis rather than assertion.

## Sources
- Guo & Berkhahn 2016 — https://arxiv.org/abs/1604.06737
- Guo, Pleiss, Sun & Weinberger 2017 — https://arxiv.org/abs/1706.04599
- Grinsztajn, Oyallon & Varoquaux 2022 — https://arxiv.org/abs/2207.08815
- Shwartz-Ziv & Armon 2021/2022 — https://arxiv.org/abs/2106.03253
- He et al. 2014 (GBDT+LR, Facebook) — https://doi.org/10.1145/2648584.2648589
