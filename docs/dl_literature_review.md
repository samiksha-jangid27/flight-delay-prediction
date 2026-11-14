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

*Our use.* The MLP learns embeddings for `carrier`, `origin`, and `dest`
(`embeddings.py`, `mlp.py`). Destination has a larger vocabulary than origin,
which has only three airports in this dataset.

*Our finding.* The embedding MLP had lower validation Average Precision than
the AML baselines. On the same validation flights, its raw probabilities had
Pearson correlation **0.907** with raw XGBoost probabilities. The residual
correlation was **0.968**, where each residual is the observed label minus
the raw predicted probability. See [the hybrid analysis](hybrid_approach.md#error-diversity-and-interpretation).

## 2. Probability calibration of neural networks
**Guo, Pleiss, Sun & Weinberger (2017), "On Calibration of Modern Neural
Networks", ICML.** Modern networks are often poorly calibrated (confidently
wrong), and simple post-hoc methods such as temperature/Platt scaling restore
calibration without harming accuracy.

*Our use.* The project deliverable requires **calibrated** probabilities. We
report Brier score and log loss alongside ranking metrics, and compare raw vs
sigmoid/isotonic-calibrated probabilities using the existing `calibration.py`.
This paper is already cited in the Phase 1 report (`paper/main.tex`).

*Our finding.* The raw MLP's validation Brier score was 0.209 and raw
XGBoost's was 0.306. Isotonic calibration reduced both scores to about 0.137.
Calibrated Logistic Regression reached 0.151.

## 3. Trees vs. deep learning on tabular data
**Grinsztajn, Oyallon & Varoquaux (2022), "Why do tree-based models still
outperform deep learning on tabular data?", arXiv:2207.08815**, and
**Shwartz-Ziv & Armon (2021/2022), "Tabular Data: Deep Learning is Not All You
Need", arXiv:2106.03253.** Across dozens of datasets, gradient-boosted trees
remain state-of-the-art on medium-sized tabular problems even after heavy tuning,
because of inductive biases better suited to irregular, feature-engineered
tabular targets. Notably, Shwartz-Ziv & Armon also found that an **ensemble of
deep models and XGBoost beat XGBoost alone** on their datasets.

*Our finding.* Calibrated XGBoost achieved higher validation Average Precision
than the calibrated MLP and the stack. The stack's Brier score was 0.000054
lower than calibrated XGBoost's on this validation window.

## 4. Representation-level ML+DL fusion (the hybrid frontier)
**He et al. (2014), "Practical Lessons from Predicting Clicks on Ads at
Facebook" (GBDT + Logistic Regression).** Gradient-boosted tree **leaf indices**
are used as categorical inputs to a downstream linear/neural model, coupling the
feature-interaction strength of trees with a learned combiner. Combined with
Guo & Berkhahn's point that learned embeddings transfer as features, this defines
the two credible representation-level hybrids (GBDT leaves → MLP, or MLP
embeddings → XGBoost).

*Our use.* The implemented stack combines the two models' output
probabilities. The [hybrid analysis](hybrid_approach.md) documents
representation-level fusion as a possible follow-up.

## 5. Problem/domain reference
**US Bureau of Transportation Statistics — Airline On-Time Performance
documentation.** Defines the on-time/delay reporting fields and the
post-departure outcome columns that must be excluded from pre-departure features;
underpins the leakage audit in `docs/dl_data_contract.md`.

## Takeaway for this project
The experiments compare learned categorical representations with engineered
tabular features on the same chronological validation rows. The residual
analysis and metrics show how closely the models' errors overlap and how
stacking affects ranking and probability quality.

## Sources
- Guo & Berkhahn 2016 — https://arxiv.org/abs/1604.06737
- Guo, Pleiss, Sun & Weinberger 2017 — https://arxiv.org/abs/1706.04599
- Grinsztajn, Oyallon & Varoquaux 2022 — https://arxiv.org/abs/2207.08815
- Shwartz-Ziv & Armon 2021/2022 — https://arxiv.org/abs/2106.03253
- He et al. 2014 (GBDT+LR, Facebook) — https://doi.org/10.1145/2648584.2648589
