# Phase 2 — Deep Learning Evaluation and AML Comparison

**Project 21 | Flight Delay Prediction and Rescheduling | Topic 5**

## Objective and protocol

Evaluate the Phase 2 embedding MLP and the XGBoost+MLP stack against the frozen
Phase 1 Logistic Regression and XGBoost baselines. The task is binary arrival
delay classification (`arr_delay >= 15 minutes`) at the scheduled-departure
decision time, under the documented prior-date historical-input protocol.

**Fairness rules:** all candidate predictions must match the *same* 50,466
validation flight IDs, dates and binary labels. Model fitting must use earlier
training dates only. MLP sigmoid/isotonic calibration fits only the separate
`train_calibration` slice (46,894 flights); calibration method and operational
thresholds are selected on validation and are **not** independent final-test
results. Existing frozen Phase 1 artifacts must remain unchanged.

Sources: `docs/data_contract.md`, `docs/historical_features.md`,
`docs/canonical_evaluation.json`, `docs/dl_data_contract.md`,
`reports/evaluation_v1_noCongestion/`, and `reports/dl/`.

## Independent validation audit (11 November 2026)

An independent calculation from the six committed prediction CSVs verified
50,466 identical flight IDs, dates and true labels in every file, with 15.6402%
delayed flights. The following ranking and probability scores match the
committed experiment results to displayed precision:

| Candidate | Average Precision (higher ↑) | ROC-AUC (higher ↑) | Brier (lower ↓) | Log loss (lower ↓) |
|---|---:|---:|---:|---:|
| Logistic Regression, raw | **0.253235** | **0.646651** | 0.521863 | 1.395895 |
| Logistic Regression, isotonic | 0.223285 | 0.635976 | 0.150899 | 0.482197 |
| XGBoost, raw | 0.211211 | 0.627764 | 0.305923 | 0.860010 |
| XGBoost, isotonic | 0.212871 | 0.627927 | 0.137199 | 0.442804 |
| Embedding MLP, raw | 0.206583 | 0.620572 | 0.208524 | 0.597556 |
| XGBoost + MLP hybrid | 0.209296 | 0.625516 | **0.137145** | **0.441138** |

The report exports **scikit-learn Average Precision** and **trapezoidal PR AUC**
as separate metrics. All values in this section describe validation.

| Selected model variant | AP | Trapezoidal PR AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| Logistic Regression, isotonic | 0.223285 | 0.264062 | 0.635976 | 0.150899 |
| XGBoost, isotonic | 0.212871 | 0.211404 | 0.627927 | 0.137199 |
| Embedding MLP, isotonic | 0.208158 | 0.203870 | 0.620736 | 0.137476 |
| XGBoost + MLP hybrid | 0.209296 | 0.209212 | 0.625516 | 0.137145 |

The saved hybrid outputs were independently replayed using the reported logistic
meta-learner weights (XGBoost `+3.490936`, MLP `-0.869675`, intercept
`-2.869349`), matching all 50,466 probabilities within `1.67e-16` maximum
absolute error. Its calibration and validation inputs are perfectly aligned.

The hybrid has only a tiny Brier improvement over isotonic XGBoost (about
0.000054). It does not outperform Logistic Regression on Average Precision or
ROC-AUC. The MLP has not demonstrated superior ranking under this feature set.
The high residual-error correlation previously documented in
`docs/hybrid_approach.md` makes this negative result plausible; it does not prove
causation or rule out all alternative DL architectures.

## Operational recall at minimum precision

For each model, select the **highest-recall nonempty threshold** meeting a
prespecified minimum precision on the validation set. Report an unavailable
threshold as **not attainable**, not zero recall and not a fabricated cut-off.

At a minimum precision of **25%**:

| Candidate | Threshold attainable? | Recall at that precision | Predicted risky flights |
|---|---|---:|---:|
| Logistic Regression, raw | Yes | 43.43% | 13,712 |
| Logistic Regression, isotonic | Yes | 5.56% | 1,152 |
| XGBoost, raw / isotonic | No | N/A | N/A |
| MLP, raw | No | N/A | N/A |
| MLP, isotonic | No | N/A | N/A |
| Hybrid | No | N/A | N/A |

Raw balanced Logistic Regression is **badly miscalibrated** despite the best AP;
its 25%-precision threshold is a **ranking-derived validation rule**, not a
trusted operational probability threshold. The script also evaluates 20% and
30% minimum precision to show sensitivity rather than hide unattainable rules.
Any cut-off chosen on validation requires an independent, later-date test before
an operational claim.

## MLP calibration extension

The original MLP probabilities are not calibrated: validation Brier is
0.208524 and mean predicted delay risk is about 40.29%, versus a true 15.64%
delay prevalence. The companion script fits **sigmoid and isotonic** candidates
on the late-training `train_calibration` probabilities and then evaluates them
on validation. It selects by validation Brier with log-loss tie-breaking, and
reports metrics for all raw and calibrated variants. This is separate from the MLP's model
training and does not access test outcomes.

The constant predictor using *train-calibration* prevalence is included as a
proper-scoring baseline. A naive comparison against validation prevalence
would be descriptive but could not be deployed without knowing future labels.

## Reproduce the evaluation

From the repository root (Python 3.11+):

```bash
python -m pip install -e '.[dl]'
python -m pytest -q tests/test_dl_comparison.py
python scripts/verify_dl_checkpoint.py
python scripts/compare_aml_dl.py --bootstrap 300
```

The evaluation script writes new results to `reports/dl/` (override with
`--output`): `summary.json`, `metrics.csv`, per-model reliability CSVs, and
all figures under `reports/dl/figures/` — ROC/PR curves, calibration
(reliability) plots, AP and Brier comparisons, and illustrative confusion
matrices. The metrics CSV includes precision, recall, F1, and confusion counts
at each model's maximum-F1 validation threshold. The 25% precision constraint
is a separate operating point and can be unattainable. The optional paired
calendar-day bootstrap is descriptive; it assumes exchangeable dates and
should not be presented as a serially robust confidence interval.

The checkpoint verifier must confirm that the committed MLP weights,
preprocessing state and dataset reproduce the saved validation probabilities
within numerical tolerance. This is **checkpoint replay**, not a full
six-trial training reproduction.

The [matched-seed ablation](dl_ablation.md) reruns all six settings on shared
seeds. The [schedule-and-route-only control](../reports/dl/schedule_only_metrics.csv)
refits all four candidates with the same chronological windows and with the
eight outcome-history inputs excluded. Use it when discussing features whose
availability follows from the schedule and route.

| Schedule-and-route-only variant | Validation AP | Brier |
|---|---:|---:|
| Logistic Regression, isotonic | 0.221725 | 0.150606 |
| XGBoost, isotonic | 0.211634 | 0.137048 |
| Embedding MLP, isotonic | 0.207461 | 0.138627 |
| XGBoost + MLP hybrid | 0.208007 | 0.136717 |

The control also evaluates the offline what-if with action cost 0.25 times
the delay cost and recovery 0.8. All four policies cost more than inaction
under those assumptions. Report AP as the primary ranking score. In the
schedule-and-route-only results, isotonic calibration creates tied score
groups and trapezoidal PR area can be sensitive to interpolation to the
precision-recall endpoint.

```bash
python scripts/run_dl_ablation.py
python -m scripts.compare_schedule_only
```

## Limitations to disclose in submission

1. The prior-day historical protocol excludes same-day and future outcomes but
   does not prove exact real-time availability of every previous-day label.
2. The chosen 2013 NYC-origin dataset limits geographic and temporal
   generalizability. This is not an evaluation on current flight traffic.
3. The MLP and hybrid do not demonstrate a ranking gain over the AML baseline.
4. Calibration and operating-threshold selection on validation are not
   independent held-out test evidence.
5. No real airline rescheduling actions or savings are verified. The existing
   offline intervention study depends on explicit cost and recovery assumptions.

## Submission conclusion

The Phase 2 model architectures, training code, and frozen validation outputs
are present. The independent validation audit is internally consistent. The
embedding MLP and prediction-level stack are scientifically useful negative
comparisons, but neither should be described as an accuracy improvement.
The repository includes the checkpoint replay command and the generated
calibration and threshold figures.
