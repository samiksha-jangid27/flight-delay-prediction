# Flight Delay Prediction — Evaluation & Calibration

Executed 22 October 2026. Contribution: Lakshya, Model Evaluation & Calibration.

## Executive summary

Formal evaluation compares ranking, probability quality and delay-warning operating points on identical flight IDs. The original full-training comparison was reproduced. A separate chronological calibration experiment refitted both base estimators on early training dates without retuning. Isotonic minimized validation Brier and log loss for both models; balanced Logistic Regression led selected-probability validation AP, but paired date-bootstrap confidence intervals include zero. The frozen held-out results favor XGBoost descriptively. These results do not justify changing the validation-selected model retrospectively or claiming production readiness.

## Experimental methodology

The original chronological boundaries are unchanged: 228,299 training, 50,466 validation and 48,581 test flights. Within training only, the first 80% of distinct training dates fit the base estimator and preprocessing; the remaining 20% fit calibrators. No calibrator-fitting labels fit those base estimators. All flights on a date remain together. Exact windows:

| Model | n | first | last |
|---|---:|---:|---:|
| base_train | 181405 | 2013-01-01 | 2013-07-23 |
| train_calibration | 46894 | 2013-07-24 | 2013-09-12 |
| validation | 50466 | 2013-09-13 | 2013-11-06 |

The existing `full` feature allowlist, preprocessing constructors, seed 42 and selected XGBoost hyperparameters are reused. Models and calibrators are saved only under ignored `reports/evaluation/frozen.joblib`; teammates' artifact paths are untouched. Bases are deliberately not refitted on the complete original training period after calibration. This experiment therefore evaluates different fitted estimators from the earlier full-training comparison.

**Conditional calibration qualification:** the existing XGBoost hyperparameters were originally selected using the full original training period, including dates now reserved for calibration. The calibration predictions are out of sample with respect to base-estimator parameter fitting, but the calibration block is not independent of the historical hyperparameter search. No retuning was performed, per project scope. A fully nested calibration study would require a training-only hyperparameter selection window that excludes this block.

The shared upstream builder computes rolling historical features before splitting. Prior-date held-out labels can update features on later dates under the documented rolling daily protocol; they never fit estimators, calibrators or thresholds. This is not a fixed-history holdout. Validation stage does not score or inspect the returned test frame.

Prediction schema: `flight_id`, ISO `flight_date`, binary `y_true`, finite `[0,1]` `y_probability`, explicit `partition`. Alignment rejects duplicate/missing IDs, unequal ID sets, inconsistent labels/dates or partition tags; no inner join discards records. Decisions use `probability >= threshold`. Zero-denominator classification ratios are zero; one-class ranking scores are null. Balanced accuracy uses the mean of the two class recalls, assigning zero to an unsupported class.

Average Precision (AP) uses recall-increment-weighted precision, not trapezoidal PR area. ROC-AUC measures ranking; prevalence is the no-skill PR reference. Brier and log loss are proper probability scores, both lower-is-better. Scikit-learn clips endpoint probabilities internally for finite log loss. Classification reports include per-class support, precision, recall and F1. Calibration-in-the-large is represented by mean prediction, observed prevalence and their difference (not a fitted calibration intercept).

## Existing full-training validation baseline

The required comparison runner succeeded before and after implementation:

| Model | AP | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.203867 | 0.601490 |
| Balanced Logistic Regression | 0.206047 | 0.604281 |
| Tuned XGBoost | 0.239206 | 0.629462 |

These reproduce the supplied direct-run validation results. Prior notebook balanced-LR values differ slightly, as already recorded in `docs/model_comparison.md`; they are not substituted here. None is a final test score.

## Calibration experiments — validation only

Sigmoid is logistic regression of labels on clipped log odds (`1e-7` clipping; `C=1e10`, approximately unpenalized). Isotonic is scikit-learn's monotone probability regression with clipping outside its fitted score range. Raw is an identity map. Calibrators fit only late-training observations. Selection minimizes validation Brier, using log loss to break ties.

| Model | average_precision | roc_auc | brier | log_loss | mean_probability | prevalence |
|---|---:|---:|---:|---:|---:|---:|
| logistic_balanced_raw | 0.253235 | 0.646651 | 0.521863 | 1.395895 | 0.781884 | 0.156402 |
| logistic_balanced_sigmoid | 0.253235 | 0.646651 | 0.155823 | 0.492023 | 0.324285 | 0.156402 |
| logistic_balanced_isotonic | 0.223285 | 0.635976 | 0.150899 | 0.482197 | 0.305147 | 0.156402 |
| xgboost_tuned_full_raw | 0.211211 | 0.627764 | 0.305923 | 0.860010 | 0.540171 | 0.156402 |
| xgboost_tuned_full_sigmoid | 0.211211 | 0.627764 | 0.138253 | 0.443644 | 0.233095 | 0.156402 |
| xgboost_tuned_full_isotonic | 0.212871 | 0.627927 | 0.137199 | 0.442804 | 0.230979 | 0.156402 |

Isotonic improves Brier and log loss versus raw/sigmoid for both candidates. Isotonic ties reduce LR AP from 0.253235 to 0.223285 and ROC-AUC from 0.646651 to 0.635976; XGBoost AP changes from 0.211211 to 0.212871. Thus calibration is not ranking-neutral. Sigmoid retains ranking in this experiment.

The validation prevalence is 0.156402. A constant predictor set to that validation prevalence has Brier approximately 0.131941 (an in-sample descriptive reference, not a deployable selection candidate). Both selected models have worse validation Brier. Lower loss versus severely miscalibrated raw scores alone does not demonstrate useful probability forecasts. Historical feature/count extrapolation and temporal changes can destabilize the early-trained models. We retain the predefined selection rule and report these weaknesses.

Reliability diagrams use ten equal-width bins over [0,1]; 0 and 1 are included, empty bins omitted, and per-bin counts are exported as CSV alongside plots. Reliability shape does not override proper scores.

## Classification and threshold selection — validation only

| Model | threshold | precision | recall | f1 | f2 | roc_auc | average_precision | brier | log_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_balanced | 0.343573 | 0.223727 | 0.604713 | 0.326616 | 0.451083 | 0.635976 | 0.223285 | 0.150899 | 0.482197 |
| xgboost_tuned_full | 0.250000 | 0.216151 | 0.593437 | 0.316883 | 0.439878 | 0.627927 | 0.212871 | 0.137199 | 0.442804 |

Threshold search covers every unique predicted probability plus a 201-point grid from 0 to 1. Equal metric values prefer the higher threshold. The provisional recommendation maximizes validation F1, balancing precision and recall without claiming a known intervention cost. F2 emphasizes recall. Strategy B maximizes recall subject to illustrative precision floors; no floor or false-positive cost has been approved operationally. Higher precision and threshold break remaining Strategy B ties. Infeasible constraints are recorded as null.


### logistic_balanced

| Model | threshold | precision | recall | f1 | f2 | fp | fn | predicted_positive_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max_f1 | 0.343573 | 0.223727 | 0.604713 | 0.326616 | 0.451083 | 16561.000000 | 3120.000000 | 0.422740 |
| max_f2 | 0.252551 | 0.174990 | 0.910047 | 0.293537 | 0.494561 | 33865.000000 | 710.000000 | 0.813379 |
| reference_0.5 | 0.500000 | 0.426136 | 0.009502 | 0.018590 | 0.011812 | 101.000000 | 7818.000000 | 0.003487 |

Illustrative precision sensitivity:

| Model | threshold | precision | recall | fp | fn |
|---|---:|---:|---:|---:|---:|
| precision >= 0.2 | 0.318681 | 0.201793 | 0.727100 | 22701.000000 | 2154.000000 |
| precision >= 0.25 | 0.358918 | 0.381076 | 0.055619 | 713.000000 | 7454.000000 |
| precision >= 0.3 | 0.358918 | 0.381076 | 0.055619 | 713.000000 | 7454.000000 |
| precision >= 0.4 | 0.369565 | 0.408333 | 0.012416 | 142.000000 | 7795.000000 |

### xgboost_tuned_full

| Model | threshold | precision | recall | f1 | f2 | fp | fn | predicted_positive_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| max_f1 | 0.250000 | 0.216151 | 0.593437 | 0.316883 | 0.439878 | 16986.000000 | 3209.000000 | 0.429398 |
| max_f2 | 0.119565 | 0.180426 | 0.877993 | 0.299339 | 0.495134 | 31479.000000 | 963.000000 | 0.761087 |
| reference_0.5 | 0.500000 | 0.166667 | 0.000633 | 0.001262 | 0.000791 | 25.000000 | 7888.000000 | 0.000594 |

Illustrative precision sensitivity:

| Model | threshold | precision | recall | fp | fn |
|---|---:|---:|---:|---:|---:|
| precision >= 0.2 | 0.212913 | 0.200329 | 0.726086 | 22877.000000 | 2162.000000 |

Infeasible floors: 0.25, 0.3, 0.4.


Both F1 choices produce many false alerts (validation precision about 22%). They are research operating points, requiring operational review before use. At 0.5, calibrated recall is very low. All choices were written to `selection.json` with estimator/data SHA-256 identities before the test command. The final stage loads frozen objects and cannot call fitting or threshold search.

## Paired uncertainty — validation

XGBoost minus balanced LR, selected probabilities and each model's frozen F1 threshold. 300 paired calendar-date resamples, seed 42, percentile 95% intervals; all 300 contained both classes:

- average_precision: [-0.026507, 0.010650].

- roc_auc: [-0.025174, 0.010878].

- f1: [-0.022234, 0.005034].


Intervals include zero. Resampling whole dates preserves within-date dependence but assumes dates are exchangeable; it does not capture serial dependence across days, fitted-model uncertainty or calibration/threshold selection optimism. Validation is a development set, not independent confirmation.

## Final held-out test — frozen one-time evaluation

The final stage ran only after selection artifacts were saved. Test support is 48,581; dates and prevalence are recorded in predictions and metrics. Both models are reported as prespecified candidates, not selected again using test performance.

| Model | n | prevalence | threshold | precision | recall | f1 | roc_auc | average_precision | brier | log_loss | tn | fp | fn | tp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_balanced | 48581 | 0.268562 | 0.343573 | 0.296195 | 0.488005 | 0.368642 | 0.543888 | 0.294262 | 0.198391 | 0.586488 | 20405 | 15129 | 6680 | 6367 |
| xgboost_tuned_full | 48581 | 0.268562 | 0.250000 | 0.334651 | 0.531923 | 0.410833 | 0.598862 | 0.337091 | 0.195484 | 0.586880 | 21736 | 13798 | 6107 | 6940 |

Balanced LR was the provisional validation AP winner after calibration selection; XGBoost performed better on test AP, ROC-AUC, precision, recall and F1. LR test log loss is slightly lower. This does not retroactively change selection or thresholds. Changes in prevalence mean validation/test AP and F1 are not directly comparable measures of model improvement. Full class-wise metrics and support are in `test_metrics.json` and the notebook.

## Figures and artifacts

All generated artifacts are ignored by Git and regenerated by the CLI:

- `reports/evaluation/figures/validation/`: ROC, PR, selected-threshold confusion matrices, per-model reliability diagrams, precision/recall/F1 versus threshold, false-positive/false-negative counts, proper-score and metric comparisons.
- `reports/evaluation/figures/calibration/`: Brier/log-loss and ranking comparisons across all six raw/calibrated variants, plus default-threshold confusion matrices.
- `reports/evaluation/figures/test/`: final ROC/PR, confusion matrices, Brier, log loss, AP, ROC-AUC and F1 comparisons.
- `reports/evaluation/*_validation.csv`, `*_test.csv`, `*_thresholds.csv`: aligned predictions and operating curves.
- `reports/evaluation/{validation_metrics,calibration_metrics,test_metrics,validation_bootstrap,selection}.json`: strict machine-readable metrics and provenance.
- `reports/evaluation/frozen.joblib`: fitted early-training bases and selected maps; load only trusted local joblib files.
- `notebooks/05_model_evaluation.ipynb`: shared-module analysis, figures and limitations; no invented outputs.

PNG figures use a headless Matplotlib backend and 220 DPI. Reliability counts are saved as CSV. The CLI stores figures inside its output directory so separate experiments stay isolated.

## Verification and reproducibility

Initial baseline attempts: pytest stopped at collection with two XGBoost import errors; comparison also failed because macOS `libomp.dylib` was missing. After resolving that environment dependency, all **53 existing tests passed**. The expanded suite contains **85 tests: 85 passed, 0 failed, 0 skipped**. It includes a synthetic end-to-end validation/final-test test, rejects calibration/threshold fitting on test tags, checks chronology and exact alignment, round-trips serialized maps, checks repeat-finalization protection and verifies deterministic bootstrap. Existing model comparison reproduced the same numbers again after implementation.

```bash
cd /Users/lakshyachoudhary/Documents/flight-delay-prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# macOS only, if XGBoost cannot load OpenMP:
# brew install libomp
python -m pytest -q
python scripts/compare_models.py --n-jobs 4
python -m flight_delay.evaluation.evaluate validation --output-dir reports/evaluation-new --n-jobs 4 --bootstrap 300
# Review reports/evaluation-new/selection.json, then finalize once:
python -m flight_delay.evaluation.evaluate final-test --output-dir reports/evaluation-new
jupyter notebook notebooks/05_model_evaluation.ipynb
```

Use `reports/evaluation` to view this run in the notebook; change its `OUT` path to inspect another run. Frozen validation output directories and completed test evaluations cannot be overwritten by the CLI. To compare external prediction CSVs, load them with pandas and call `align_predictions(frames, 'validation')` then `evaluate`; this does not establish the provenance needed to fit a calibrator. The runner intentionally fits known frozen configurations rather than trusting provenance-free in-sample prediction files.

For this session, OpenMP was obtained from the official Homebrew ARM64 Sequoia bottle with verified SHA-256, extracted into ignored `.venv/libomp`, and supplied with `DYLD_LIBRARY_PATH="$PWD/.venv/libomp"` before Python commands. No system libraries were modified. Standard `brew install libomp` is the usual portable macOS setup. Package versions and dataset/artifact hashes are persisted in `selection.json`. Project dependencies are unpinned, so future installs may differ; use those recorded versions to recreate this environment.

## Limitations and recommendations

- Historical features use earlier calendar-date outcomes. Exact availability at flight prediction time is not fully verified: availability-time leakage remains possible despite strict prior-date chronology.
- Historical updates use prior held-out outcomes under the team's rolling daily protocol, not a frozen-history protocol.
- Hyperparameters were previously selected across the original training period; the calibration study is conditional on those choices rather than fully nested.
- Class imbalance, seasonal distribution shift, one year of New York flight data and a finite late-training calibration block limit generalization. Isotonic ties can harm ranking. Validation reliability may not transfer to later dates.
- Validation selects several decisions and therefore has optimism. Date bootstrap does not account for the full model-selection process.
- No operational false-positive costs, minimum precision requirement or capacity limits were provided. F1 is a research recommendation; sensitivity tables require stakeholder interpretation.
- Selected probabilities are not convincingly better than a prevalence-only validation reference. Further research should examine calibration stability using earlier training-only windows and feature availability, without repeatedly optimizing against this test set.

For this frozen study, retain the validation-selected isotonic maps and thresholds exactly as recorded; describe balanced LR as the provisional selected-AP validation leader with uncertain differences and XGBoost as the stronger descriptive held-out candidate. No production-readiness claim is supported.

## Phase 1 presentation — five-slide outline

1. **Methodology:** unchanged chronological partitions; early-training base fit and late-training calibration; explicit validation/final-test separation and availability-time caveat.
2. **Model comparison:** reproduce original full-training AP/ROC-AUC; distinguish early-trained calibration candidates; show validation ROC/PR and paired intervals.
3. **Calibration:** raw/sigmoid/isotonic Brier and log loss; reliability with counts; isotonic ranking ties and prevalence-baseline limitation.
4. **Threshold selection:** F1 versus F2 and illustrative precision constraints; chosen thresholds and false-alert/missed-delay counts; freeze before test.
5. **Conclusions:** actual held-out AP/F1 and confusion matrices; changed model ordering, temporal shift, operational costs and research next steps.
