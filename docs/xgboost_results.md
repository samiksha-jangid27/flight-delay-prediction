# XGBoost experiments — Phase 1 (Advanced Machine Learning)

**Project:** Flight Delay Prediction (Project 21, Team T022)
**Component owner:** Shubhaang Kataruka — XGBoost and Hyperparameter Tuning
**Evaluation date:** 20 October 2026
**Status:** Training and chronological validation complete; final held-out test evaluation and calibration pending.

## Objective and data

Predict whether a flight arrives **15 minutes or more** after its scheduled arrival time (`is_delayed = 1`). The model uses scheduling-time information and the project's prior-date historical features; actual departure/arrival outcomes are excluded from predictors.

The dataset contains 2013 flights departing from New York City's JFK, LGA, and EWR airports. The team pipeline retains 327,346 flights with defined arrival-delay labels and splits by **calendar date**:

| Partition | Flights | Intended use |
|---|---:|---|
| Training | 228,299 | Fit models and perform date-grouped hyperparameter cross-validation |
| Validation | 50,466 | Compare candidate feature sets and select the reported model |
| Test | 48,581 | Held out; **not evaluated** in this XGBoost work |

**Validation delay prevalence:** 15.64%, versus 25.91% on training. The no-skill reference for sklearn Average Precision (AP) is approximately 0.1564 on validation.

## Approach

Three XGBoost feature sets were compared:

1. **Schedule:** scheduled timing, calendar, carrier, airport and distance information.
2. **Route:** schedule features plus route identifier and engineered route/distance variables.
3. **Full:** route features plus previous-date delay rates and observation counts for carrier, origin, destination and route.

The model uses a fixed **feature allowlist**, numerical missing-value imputation, categorical imputation, one-hot encoding, and an XGBoost classifier in a single scikit-learn `Pipeline`. Preprocessors are fit only on the relevant training partition. The historical features follow the project's explicitly documented **rolling daily forecast** protocol.

Hyperparameter tuning used `RandomizedSearchCV` with **8 sampled configurations**, **3 expanding chronological folds grouped by flight date**, a **one-date embargo**, and sklearn **Average Precision** for selection. This is **not** trapezoidal PR-curve AUC. The final selected candidate was refitted on the full training partition and compared with the baselines on validation. The held-out test set was not used for model selection.

## Validation results

| Experiment | Feature set | Tuned | Average Precision (AP) | ROC-AUC |
|---|---|---|---:|---:|
| `baseline_schedule` | Schedule | No | 0.220223 | 0.606259 |
| `baseline_route` | Route | No | 0.220314 | 0.602096 |
| `baseline_full` | Full | No | 0.223590 | 0.621529 |
| **`tuned_full`** | **Full** | **Yes** | **0.239206** | **0.629462** |

The selected model improved AP by **0.015616 absolute** (approximately **6.98% relative**) and ROC-AUC by **0.007933 absolute** over the untuned full-feature baseline. Its AP is approximately **1.53×** the validation delay prevalence (0.239206 / ~0.1564). ROC-AUC of 0.6295 indicates **modest ranking/discrimination**, not production-grade prediction accuracy.

### Training-only cross-validation

| Fold | Delay prevalence | Average Precision | AP / prevalence |
|---|---:|---:|---:|
| Fold 1 | 25.91% | 0.3595 | 1.39× |
| Fold 2 | 29.62% | 0.4604 | 1.55× |
| Fold 3 | 24.88% | 0.3739 | 1.50× |
| **Mean** | — | **0.397892** | — |

The mean CV AP (0.397892) is substantially higher than validation AP (0.239206), but validation's positive-class prevalence is also much lower. The AP-to-prevalence ratios are broadly comparable, which helps explain the raw-score difference. This **does not rule out** seasonal distribution shift or overfitting.

### Monthly validation performance

| Validation month | Flights | Delay prevalence | AP |
|---|---:|---:|---:|
| September 2013 | 16,438 | 11.97% | 0.2165 |
| October 2013 | 28,618 | 17.56% | 0.2650 |
| November 2013 (partial) | 5,410 | 16.62% | 0.2623 |

These are descriptive **post-selection validation diagnostics**, not additional independent test evaluations.

## Selected hyperparameters

| Parameter | Value |
|---|---:|
| `n_estimators` | 250 |
| `max_depth` | 6 |
| `learning_rate` | 0.03 |
| `min_child_weight` | 10 |
| `subsample` | 0.90 |
| `colsample_bytree` | 0.75 |
| `reg_lambda` | 10.0 |
| `scale_pos_weight` | 2.859 |

Eight randomly sampled configurations do **not** establish global hyperparameter optimality. Class weighting improves some ranking scenarios but may affect probability calibration.

## Feature importance

The saved `tuned_full` model's highest gain-importance encoded features were:

| Rank | Encoded feature | Gain importance |
|---|---|---:|
| 1 | `numeric__hour` | 0.115846 |
| 2 | `numeric__sched_dep_time` | 0.055305 |
| 3 | `numeric__month_cos` | 0.027566 |
| 4 | `numeric__hour_sin` | 0.025053 |
| 5 | `numeric__season` | 0.021976 |
| 6 | `categorical__carrier_EV` | 0.021409 |
| 7 | `numeric__is_weekend` | 0.021342 |
| 8 | `numeric__day_of_week_sin` | 0.020424 |
| 9 | `numeric__month_sin` | 0.019816 |
| 10 | `numeric__month` | 0.019153 |

Importance is **model-relative** and does not establish causal effects. Correlated calendar and time features can share or dilute their importance.

## Verification and reproducibility

The initial repository suite passed **33 tests**, and after adding the XGBoost component the combined suite passed **42 tests**, including **9 XGBoost-specific tests**. *(These are historical snapshots from the XGBoost-integration phase; the full project suite now stands at **223 tests** — see the main README.)* The final selected fitted pipeline was saved, reloaded, and verified to reproduce every saved validation probability (within floating-point tolerance), matching all **50,466 flight IDs** and labels. This verification was performed on the developer's Mac with Python 3.12.15 and XGBoost 3.4.1.

Commands (run from repository root in the prepared environment):

```bash
python -m pytest -q
python -m flight_delay.models.xgboost_model --skip-tuning --n-jobs 4
python -m flight_delay.models.xgboost_model --n-iter 8 --cv-folds 3 --n-jobs 4
```

The final full command produces the selected model and the following **git-ignored** files:

```text
models/artifacts/xgboost_pipeline.joblib
reports/results/xgboost_validation_predictions.csv
reports/results/xgboost_experiment_results.csv
reports/results/xgboost_feature_importance.csv
reports/results/xgboost_tuning_cv_results.csv
reports/results/xgboost_manifest.json
```

The validation CSV exposes `flight_id,flight_date,y_true,y_probability,model_name,experiment` for subsequent row-aligned comparisons. `flight_id` is a join key only, never a predictor. Share the probability data and saved model with the evaluation lead by an approved method; **generated artifacts are not included in the Git commit**.

## Limitations and evaluation handoff

- **No held-out test score yet.** Validation was used to compare/select the feature sets and tuned candidate, so its AP and ROC-AUC are **selection-set results**, not an independent estimate of final generalization.
- **No probability calibration or operational threshold.** The evaluation lead should use an appropriate calibration-only split or independent/out-of-fold predictions, avoiding optimistic calibration assessment on the same data used for selection.
- **Historical-label timing is an upstream caveat.** Features are based on earlier calendar dates, but it has not been proven that *all* previous-date arrival outcomes would have been observable before each prediction timestamp. This should be tested against outcome availability if the model is deployed operationally.
- **Limited geographic/temporal coverage.** The 2013 NYC-origin dataset does not establish performance across modern routes or all airports; cancellations and diversions with undefined arrival delays are outside this classifier.
- **Comparative evaluation pending.** Once the Logistic Regression baseline is available, compare both models using the same date partitions, outcomes and metric definitions. Keep the held-out test set untouched until all choices are frozen.

**Phase 1 XGBoost status:** Implementation, baseline experiments, tuning, validation predictions, importance, and artifact-reload verification are complete. Model comparison, independent final evaluation, calibration, and any rescheduling-impact analysis are team-level next steps.
