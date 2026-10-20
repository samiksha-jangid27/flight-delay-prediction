# XGBoost implementation — Project 21 (Phase 1)

**Owner:** Shubhaang Kataruka (XGBoost & Gradient Boosting Lead)

## Scope

An arrival delay of **15 minutes or more** is the positive class. All predictors must be available before scheduled departure. The raw target `arr_delay`, actual departure and arrival times, actual departure delay, and actual air time are forbidden model inputs.

This implementation is independent of Ankit's Logistic Regression work and Lakshya's probability calibration, but it uses exactly the team's upstream data-processing functions. It does not change the upstream pipeline. To compare models fairly, both models must use the same chronological train/validation/test partitions and same binary outcome definition.

## Three compared feature sets

| Set | Features |
|---|---|
| `schedule` | Scheduled time, day/weekend/month/season, carrier, origin, destination, distance, cyclic time features |
| `route` | `schedule` + route, distance band, long-haul, logarithmic distance |
| `full` | `route` + previous-date carrier, origin, destination, route delay rates and flight counts |

The model applies an **explicit allowlist** and intentionally excludes `flight`, `tailnum`, `time_hour`, `id` and all actual/outcome fields. Numeric missing values use a `-1` sentinel; categorical values are imputed and one-hot encoded. The fitted preprocessing and model are in a single scikit-learn `Pipeline`. XGBoost uses `tree_method='hist'` for CPU training.

## Experimental protocol

1. Load the original CSV and the upstream leakage-aware pipeline, then keep the chronological training and validation partitions only.
2. Train a baseline XGBoost model for each feature set using the *training* partition.
3. On training data alone, run `RandomizedSearchCV` on the `full` feature set. It uses 3 expanding chronological folds grouped by whole flight dates, separated by a one-date embargo. Candidates cover number of trees, depth, learning rate, minimum child weight, row/feature subsampling, L2 regularization and positive-class weighting.
4. Compare on chronological validation data using sklearn **Average Precision (AP)** and **ROC-AUC**; save the highest-AP validation candidate. AP is *not* identical to trapezoidal PR-AUC.
5. Save the fitted preprocessing + XGBoost pipeline, feature importance, CV leaderboard, all validation model comparisons and **row-aligned validation probabilities** for Lakshya.
6. **Never consult the final test set** to tune parameters, select feature sets, select thresholds or evaluate a calibrator. After the team freezes all choices, perform one final evaluation of the selected and calibrated model on the test set.

### Important leakage qualification

The upstream historical features use earlier calendar-date outcomes, including earlier validation dates in its *rolling daily* protocol. That protocol is documented in `docs/data_contract.md`, and is compatible with rolling online updates **if historical labels were actually available by the forecast time**. A late-arriving flight from yesterday can violate that assumption. This module does not establish arrival-observation availability; the team should flag this limitation and consider a larger delay/embargo or measured availability timestamps for production realism.

### Data-population qualification

Flights with unavailable `arr_delay` are excluded from this binary *arrival-delay* task. Cancellation/diversion prediction is not included. The starter dataset covers **2013 departures from three NYC origin airports**, not general aviation, modern flights or all US origins.

## Reproducible commands

Use Python 3.11 or 3.12 and run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
pytest -q tests/test_xgboost.py

# Quick integration smoke test (no search)
python -m flight_delay.models.xgboost_model --skip-tuning --n-jobs 4

# Main experiment: 8 sampled configurations, 3 date-aware CV folds
python -m flight_delay.models.xgboost_model --n-iter 8 --cv-folds 3 --n-jobs 4
```

Or open `notebooks/04_xgboost.ipynb` and execute cells in order.

## Output contract for evaluation lead

Generated files (intentionally ignored in the current `.gitignore`):

```text
models/artifacts/xgboost_pipeline.joblib
reports/results/xgboost_validation_predictions.csv
reports/results/xgboost_experiment_results.csv
reports/results/xgboost_feature_importance.csv
reports/results/xgboost_tuning_cv_results.csv
reports/results/xgboost_manifest.json
```

The validation predictions CSV has columns:

```text
flight_id,flight_date,y_true,y_probability,model_name,experiment
```

`flight_id` is the original dataset `id`, not a predictor. `y_probability` is **uncalibrated** P(arrival delay ≥15 minutes). Its probabilities must be passed to Lakshya; calibration/threshold fitting requires a well-defined validation split or additional out-of-fold predictions. Once both models are complete, Lakshya can compare their precision-recall curves, ROC curves, threshold behavior and calibration on like-for-like dates. Use `joblib.load` only for trusted, locally produced model files.

## Testing and results

`tests/test_xgboost.py` uses generated synthetic records to verify feature isolation, unseen categories, numerical missing values, no same-date mixing in CV, date chronology, hyperparameter search, deterministic artifact reload and prediction export. **Synthetic tests do not establish actual flight-delay predictive performance.**

Report actual AP, ROC-AUC, selected parameters and sample counts **only after the code is run against** `data/raw/flights.csv`. Do not report notebook EDA statistics as model scores.
