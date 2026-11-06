# DL dataset contract — Lakshya

## Scope and API

`flight_delay.deep_learning.dataset.build_dl_dataset(path)` adapts the existing
`build_data_pipeline()` without changing it, the label definition, or split rules.
It returns NumPy arrays, not a trained neural network or a framework DataLoader.
No model training, calibration fitting, threshold selection, or test scoring is
performed. No dataset/model/evaluation files are written.

```python
from flight_delay.deep_learning.dataset import build_dl_dataset

data = build_dl_dataset("data/raw/flights.csv")
train = data.partitions["train"]
x_numeric = train.numeric          # float32, shape (N, 26)
x_categorical = train.categorical  # int64, shape (N, 5)
y = train.labels                  # int64 binary target, shape (N,)
# Embedding column order: data.categorical_features
# Embedding table sizes: data.embedding_sizes
# Comparison rows, in exactly the same positional order:
identity = train.metadata          # flight_id, flight_date, y_true, partition
```

The returned `medians`, `means`, `scales` and `vocabularies` describe the fitted
preprocessing. Reuse these partition arrays together; do not independently encode
held-out data. Arrays and metadata are mutable, so callers must keep their row
order synchronized. The adapter retains upstream within-partition row order;
it does not shuffle, balance, oversample, or drop additional labeled rows.
Flight IDs remain strings, including leading zeros. Dates are calendar dates,
not absolute prediction/outcome availability timestamps.

## Population and chronological boundaries

The original `arr_delay >= 15` label is unchanged. Missing `arr_delay` rows remain
excluded upstream. The local dataset contains **327,346 labeled rows**. Whole dates
are assigned by the existing 70%/15%/15% unique-date split. Original training is
subdivided at `sorted(training_dates)[int(number_of_training_dates * 0.8)]`, exactly
as in Phase 1 evaluation; the earlier dates train preprocessing and the model,
and the later dates are reserved for calibration. Validation and test are intact.

Verified locally against dataset SHA-256
`8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa`:

| Adapter partition | Rows | First date | Last date |
| --- | ---: | --- | --- |
| train | 181,405 | 2013-01-01 | 2013-07-23 |
| train_calibration | 46,894 | 2013-07-24 | 2013-09-12 |
| validation | 50,466 | 2013-09-13 | 2013-11-06 |
| test | 48,581 | 2013-11-07 | 2013-12-31 |

The first two partitions together preserve the original 228,299 training rows.
All IDs must be globally unique and nonmissing, labels binary and present, and
all four windows nonempty and strictly date-ordered. Invalid inputs fail rather
than silently changing observations. Tiny inputs need sufficient dates for all
four windows. Row counts above are a verified property of the original file,
not a hardcoded constraint preventing synthetic testing.

## Feature provenance and representation

The exact starting schema is `FEATURE_SETS["full"]`, the canonical **31-feature**
allowlist in `features/model_features.py`. The adapter exposes ordered column
names in `numeric_features` and `categorical_features`.

| Source | Features | Provenance |
| --- | --- | --- |
| Schedule/calendar | month, day, sched_dep_time, sched_arr_time, hour, minute, carrier, origin, dest, distance | Published schedule and route information |
| Calendar transforms | day_of_week, is_weekend, season, hour_sin, hour_cos, month_sin, month_cos, day_of_week_sin, day_of_week_cos | Deterministic `features/temporal.py` transforms |
| Route transforms | route, distance_band, is_long_haul, distance_log | Deterministic `features/route.py` transforms of origin/destination/distance |
| Historical | carrier/origin/destination/route historical_delay_rate and historical_flight_count (8 total) | Strictly earlier calendar-date labeled flights; rolling daily protocol |

`carrier`, `origin`, `dest`, `route`, and `distance_band` stay separate categorical
columns suitable for embeddings; none is ordinalized as a continuous number.
Each vocabulary is sorted deterministically using **early training only**.
Index **0** represents missing values, **1** unseen values, and **2 onward** known
training categories. Missing/unseen categories never discard rows.

The remaining 26 features use early-training medians for missing values (zero
when an entire training column is missing), then early-training mean and
population standard deviation after imputation. Constant columns use scale 1.
Outputs must be finite float32; infinite input values fail. No calibration,
validation or test data fit vocabulary, imputation or normalization. Historical
rates with no prior observations are missing upstream; this explicit imputation
handles them without borrowing later outcomes. No missingness indicator or extra
feature is added. Scheduled HHMM fields retain the canonical numeric treatment;
this adapter does not redesign their representation.

## Leakage audit and availability limitation

The positive allowlist excludes `dep_time`, `dep_delay`, `arr_time`, `arr_delay`,
`air_time`, raw labels, flight IDs, tail numbers, and all other unapproved columns.
IDs/dates/labels exist only in comparison metadata. The newer congestion ablation
features are generated upstream but do not enter these 31 inputs.

Historical feature construction aggregates labels by entity/date and subtracts
the entire current date from cumulative totals. Therefore neither a flight's own
label, other same-day labels, nor future-date outcomes enter its predictors.
Tests perturb current/future outcomes and post-departure columns and check that
current/earlier inputs are unchanged, including across partition boundaries.
Existing independent historical tests check all four entities against strict-past
aggregations and verify rolling updates.

**Date-order safety does not prove real-time availability.** The Phase 1 rolling
protocol lets earlier calibration/validation/test dates update histories on later
dates. It assumes those earlier outcomes are already known. Overnight arrivals,
long delays and publication latency can violate this assumption. The CSV lacks
verified absolute outcome-publication timestamps and timezone/date-rollover
information. Consequently these features are conditionally available, and this
adapter must not be described as proving deployment-time leakage freedom.
An as-of availability join would need additional verified event timestamps; a
fixed training-cutoff history would be a different protocol. Neither is silently
substituted here. The labeled-only population also excludes undefined arrival
outcomes and cannot establish performance for all scheduled flights.

## Verification

Run `python -m pytest -q tests/test_dl_dataset.py tests/test_historical_availability.py tests/test_split.py`.
Tests cover reproducibility, schema/dtypes, ID-label-date alignment, original
population preservation, whole-date boundaries, missing and unseen values,
train-only fitting, invalid inputs and outcome perturbations. The real-population
test runs when the local CSV exists and explicitly skips otherwise. No DL model
performance claim follows from dataset verification.
