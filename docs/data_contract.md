# Flight Delay Prediction — Data Contract

## 1. Prediction Objective

Predict whether a scheduled flight will experience a substantial
arrival delay using information available before the flight departs.

## 2. Prediction Time

The model operates at the scheduled departure decision point.

Only information known before departure may be used as a model feature.

## 3. Target

The target is based on `arr_delay`.

For the initial AML experiments, a flight is labelled as substantially
delayed when:

`arr_delay >= 15 minutes`

This threshold is configurable and should be confirmed against the
final project evaluation requirements.

## 4. Modeling Population

Rows with missing `arr_delay` are excluded from the arrival-delay
classification dataset because the arrival-delay outcome is undefined
for those observations.

These rows remain part of the raw dataset and should be documented
rather than silently deleted.

## 5. Allowed Candidate Features

- month
- day
- sched_dep_time
- sched_arr_time
- carrier
- flight
- tailnum
- origin
- dest
- distance
- hour
- minute
- time_hour

Some high-cardinality features may be excluded from the first baseline
and evaluated separately.

## 6. Excluded Features

### Post-departure / leakage features

- dep_time
- dep_delay
- arr_time
- arr_delay
- air_time

### Identifier

- id

### Redundant field

- name

### Constant field

- year

The `year` column is still required internally to reconstruct each flight's
calendar date for chronological splitting and historical features; it is not
passed to the model as a standalone feature.

## 7. Historical Feature Rule and Evaluation Protocol

The project uses a **rolling daily forecast protocol** for historical
features:

- For a flight on calendar date **D**, carrier, origin, destination and route
  aggregates use labelled flights on calendar dates strictly earlier than D.
- Every flight on the same date receives history computed before that date;
  same-day outcomes are never used, even if another flight on that date has
  already arrived.
- During validation and test, actual outcomes from earlier held-out dates may
  update the historical features of later dates. This simulates daily operation,
  where yesterday's outcomes are available before predicting today's flights.
- These prior-date outcomes are used only to construct later-date history; they
  must not be used to fit the model or preprocessing on validation/test data.

This is a rolling evaluation, not a fixed holdout with historical statistics
frozen at the training cutoff. If the final evaluation requires fixed holdout
history, the feature-generation protocol must instead freeze historical
aggregates at the training cutoff and must be documented as a separate setup.

Future-date outcomes must never contribute to current or earlier historical
features.

## 8. Data Split

The dataset will be split chronologically by flight date:

- 70% earliest dates → training
- 15% subsequent dates → validation
- 15% latest dates → test

Random train/test splitting will not be used for the main experiment.

## 9. Evaluation

The main evaluation will include:

- PR-AUC
- ROC-AUC
- Precision
- Recall
- Calibration

The operational probability threshold will be selected separately from
the target-definition threshold.
