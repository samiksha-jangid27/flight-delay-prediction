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

- year
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

## 7. Historical Feature Rule

Historical features must use only observations that occurred before the
flight being predicted.

Future observations must never contribute to a historical aggregate.

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