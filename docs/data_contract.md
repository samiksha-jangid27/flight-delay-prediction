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

## 5. Candidate Features

The raw schedule provides candidate fields such as month, day, scheduled
departure/arrival times, carrier, origin, destination, distance, hour and
minute. `flight`, `tailnum` and `time_hour` are not admitted by the model
allowlist. Engineered calendar and route fields are described in the data
dictionary.

The canonical frozen evaluation uses the 31-feature `full` allowlist. The
two congestion features below are an additional 33-feature variant used only
for a validation ablation; no test score is reported for that variant.

### Schedule-derived congestion proxies

- `origin_hourly_departures` counts flights scheduled to depart the same origin
  in the same departure-date and scheduled-departure-hour bucket.
- `dest_hourly_arrivals` counts flights in this NYC-origin dataset scheduled
  to arrive at the same destination in the same scheduled-arrival-hour bucket,
  grouped by departure date.

Both features are computed over the complete raw timetable before flights with
undefined `arr_delay` are removed. Thus a cancelled/diverted flight still
contributes its scheduled slot. The destination feature is only a partial
schedule-based load proxy, not a count of all airport arrivals. Flights
scheduled across midnight remain grouped under their departure date; the
arrival date is not reconstructed. These assumptions are a limitation, not a
claim that realized airport congestion is known.

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

The frozen evaluation reports scikit-learn Average Precision (AP), ROC-AUC,
precision, recall, F1 and calibration:

AP is recall-increment-weighted precision and is not identical to trapezoidal
area under a precision-recall curve. Use “Average Precision (AP)” when
referring to the stored metric.

The operational probability threshold will be selected separately from
the target-definition threshold.
