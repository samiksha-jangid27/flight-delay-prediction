# Historical Feature Design

## Objective

Provide historical estimates of delay behavior for the carrier,
origin airport, destination airport, and route.

## Leakage Rule

Historical statistics use observations from previous calendar dates
only.

The current calendar day's flights are excluded from their own
historical statistics.

## Features

### Carrier

- `carrier_historical_delay_rate`
- `carrier_historical_flight_count`

### Origin

- `origin_historical_delay_rate`
- `origin_historical_flight_count`

### Destination

- `destination_historical_delay_rate`
- `destination_historical_flight_count`

### Route

- `route_historical_delay_rate`
- `route_historical_flight_count`

## Missing History

For groups with no previous observations, the historical rate is
undefined (`NaN`) and the historical count is zero.

These values will be handled during the modeling preprocessing stage.

## Important

Historical features must never be calculated using the full dataset
without respecting chronological order.

## Rolling protocol and prediction-time availability audit

The implemented protocol is a rolling daily forecast at the scheduled-departure
decision point. For each entity/date, the implementation aggregates daily target
sums and nonmissing-label counts, takes a cumulative sum by entity, and subtracts
the current day's aggregate. Pandas groups dates in sorted order. Consequently:

- A flight's own label and all same-date labels are excluded from its predictors.
- Labels on future dates cannot affect earlier or current-date predictors.
- During validation and test, completed labels from an earlier **calendar date**
  enter the history of later dates. For example, validation day D updates D+1
  validation/test histories, and test day D updates later test-day histories.
  They never fit the frozen estimator, calibrator or threshold.
- The base/calibration boundary is inside the original training period; train,
  calibration, validation and test dates remain strictly ordered. No date is split.

Synthetic tests in `tests/test_historical_availability.py` verify all four entity
aggregates against an independent strict-past calculation, shuffled input order,
own-label exclusion, multiple current/future-label perturbations, cross-partition
rolling updates, and chronological partition boundaries. They verify **date-order
properties**, not real-time outcome availability.

The CSV contains date fields, clock-valued scheduled/actual arrival and departure
fields, delays and `time_hour`. It has no explicit outcome-publication/ingestion
timestamp or verified absolute arrival timestamp with destination timezone and
date rollover. Clock fields alone do not prove that all earlier-date flight
outcomes were observable before today's prediction; overnight arrivals, long
delays and reporting latency remain unresolved. `time_hour` in this local CSV is
naive schedule-hour text and is not an outcome-availability event log.

An additional fixed calendar-date lag is technically feasible by shifting the
cutoff further into the past. It would reduce some overnight exposure, but no
finite date-only lag proves availability under arbitrary delays or reporting
latency. An availability-aware alternative would require verified absolute outcome
and publication timestamps, then an as-of join with availability strictly before
the prediction time. Do not reconstruct a guaranteed availability timestamp from
ambiguous clock fields alone. Neither alternative was substituted or selected
using final-test performance in this audit. The existing rolling protocol and all
model configurations remain unchanged.
