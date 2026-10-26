# Feature Availability Policy

## Prediction Objective

Predict whether a flight will experience substantial delay using
information available before departure.

## Allowed Information

The model may use information known before departure, including:

- Scheduled departure time
- Scheduled arrival time
- Airline/carrier
- Origin airport
- Destination airport
- Route
- Calendar features
- Historical origin and destination statistics
- Historical carrier statistics
- Historical route statistics
- Schedule-based origin departures and destination arrivals, computed from
  the complete timetable before target filtering (validation-ablation variant
  only; not part of the frozen 31-feature model)

The congestion values are scheduled-load proxies over this NYC-origin dataset,
not total airport traffic. Destination arrivals are grouped by departure date
and scheduled arrival hour; an arrival crossing midnight is not assigned to
the next calendar date. See `data_contract.md` for the feature definitions and
limitations.

## Forbidden Information

The model must not use information that becomes available after
departure or after the prediction timestamp, including:

- Actual departure time
- Actual arrival time
- Departure delay
- Arrival delay
- Taxi-out time
- Wheels-off time
- Wheels-on time
- Any post-departure operational information

## Historical Feature Rule and Evaluation Protocol

The historical features use a **rolling daily forecast protocol**:

- For every flight on date **D**, historical rates/counts use labelled flights
  from dates strictly earlier than D.
- All flights on the same date share the same pre-date history. Outcomes from
  other flights on that date are excluded.
- Actual outcomes from earlier validation/test dates may update features for
  later dates, simulating daily predictions where the previous date's outcomes
  have become available.
- Validation/test outcomes must never fit the predictive model or preprocessing;
  they can only become historical inputs for later calendar dates under this
  explicitly rolling protocol.

This means held-out historical features are **not frozen at the train cutoff**.
If evaluation requires a static holdout, compute the aggregates from training
data only and freeze them for all validation/test dates instead.

Never allow same-day or future-date outcomes to influence a flight's features.

Prior-date labels are used under the documented rolling daily protocol, but
their availability at the exact operational prediction timestamp has not been
verified. Do not describe the feature set as proven deployable on that basis.
