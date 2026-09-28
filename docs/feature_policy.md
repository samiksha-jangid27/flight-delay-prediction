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
- Historical airport statistics
- Historical carrier statistics
- Historical route statistics

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

## Historical Feature Rule

Historical aggregates must only use information from flights occurring
before the flight being predicted.

Future observations must never influence historical features.