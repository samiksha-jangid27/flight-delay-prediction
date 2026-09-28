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