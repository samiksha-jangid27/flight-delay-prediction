# Data Dictionary

**Project 21: Flight Delay Prediction** · Team T022

Source: 2013 US flights (NYC origins — EWR, LGA, JFK). 336,776 raw rows;
327,346 kept after dropping rows with missing `arr_delay`.

Availability key:
- **pre** — known before departure → usable as a model feature.
- **post** — only known after departure → forbidden (leakage); excluded.
- **target** — the label.
- **id** — identifier / redundant; excluded from features.

## Raw columns

| Column | Type | Availability | Description |
|---|---|---|---|
| `id` | int | id | Row identifier. |
| `year` | int | id | Calendar year (all 2013; dropped as constant). |
| `month` | int | pre | Month of scheduled flight (1–12). |
| `day` | int | pre | Day of month of scheduled flight. |
| `sched_dep_time` | int | pre | Scheduled departure, HHMM (e.g. 515 = 05:15). |
| `dep_time` | float | **post** | Actual departure time. |
| `dep_delay` | float | **post** | Actual departure delay (min). |
| `sched_arr_time` | int | pre | Scheduled arrival, HHMM. |
| `arr_time` | float | **post** | Actual arrival time. |
| `arr_delay` | float | **post / target source** | Actual arrival delay (min). Drives the target. |
| `carrier` | str | pre | Airline code (e.g. UA, AA, DL). |
| `flight` | int | pre | Flight number. |
| `tailnum` | str | pre | Aircraft tail number. |
| `origin` | str | pre | Origin airport (EWR / LGA / JFK). |
| `dest` | str | pre | Destination airport. |
| `air_time` | float | **post** | Minutes in the air. |
| `distance` | int | pre | Route distance (miles). |
| `hour` | int | pre | Scheduled departure hour (0–23). |
| `minute` | int | pre | Scheduled departure minute. |
| `time_hour` | str | pre | Scheduled departure timestamp (hourly). |
| `name` | str | id | Airline full name (redundant with `carrier`). |

## Engineered columns

### Target (`features/target.py` — Samiksha)
| Column | Type | Description |
|---|---|---|
| `is_delayed` | int8 | **Target.** 1 if `arr_delay >= 15`, else 0. Base rate ≈ 25.9%. |

### Historical features (`features/historical.py` — Samiksha, prior calendar dates only)
| Column | Type | Description |
|---|---|---|
| `carrier_historical_delay_rate` | float | Carrier delay rate over prior dates. |
| `carrier_historical_flight_count` | int | Carrier flights seen on prior dates. |
| `origin_historical_delay_rate` | float | Origin delay rate over prior dates. |
| `origin_historical_flight_count` | int | Origin flights on prior dates. |
| `destination_historical_delay_rate` | float | Destination delay rate over prior dates. |
| `destination_historical_flight_count` | int | Destination flights on prior dates. |
| `route_historical_delay_rate` | float | Route delay rate over prior dates. |
| `route_historical_flight_count` | int | Route flights on prior dates. |

These features exclude same-day and future outcomes. Their availability at the
exact prediction timestamp has not been verified; see the
[data contract](data_contract.md).

### Airport congestion features (`features/congestion.py`)
| Column | Type | Description |
|---|---|---|
| `origin_hourly_departures` | int32 | NYC-origin flights scheduled to depart the same origin in the same departure-date/hour bucket. Computed before target filtering. |
| `dest_hourly_arrivals` | int32 | NYC-origin flights scheduled to arrive at the same destination in the same scheduled-arrival-hour bucket, grouped by departure date. Computed before target filtering; not a total-airport count and does not roll arrivals across midnight. |

These are schedule-based proxies available from the timetable, not realized
congestion. They are included in the 33-feature validation ablation only; the
canonical frozen test evaluation uses 31 features without them.

### Temporal features (`features/temporal.py` — Rashmi)
| Column | Type | Description |
|---|---|---|
| `day_of_week` | int8 | 0 = Monday … 6 = Sunday. |
| `is_weekend` | int8 | 1 if Saturday/Sunday. |
| `season` | int8 | 1 winter, 2 spring, 3 summer, 4 autumn. |
| `hour_sin`, `hour_cos` | float | Cyclic scheduled-hour encoding (period 24). |
| `month_sin`, `month_cos` | float | Cyclic month encoding (period 12). |
| `day_of_week_sin`, `day_of_week_cos` | float | Cyclic weekday encoding (period 7). |

### Route features (`features/route.py` — Rashmi)
| Column | Type | Description |
|---|---|---|
| `route` | str | `origin>dest` identifier. |
| `distance_band` | category | Binned distance (0-250 … 2000+). |
| `is_long_haul` | int8 | 1 if `distance >= 1500`. |
| `distance_log` | float | `log1p(distance)` — compresses the long tail. |

## Excluded from the model

`EXCLUDED_COLUMNS` in `features/selection.py`: `is_delayed`, `arr_delay`,
`dep_time`, `dep_delay`, `arr_time`, `air_time`, `id`, `name`, `year`.
See [feature_policy.md](feature_policy.md) for the leakage rules.
