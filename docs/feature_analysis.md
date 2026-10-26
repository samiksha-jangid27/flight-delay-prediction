# Feature Analysis

**Project 21: Flight Delay Prediction** · Team T022
**Owner:** Rashmi Anand (EDA & Feature Engineering)

Why each engineered feature may help, grounded in the EDA
([notebooks/01_data_exploration.ipynb](../notebooks/01_data_exploration.ipynb);
the notebook embeds plots and writes PNGs to `reports/figures/` when run).
All statistics are from the **training split only** (base delay rate ≈ **25.9%**).
Train covers the earliest ~70% of 2013 dates (roughly Jan–Sep) under the
chronological split.

All features below use pre-departure information only — see
[feature_policy.md](feature_policy.md).

---

## Temporal features (`features/temporal.py`)

### Scheduled hour — `hour_sin`, `hour_cos`
Delay rate climbs steeply through the day: **10.5% at 05:00 → ~38–40% by 20:00–21:00**.
Late flights inherit delays that cascade from earlier ones. A raw integer hour treats
23 and 0 as far apart; the **cyclic sin/cos encoding** keeps midnight adjacent to the
early hours, which matches reality.
*Expected effect:* strong, monotonic predictor — likely a top feature.

### Month / season — `month_sin`, `month_cos`, `season`
Clear seasonal structure: **June (31.5%) and July (32.2%)** are the worst months
(summer thunderstorms, peak travel), **September (16.4%)** the best. `season`
captures the slow winter/summer swing as a compact categorical; the cyclic month
pair captures finer month-to-month movement.
*Expected effect:* moderate; strongest for summer months.

### Day of week / weekend — `day_of_week`, `is_weekend`
Weekday flights are delayed more than weekend flights: **27.8% vs 20.4%**. Business
travel density and tighter weekday schedules drive this.
*Expected effect:* small but consistent lift.

**Why cyclic encoding (theory):** hour, month and weekday are periodic. Encoding each
as `(sin(2πx/P), cos(2πx/P))` places values on a circle so distance reflects true
temporal closeness. Linear models especially benefit, since they cannot otherwise
learn that 23:00 ≈ 00:00.

---

## Route features (`features/route.py`)

### Route identifier — `route`
`origin>dest` captures corridor-specific effects (congestion, air-traffic flow) that
origin and destination alone miss. High cardinality — modelers should target/one-hot
encode or lean on the route historical-delay-rate feature.

### Distance band / long-haul — `distance_band`, `is_long_haul`, `distance_log`
Longer flights are slightly *less* likely to be counted delayed on arrival:
**long-haul (≥1500 mi) 23.2% vs short-haul 26.7%** — longer cruises have slack to
absorb departure slippage. `distance_log` compresses the heavy right tail of the
distance distribution so models aren't dominated by a few transcontinental routes.
*Expected effect:* modest; mainly an interaction signal with carrier/hour.

**Boundary note:** route *historical delay rates* are produced by Samiksha's
leakage-safe historical engine (`features/historical.py`). This module only adds
**static, outcome-independent** route attributes and deliberately does not duplicate
that engine.

---

## Context: strong raw signals confirmed in EDA

These aren't engineered here but the EDA quantifies them and they justify keeping the
columns:

- **Carrier** spans a wide range: Hawaiian (HA) **13.8%** → Frontier (F9) **40.6%**.
  A very strong predictor.
- **Origin**: EWR **27.9%** > JFK **25.5%** > LGA **24.0%** — only 3 NYC airports, so
  limited variety (flag for the DL embedding phase).
- **Historical delay-rate features** correlate positively with the target (see EDA §9).
- **Airport congestion** (`origin_hourly_departures`, `dest_hourly_arrivals`, schedule-derived)
  represents time-specific scheduled load. On this single-hub/one-year dataset it adds little
  (weak, slightly negative correlation; low model importance) because the scheduled `hour`
  already captures most time-of-day congestion — but it satisfies the brief's congestion
  requirement and should matter more on a larger multi-airport dataset.

---

## Guidance handed to the modelers

1. **Encode** `carrier`, `origin`, `dest`, `route`, `distance_band` (one-hot or target
   encoding); **drop or hash** very high-cardinality `tailnum` and `flight`.
2. Use the **cyclic pairs** instead of raw `hour`/`month` for Logistic Regression.
3. Class imbalance (25.9% positive) → `class_weight="balanced"` (LR) /
   `scale_pos_weight ≈ 2.86` (XGBoost); evaluate with **PR-AUC**.
4. Fit all encoders on **train only**, transform validation/test — no fitting on
   held-out data.
