# PPT CONTENT PACK (10 slides) — Flight Delay Prediction

## INSTRUCTIONS TO THE AI (paste at the top)

> Create a clean, modern, **"journey"-style 10-slide presentation** for an academic project
> **Flight Delay Prediction & Rescheduling**. The audience knows **nothing** about ML, so explain
> simply and build up step by step. **Design:** light theme, aviation blues/teal, one idea per
> slide, large fonts, **max 5–6 short bullets on a slide**, put detail in **speaker notes**. Use
> icons. Where a slide says `[IMAGE: filename]`, insert that image (large, readable) from the
> project's `paper/figures/` folder. **Keep every number exactly as written — they are real
> results.** Tone: confident, honest, scientific — this project's strength is rigor and honesty.

**Images (in `paper/figures/`):** `delay-hour-month.png`, `delay-carrier.png`, `cyclic-hour.png`,
`prevalence-shift.png`, `val-roc.png`, `val-pr.png`, `confusion.png`, `xgb-importance.png`,
`xgb-netbenefit.png`, `target-distribution.png`.

---

## SLIDE 1 — Title & the problem

**Title:** Flight Delay Prediction — predicting delays *before* departure, honestly

**On slide:**
- **Story:** 6 AM at Newark. Will tonight's 7:40 PM flight land **15+ min late**? If yes, the
  ops team can act now — swap aircraft, add buffer, re-book.
- **But at 6 AM we don't know** the actual departure/air time.
- **Our task:** predict **arrival delay (Yes/No)** using **only pre-departure info** (schedule,
  route, carrier, history). 15 min = the US **BTS** standard.
- Project 21 · Team T022 · Phase 1 (Advanced ML)

**Speaker notes:** "We predict whether a flight lands 15+ minutes late, using only what's known
before it takes off. That one constraint — no post-departure data — shapes everything."

---

## SLIDE 2 — Dataset & cleaning

**Title:** The Data: 2013 US flights from New York

**On slide:**
- Kaggle 2013 US flights (`nycflights13`): **336,776 flights**, 21 columns.
- **3 airports** (EWR, JFK, LGA) · **16 airlines** · **224 routes**.
- Cleaned: dropped flights with no arrival outcome → **327,346 usable** (0 duplicates).
- About **26%** of flights are delayed → the minority class.

**[IMAGE: target-distribution.png]**

**Speaker notes:** "One year of real NYC flights. We only drop rows where the answer doesn't exist.
Only ~26% are delayed — an imbalance that makes plain accuracy misleading, so we use PR-AUC and
ROC-AUC."

---

## SLIDE 3 — EDA: what drives delays

**Title:** Exploring the data — finding the signals

**On slide:**
- **Time of day is #1:** delay risk rises from **~10.5% (morning) to ~40% (night)**.
- **Season:** worst July (~32%), best September (~16%).
- **Airline:** varies hugely, **~14% to ~41%**.
- **Airport:** EWR 27.9% > JFK 25.5% > LGA 24.0%.
- (All explored on training data only — no peeking.)

**[IMAGE: delay-hour-month.png]** (optional second image: `delay-carrier.png`)

**Speaker notes:** "Before modelling we found the real drivers: time-of-day, season, airline, and
route. Each insight became a feature. Delays cascade through the day, which is why night flights are
far more likely to be late."

---

## SLIDE 4 — Feature engineering + leakage defence

**Title:** 33 honest features — and three walls against cheating

**On slide:**
- **33 features:** schedule/calendar + route + historical delay rates (prior dates only) + airport congestion.
- **Clever trick — cyclic time:** hour 23 and hour 0 are 1 hour apart, not 23 → we map time onto
  a circle (sin/cos) so the model understands it.
- **3 walls vs data leakage:** (1) pre-departure allowlist, (2) past-dates-only history,
  (3) train on past → test on future.

**[IMAGE: cyclic-hour.png]**

**Speaker notes:** "We engineered 33 features, all available before departure — including schedule-based airport-congestion counts. The cyclic encoding
teaches the model that late night is close to early morning. And three independent safeguards make
sure no 'future' information ever sneaks in — this is what makes the model actually deployable."

---

## SLIDE 5 — Our approach (split, models, evaluation)

**Title:** How we built and tested it — fairly

**On slide:**
- **Split by date** 70/15/15 (train on past, test on future): 181k / 50k val / 49k test.
- **Prevalence shifts by season:** train 25.9%, val 15.6%, test 26.9% — a real effect a
  chronological split *exposes* (a random split would hide it).
- **Two models:** Logistic Regression (simple baseline) vs **XGBoost** (strong learner).
- **Calibration:** probabilities made honest (isotonic) so "0.3" ≈ 30% real risk.
- **Frozen test:** every choice locked on validation, then test run **once** — no cheating.

**[IMAGE: prevalence-shift.png]**

**Speaker notes:** "We split by time like real deployment. Notice the delay rate differs across
partitions — validation lands in the low-delay autumn months — so we don't compare
precision-recall numbers across windows as 'improvement.' We compare a linear baseline against
XGBoost on identical data, calibrate the probabilities, and lock everything before touching the test
set so the final number is unbiased."

---

## SLIDE 6 — Results

**Title:** Results — XGBoost is our best model

**On slide (held-out TEST set):**

| Metric | Logistic Reg. | **XGBoost** |
|---|---:|---:|
| ROC-AUC | 0.545 | **0.599** |
| PR-AUC | 0.296 | **0.338** |
| Recall | 0.465 | **0.542** |
| F1 | 0.364 | **0.412** |

- XGBoost catches **~54% of true delays**.

**[IMAGE: val-roc.png]** + **[IMAGE: val-pr.png]** (optional: `confusion.png`)

**Speaker notes:** "On the locked test set XGBoost wins every metric. ROC-AUC 0.60 means modestly
better than random — which, as the next slide shows, is the *honest* ceiling of this problem."

---

## SLIDE 7 — ⭐ Why our numbers are ~0.60 (and why that's GOOD)

**Title:** Honesty: we could show 90% — we refuse to

**On slide:**
- On this data, "90% accuracy" comes from only two tricks:
  - **Leakage:** using `dep_delay` (known only *after* departure).
  - **Imbalance:** "never delayed" already scores **84%** accuracy.
- **We proved it on our own data:**

| Model | ROC-AUC | Accuracy |
|---|---:|---:|
| "Never delayed" (no model) | — | 0.84 |
| **Our honest model** | **~0.60** | ~0.60 |
| Leaky model (uses `dep_delay`) | **0.88** | **0.90** |

**Speaker notes:** "This is our key slide. High accuracy here is either cheating with post-departure
data or just predicting the majority class. We tested it — adding departure delay gives us 90% too,
but that model can't be deployed. Our 0.60 is honest. That honesty is our contribution."

---

## SLIDE 8 — From prediction to decision (rescheduling)

**Title:** Should we actually act? A cost-based rule

**On slide:**
- Built a **cost model**: act on a flight only if it's worth it.
- Cost-optimal threshold **τ\* = C_action / (recovery × C_delay)**.
- **Finding:** acting beats doing nothing **only when action is cheap** (≤ ~15% of a delay's cost).
- Shows *why* calibrated probabilities matter.

**[IMAGE: xgb-netbenefit.png]** (optional: `xgb-importance.png`)

**Speaker notes:** "A probability isn't a decision. Our cost rule tells operators when intervening
actually saves money — only when the fix is cheap relative to the delay. This is the practical
payoff of calibration."

---

## SLIDE 9 — Limitations & how we improve

**Title:** Honest limitations → how we make it stronger

**On slide:**
- **Limits:** no weather (biggest missing driver); no real-time congestion; only 1 year & 3 NYC
  airports; discrimination modest (~0.60); not production-ready.
- **Improvements (legitimate):** add **weather data**, add **live congestion**, use the larger
  **BTS multi-year** dataset → then **Deep Learning** (next slide).

**Speaker notes:** "We're upfront about the ceiling. The only honest way to raise the score is more
signal — weather, live congestion, more data — which leads straight into our deep-learning phase."

---

## SLIDE 10 — Deep Learning future & conclusion

**Title:** Phase 2 — Deep Learning roadmap

**On slide:**
- **Why DL:** learn rich **embeddings** of airline/airport/route + complex interactions our manual
  features miss.
- **The plan (journey):** ✅ baseline (this project) → build an **MLP with embeddings** →
  optional temporal model → tune & validate (same leakage rules) → **fairly compare vs XGBoost on
  the same test**.
- **How it helps:** better generalisation, scales with weather + BTS data. DL must *prove* it beats
  XGBoost — not assumed.
- **Takeaway:** a leakage-safe, calibrated, honestly-evaluated predictor + a clear path forward.

**Speaker notes:** "Deep learning is Phase 2. We'll add an MLP with learned embeddings, keep every
safeguard, and only accept it if it genuinely beats XGBoost on the same honest test. Our project's
real achievement is rigor — a model that would actually work in deployment, with a clear roadmap to
improve it."

---

## QUICK NUMBER SHEET (don't mis-state these)

- Raw **336,776** × 21 → usable **327,346** (dropped 9,430; 0 dups). 16 carriers · 3 origins · 224 routes · 2013.
- Base rate ≈ **26%**; val prevalence 0.156, test 0.269. Features **33** (incl. airport congestion). Seed **42**. **223** tests.
- Split: base **181,405** / calib **46,894** / val **50,466** / test **48,581**.
- **Test:** LR → ROC 0.545, PR 0.296, F1 0.364 · **XGBoost → ROC 0.599, PR 0.338, Rec 0.542, F1 0.412**.
- Honesty proof: "never delayed" acc **0.84** · leaky (`dep_delay`) ROC **0.88** / acc **0.90** · our honest model ROC **~0.60**.
