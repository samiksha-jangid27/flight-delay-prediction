# Offline Rescheduling What-If Analysis

Executed 24 October 2026 on the validation (development) partition. This is the
decision-analysis half of the project's expected deliverable: *"a delay-risk
model with calibrated probabilities and an offline what-if rescheduling analysis
based on explicit assumptions."* It consumes the frozen calibrated predictions
produced by [`evaluation/evaluate.py`](../src/flight_delay/evaluation/evaluate.py)
and fits, selects or calibrates nothing.

## What this analysis is, and is not

It is a retrospective replay. Given each flight's calibrated pre-departure delay
probability, we ask: *if we had applied a proactive rescheduling rule, what would
the net operational cost have been, relative to doing nothing?* It is **not** a
claim of realisable savings. No operational intervention cost has been approved,
so every monetary quantity is expressed relative to the delay cost and reported
as a sensitivity across assumed cost ratios rather than a single headline number.

## Explicit assumptions

The cost model is intentionally simple and fully stated. Per flight, with a
single proactive action (for example an aircraft or crew swap, an added schedule
buffer, or pre-emptive passenger rebooking):

| Symbol | Meaning |
|---|---|
| `C_d` (`cost_delay`) | Disruption cost of a substantially delayed arrival that was **not** acted on |
| `C_a` (`cost_action`) | Cost of taking the action, paid whenever we act, regardless of outcome |
| `r` (`recovery`) ∈ [0,1] | Fraction of the delay cost avoided when we act on a flight that **does** delay |

Realised cost by decision `a` and outcome `y`:

| | `y = 0` (on time) | `y = 1` (delayed) |
|---|---|---|
| `a = 0` (no action) | `0` | `C_d` |
| `a = 1` (action) | `C_a` | `C_a + (1 − r)·C_d` |

Four further assumptions, each a stated limitation:

1. **Offline, no feedback.** Acting on one flight does not change any other
   flight's outcome. Real rescheduling propagates through aircraft rotations,
   crew legality and gate capacity; those effects are out of scope.
2. **Homogeneous costs.** `C_d`, `C_a` and `r` are identical across flights. In
   reality they vary by aircraft size, connection load and time of day.
3. **One binary action.** No partial or graduated interventions.
4. **Decisions use the calibrated probability** with an action threshold τ:
   act iff `P(delay) ≥ τ`.

## The two results that make this a *model-driven* rule

**Net savings have a closed form.** For a policy acting on `tp` true delays and
`fp` on-time flights (false alarms), savings versus the no-action status quo are

```
net_savings = tp · (r·C_d − C_a) − fp · C_a
```

Each correctly flagged delay returns `r·C_d − C_a`; each false alarm costs `C_a`.

**The cost-optimal threshold is `τ* = C_a / (r·C_d) = (C_a/C_d) / r`.** Acting on
a flight has non-negative expected value exactly when its calibrated probability
clears this boundary. This is why calibration (the Guo et al. pre-read) matters
operationally: the decision boundary is read directly off the cost assumptions,
and only a well-calibrated probability makes `τ*` the right place to cut. The
rule generalises decision-curve net benefit with an explicit recovery factor.

Baselines for context: **no action** costs `C_d · (#delayed)`; **reschedule
everything** costs `C_a · N + (1−r)·C_d · (#delayed)`; the **perfect-foresight
oracle** is the attainable lower bound and sets the scale for "oracle efficiency"
(the fraction of attainable savings a policy captures).

## Results — validation partition (50,466 flights, delay prevalence 0.1564)

### Headline operating point: `C_a/C_d = 0.25`, `r = 0.8` (so `τ* = 0.3125`)

Per-flight baseline costs (units of `C_d`): no action **0.1564**, reschedule all
**0.2813**, oracle **0.0704**.

| Model | acts on | precision | recall | net/flight | vs status quo | beats no-action |
|---|---:|---:|---:|---:|---:|:--:|
| Balanced LR (isotonic) | 60.8% | 0.196 | 0.764 | **−0.0564** | −36.1% | ✗ |
| Tuned XGBoost (isotonic) | 27.6% | 0.233 | 0.411 | **−0.0175** | −11.2% | ✗ |

At moderate action cost **both policies are worse than doing nothing.** This is
the honest consequence of weak discrimination (validation ROC-AUC ≈ 0.60–0.65,
precision ≈ 20–23%): the false-alarm cost outweighs the mitigated delays. Balanced
LR is worst because its isotonic map pushes 61% of flights over `τ*`, over-acting
severely. At the frozen F1 thresholds the picture is the same (LR −0.030/flight,
XGBoost −0.033/flight; neither beats no-action).

### Sensitivity: where the rule *does* pay off

Net savings per flight (units of `C_d`) at the cost-optimal threshold for each
assumed cost ratio, recovery fixed at 0.8:

| `C_a/C_d` | `τ*` | LR net/flight | LR beats no-action | XGB net/flight | XGB beats no-action |
|---:|---:|---:|:--:|---:|:--:|
| 0.05 | 0.0625 | +0.0751 | ✓ | +0.0751 | ✓ |
| 0.10 | 0.1250 | +0.0254 | ✓ | +0.0341 | ✓ |
| 0.15 | 0.1875 | −0.0160 | ✗ | +0.0047 | ✓ |
| 0.20 | 0.2500 | −0.0488 | ✗ | −0.0116 | ✗ |
| 0.25 | 0.3125 | −0.0564 | ✗ | −0.0175 | ✗ |
| 0.30 | 0.3750 | +0.0001 | ✓ | −0.0209 | ✗ |

Three findings:

1. **There is a break-even cost ratio.** The model-guided rule beats the status
   quo only when the action is cheap relative to a delay: up to `C_a/C_d ≈ 0.10`
   for Balanced LR and up to `≈ 0.15` for XGBoost. Above that, intervention is not
   justified by these probabilities.
2. **XGBoost dominates Balanced LR across the sweep** — a positive-savings region
   that extends further and higher oracle efficiency at equal cost ratios (e.g. 31%
   vs 23% at ratio 0.10). This is a decision-relevant reason to prefer XGBoost that
   the ranking metrics alone did not make vivid, and it is consistent with
   XGBoost's stronger held-out AP/ROC-AUC.
3. **Best attainable efficiency is modest** (≈ 64% of the oracle at ratio 0.05,
   falling fast as actions get costlier), bounding how much any policy built on the
   current probabilities can deliver.

(The isolated row at LR `C_a/C_d = 0.30` is a degenerate high-threshold point acting
on only 238 very-high-probability flights; it is near zero and not operationally
meaningful.)

## Conclusions and recommendation

A calibrated delay model supports a defensible rescheduling rule **only under
favourable cost assumptions** — cheap, effective interventions on a minority of
high-probability flights. Under moderate costs the current models do not beat
doing nothing, and Balanced LR in particular over-acts. If a rescheduling rule is
pursued, XGBoost at a cost-derived threshold `τ* = (C_a/C_d)/r` is the better
candidate, and the organisation must first supply real `C_a`, `C_d` and `r`
estimates. These validation numbers are development-stage; the held-out test set
was deliberately **not** spent on this what-if, preserving the frozen final-test
discipline described in [evaluation_results.md](evaluation_results.md).

## Reproduce

```bash
# 1. Generate frozen calibrated predictions (validation stage), if not present
python -m flight_delay.evaluation.evaluate validation \
  --data data/raw/flights.csv --output-dir reports/evaluation

# 2. Run the offline rescheduling what-if over the selected calibrated frames
python scripts/rescheduling_analysis.py --eval-dir reports/evaluation \
  --cost-ratio 0.25 --recovery 0.8
```

The script discovers each model's selected calibration variant and frozen
threshold from `selection.json`, then writes per-model JSON reports, a cost-ratio
sweep CSV, a net-benefit-curve CSV and a figure under
`reports/evaluation/rescheduling/`, plus a combined `rescheduling_summary.json`.
Pass `--predictions <csv>` to analyse any single saved-prediction CSV (for
example a `*_test.csv`, only with the deliberate choice to spend the held-out
test). The core logic and its guarantees are unit-tested in
[`tests/test_rescheduling.py`](../tests/test_rescheduling.py), including the
closed-form savings identity, the `τ* = C_a/(r·C_d)` boundary and the oracle
upper bound on every policy.

## Limitations

- No approved costs; results are a sensitivity study, not a forecast of savings.
- No-feedback and homogeneous-cost assumptions are strong; real network and crew
  effects are not modelled.
- Built on validation probabilities, which the evaluation audit shows are not
  convincingly better than a prevalence baseline on proper scores; the ceiling on
  achievable savings is correspondingly low.
- Inherits every upstream caveat, including the unverified exact availability of
  historical features at prediction time.
