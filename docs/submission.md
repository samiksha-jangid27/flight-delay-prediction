# Project 21 submission and viva

**Team T022, Section C**

Flight Delay Prediction via Gradient Boosting and Temporal Deep Learning

| Material | File |
| --- | --- |
| Presentation with speaker notes | [PowerPoint deck](../presentation/project21-presentation.pptx) |
| IEEE report | [PDF](../paper/main.pdf) · [LaTeX source](../paper/main.tex) · [build instructions](../paper/README.md) |
| Evaluation evidence | [Phase 1 manifest](canonical_evaluation.json) · [Phase 2 metrics](../reports/dl/metrics.csv) · [calibration and threshold protocol](dl_evaluation.md) |
| Supplementary controls | [Matched-seed ablation](../reports/dl/controlled_ablation.csv) · [Schedule-and-route comparison](../reports/dl/schedule_only_metrics.csv) |

## Demo from a clean checkout

Use Python 3.12 and run from the repository root. The tracked flight CSV and
the published MLP checkpoint are sufficient for the checkpoint replay.
Use a separate output directory for new comparisons.

```bash
python -m pip install -e '.[dl]'
python scripts/verify_dl_checkpoint.py
python scripts/compare_aml_dl.py --bootstrap 300 --output reports/evaluation/demo-phase2
```

The checkpoint verifier should report 50,466 aligned validation predictions.
The comparison writes the metrics CSV, reliability data and plots. For a
full regeneration, follow the [pipeline runbook](../README.md#reproducing-the-full-pipeline),
including the Phase 1 validation run that produces the frozen XGBoost
checkpoint before rebuilding the hybrid. The [rescheduling analysis](rescheduling_analysis.md)
documents the decision rule and cost assumptions.

## Presenter roles

| Team member | Lead topic | Prepare to explain |
| --- | --- | --- |
| Lakshya | Dataset and chronology | Label definition, date windows, and the historical availability caveat |
| Ankit Raj Singh | AML and encoding | Logistic Regression, train-fitted categories, and missing and unseen IDs |
| Samiksha Jangid | MLP architecture | Embedding lookups, the dense tensor, and the single flight-delay logit |
| Rashmi Anand | Training and ablation | BCE with logits, AdamW, early stopping, and the three paired seeds |
| Shubhaang Kataruka | Evaluation and decisions | Flight-level alignment, calibration, hybrid results, and the cost threshold |

Each member should also explain the full data-to-decision path and one
limitation. Keep Phase 1 held-out results separate from Phase 2 validation.

## Viva questions

- **What does the model predict?** Arrival delay of at least 15 minutes.
  Undefined arrival outcomes are excluded from the labeled dataset.
- **Why use embeddings?** They let the MLP learn carrier and airport
  representations jointly with the tabular predictors. The validation results
  measure whether that representation improved this dataset's scores.
- **What did the hybrid add?** A logistic model fitted on calibration flights
  combines the XGBoost and MLP probabilities. On validation, its Brier score
  was 0.000054 lower than calibrated XGBoost's, while AP was lower.
- **Where did calibration happen?** The probability maps were fitted on the
  late-training calibration window. Phase 2 comparisons use the same 50,466
  validation flights for each model.
- **When does the rescheduling rule trigger?** The decision threshold is
  action cost divided by recoverable delay cost. A cost ratio of 0.25 and
  recovery of 0.8 give a threshold of 0.3125.

At the headline cost assumptions, inaction had lower simulated cost than
either AML-guided policy. Historical input availability is conditional in
the 31-feature experiment. The schedule-and-route control excludes the eight
outcome-history inputs.

The cover photo is [John F. Kennedy Airport 2021c](https://commons.wikimedia.org/wiki/File:John_F._Kennedy_Airport_2021c.jpg)
by Antony-22, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/),
resized and cropped for the presentation. It illustrates the airport and
dates from 2021. The model data are from 2013.
