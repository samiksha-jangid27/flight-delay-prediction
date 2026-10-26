# Evaluation audit — AML Phase 1

> **⚠️ Superseded run.** This audit records the original frozen evaluation of
> 22 October 2026, **before** the airport-congestion features were added. The
> **current canonical results** (33-feature model) are in
> [`paper/main.pdf`](../paper/main.pdf) and the [README](../README.md); they
> differ only marginally (e.g. test PR-AUC 0.337 → 0.338) and the conclusions
> hold. Numbers below are the dated log for the original feature set.

Executed 22 October 2026 on `fix/lakshya-evaluation-audit`, based on reviewed merge
`5be554cefb335de8030cc12c09bbb343eea7a527`. The branch adds verification and
supplementary analysis. Original classifier implementations, hyperparameters,
calibration maps, chronological boundaries and published selections are preserved.

## Reproducibility and original-run preservation

The initial full suite passed: **123 passed in 25.65 seconds**. The independent
auditor passed **469 checks** on the original output. A fresh isolated directory,
`reports/evaluation/repro.IOEi20`, was then created. Validation refitted only the original
specified early-training bases and training-only calibrators; choices were frozen
and compared before the final-test reproduction. Dataset, feature columns, exact
XGBoost parameters, dependency versions, date windows, calibration methods,
thresholds, selected model **and frozen joblib bytes matched exactly**.

The held-out test was reapplied once in that new directory solely to reproduce
already-published scores. It was not used to revise a model, map, threshold or
feature variant. The fresh output passed **469 independent checks**. These four
JSON files were identical to the originals after parsing:

- `validation_metrics.json`
- `calibration_metrics.json`
- `test_metrics.json`
- `validation_bootstrap.json`

Original validation/model-comparison and later calibration study are distinct:

| Experiment | Balanced LR validation AP | Tuned XGBoost validation AP |
|---|---:|---:|
| Original bases fitted on all 228,299 training flights | 0.206047 | 0.239206 |
| Early-training refit, selected isotonic calibration | 0.223285 | 0.212871 |

The later frozen test AP/F1 remains LR **0.294262/0.368642**, XGBoost
**0.337091/0.410833** on 48,581 flights. These are verified reproductions, not a new
winner-selection exercise. The frozen validation AP winner remains balanced LR.

## Identity and environment

- Dataset SHA-256: `8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa`.
- Original and reproduced joblib SHA-256: `fcacf7ae9a12e21297de0ba040bd8177fb49125cf44e2de2d98499603c6de217`.
- Python locally: 3.14.7, existing macOS environment, seed 42.
- numpy: 2.5.4.
- pandas: 3.0.6.
- scikit-learn: 1.9.1.
- xgboost: 3.4.1.

The local OpenMP runtime is the previously verified Homebrew bottle extracted
under ignored `.venv/libomp`; commands prepend that directory to
`DYLD_LIBRARY_PATH`. No system runtime or dependency version was replaced for
reproduction. CI explicitly targets Python 3.12 on Linux; local reproduction on
3.14 does not establish exact frozen-score equivalence on CI's environment.

## Independent auditor and confirmed fixes

`scripts/audit_evaluation.py` rebuilds canonical IDs, dates and >=15-minute labels
from raw data and the original unique-date rules. It independently recalculates
metrics with sklearn, without calling the producer's `evaluate()` helper. It
requires complete aligned populations, matching raw labels/dates, correct tags,
finite bounded probabilities, valid hashes and consistent frozen choices.
Calibration/threshold/model choices are replayed from validation predictions
before test metrics are opened. Every check prints PASS/FAIL; failures return
nonzero and are recorded, rather than replaced with expected scores. Each report
uses exclusive creation and a unique filename, protecting existing artifacts.

The CSV ingestion reproducibility problem is handled by the auditor's
`float_precision="round_trip"`. At LR's exact frozen threshold
`0.34357316087308004`, default pandas parsing changes 2,791 validation decisions
and 2,689 test decisions; saved test F1 0.368641982 becomes 0.353236642 under default
reload. Round-trip parsing reproduces the published values. Regression tests
exercise a threshold equal to a serialized score, tampering, duplicates, incorrect
labels/dates, missing files, altered thresholds, invalid probabilities and
non-overwriting report creation. No frozen scores or thresholds were rounded as
a repair.

README now reflects implemented LR/XGBoost/evaluation, qualifying date-order
safety. CI is added with read-only repository permissions, no custom secrets and
pinned official actions; offline rescheduling and deep learning remain pending.
The earlier prototype's stale status/safety wording is corrected; no classifier
implementation or selected-experiment behavior was changed.

## Probability-calibration baselines and evidence

| Candidate | brier | log_loss | average_precision |
|---|---:|---:|---:|
| base_train | 0.143920014 | 0.467913933 | 0.156402330 |
| train_calibration | 0.137829746 | 0.451675039 | 0.156402330 |
| logistic_balanced (frozen isotonic) | 0.150899305 | 0.482196703 | 0.223284804 |
| xgboost_tuned_full (frozen isotonic) | 0.137199003 | 0.442804442 | 0.212871158 |

Constants use only base-training (0.265852650) or calibration-training
(0.233142833) prevalence. LR loses to both constants on proper scores; XGBoost
slightly beats both. Calibration improves raw Brier/log loss for both but does
not establish universally useful probabilities. The in-sample validation oracle
0.131940641 remains explicitly descriptive. Neither baseline is promoted to a
new selected model and no operational threshold is changed.

Isotonic reduces LR validation scores from 50,464 distinct values to 84, creating
59,576,982 cross-class tied pairs and reducing AP/AUC. Probability means shift
upward from calibration to validation despite a lower outcome prevalence. Frozen
LR link diagnostics implicate calendar/covariate movement algebraically; history
counts' aggregate contribution goes downward, so they cannot alone explain the
increase. These diagnostics are not a causal experiment. Full verified values
and the reproducible analysis command are appended to
[evaluation_results.md](evaluation_results.md).

## Historical availability findings

All four aggregate types match strict-past-date arithmetic on synthetic shuffled
flights. Own/current/future labels cannot affect current or earlier predictors.
Earlier validation/test-day labels **do** update later days' historical features,
as specified by the rolling protocol; they do not fit the frozen classifier,
calibrator or threshold. Regression tests exercise both cross-boundary updates
and strict chronological train/calibration/validation/test windows.

The CSV has clock times and delays, not verified absolute outcome-publication
or ingestion timestamps. Prior-date labels may still have been unavailable before
a particular scheduled departure. Exact availability is **unresolved**, not a
verified leakage-free property. A further fixed calendar lag could be implemented
but cannot guarantee availability under arbitrary latency. An as-of availability
join needs verified event/publication timestamps and timezone/rollover semantics.
Neither alternative was substituted or tuned against the final test. See
[historical_features.md](historical_features.md).

## Remaining implementation and methodological limitations

- At the PR #5 snapshot, the original runner's incomplete-run guards allowed
  retries after interrupted plotting and unprotected orphan artifacts. These
  were not fixed by the independent auditor. The PR #6 hardening section below
  records the subsequent runner fixes. The earlier reproduction completed in a
  fresh directory without such failures.
- Producer finalization checks dataset/joblib hashes but does not enforce all
  recorded schema/version/source provenance. The independent auditor checks
  recorded versions and schema; it cannot retroactively prove execution history.
- At the PR #5 snapshot, the generic splitter accepted NaT dates while the
  independent audit rebuild rejected them. The PR #6 hardening below adds
  defensive splitter validation; the valid-data partition rule remains unchanged.
- XGBoost parameters were previously selected over all original training dates,
  including those subsequently reserved for calibration. This is conditional,
  rather than fully nested, calibration evidence.
- Validation selects multiple decisions, so its scores are optimistic. The
  date bootstrap assumes exchangeable dates, conditions on fitted decisions and
  does not account for full selection uncertainty. Its AP/AUC/F1 difference
  intervals include zero.
- Class imbalance, seasonal drift, one historical dataset, uncertain feature
  availability and unspecified false-alarm costs limit deployment conclusions.
- Mutable files cannot prove which labels a historical run consulted or prove
  that an earlier test run happened exactly once. Verified validation-only replay
  is reported separately from that unverified history.

## Checks and CI

Final full local suite: **145 passed, 0 failed, 0 skipped**.
CI-equivalent synthetic command: **138 passed, 0 failed, 0 skipped**.
Ruff checks for all added Python modules/tests and `git diff --check` passed.
All **63** files present under the original evaluation directory before this
reproduction remain byte-identical; only new reports and the isolated reproduction
were added. Both training-only constants share the project's unresolved prior-date
outcome-availability assumption; they do not use validation/test labels.
The seven existing tests in `test_load.py`, `test_pipeline.py` and `test_prepare.py`
read the real dataset and are excluded in CI. All other model fits operate on tiny
synthetic fixtures and temporary outputs. No real-data training, search or real
final-test scoring command is part of the workflow. Pinned action sources:
[checkout](https://github.com/actions/checkout) and
[setup-python](https://github.com/actions/setup-python).

During test development, one new fixture initially changed raw validation labels
without updating its saved prediction labels. The analysis correctly rejected
the mismatch. The fixture was corrected; this was not a production failure or a
reason to weaken alignment checks. Final execution outcomes, not this intermediate
failure, determine readiness. GitHub-hosted execution is tracked separately from
local CI-equivalent runs and must not be claimed successful before GitHub reports it.

## Exact local verification commands

Run from the repository root after `python -m pip install -e .`:

```bash
source .venv/bin/activate
# This local runtime workaround is needed only if OpenMP is not installed globally:
export DYLD_LIBRARY_PATH="$PWD/.venv/libomp:${DYLD_LIBRARY_PATH:-}"
python -m pytest -q
python scripts/audit_evaluation.py --data data/raw/flights.csv --output-dir reports/evaluation

# Preserve originals; mktemp chooses a NEW directory for a reproduction.
export AUDIT_OUT="$(mktemp -d reports/evaluation/repro.XXXXXX)"
python -m flight_delay.evaluation.evaluate validation \
  --data data/raw/flights.csv --output-dir "$AUDIT_OUT" --n-jobs 4 --bootstrap 300
python -m json.tool "$AUDIT_OUT/selection.json"
# Verify hashes, parameters, maps and thresholds before reproducing the published test:
python -m flight_delay.evaluation.evaluate final-test \
  --data data/raw/flights.csv --output-dir "$AUDIT_OUT"
python scripts/audit_evaluation.py --data data/raw/flights.csv --output-dir "$AUDIT_OUT"
python -m json.tool "$AUDIT_OUT/validation_metrics.json"
python -m json.tool "$AUDIT_OUT/test_metrics.json"

# Supplementary validation analysis; no test metrics used for baseline comparisons.
python -m scripts.analyze_calibration_baselines \
  --data data/raw/flights.csv --output-dir reports/evaluation --diagnose-logistic
python -m pytest -q --ignore=tests/test_load.py --ignore=tests/test_pipeline.py --ignore=tests/test_prepare.py
git diff --check
git status --short
```

Avoid rerunning the real test to search for better choices. Once a reproduction is
complete, repeated audits can read its saved CSV/JSON files without generating
new model predictions. The audit and baseline commands write unique reports;
raw data, joblib files, prediction streams and original metrics are never replaced.
Generated reports/models remain ignored and are not part of the commit.

## Evaluation hardening — PR #6

This follow-up changes safety and reproducibility, not model selection or scores.
The baseline on merged main `ed5b275` was **145 passing tests**. No real classifier
was retrained and no real final-test command was executed for this change.

### Confirmed root causes and fixes

1. **Prediction CSV ingestion:** default pandas float parsing can move a score
   below a threshold it originally equalled. Default numeric ID inference also
   destroys leading zeros. `flight_delay.evaluation.predictions.load_prediction_csv`
   now specifies `dtype={"flight_id": "string"}` and
   `float_precision="round_trip"`, validates required fields, labels, dates,
   unique/nonmissing IDs and finite [0,1] probabilities, and optionally requires a
   matching explicit partition tag. The older model CSVs have four core columns;
   their notebook consumer uses the helper without inventing partition metadata.
   Raw input loading now preserves `id` as a string as well, so rebuilding
   features cannot lose the ID's leading zeros. IDs are never model features.
   The independent auditor already used safe parsing and retains its own loader
   and independent sklearn metric calculations; the baseline script already
   consumed that safe auditor loader. No redundant metric refactor was made.
   Test fixtures that load non-threshold demonstration outputs are not external
   prediction APIs. External consumers must use this helper or the equivalent
   dtype/round-trip contract; the repository cannot enforce their pandas options.

2. **Interrupted output lifecycle:** previously, only selection.json or
   test_metrics.json protected a run, and these files were written after some
   other work. A failure could leave unguarded outputs. New validation requires
   an empty directory. Both stages exclusively create `.validation_started.json`
   or `.final_test_started.json` **before work**, then generate everything in
   `.validation_staging/` or `.final_test_staging/`. Publication creates files
   using non-replacing filesystem links and creates partition figure directories
   exclusively; it never replaces an existing destination. A corresponding
   `*_complete.json` marker is written and flushed only after publication succeeds.
   Completed staging is then removed. Shared `figures/` may remain from validation;
   an existing `figures/test/` is never merged or replaced.

   The start marker is a persistent at-most-once claim, not a PID lock that expires
   when a process dies. Exceptions retain the marker, staged artifacts and any
   partially published files. Concurrent cooperative invocations cannot both
   acquire the same exclusive marker. Finalization rejects old test metrics,
   `*_test.csv`, `figures/test`, final-stage markers/staging, or incomplete new-style
   validation before deserializing/scoring models. Read-only dataset/joblib hash
   verification remains in place. Legacy completed validation artifacts without
   phase markers remain readable, and legacy final-test files still block repeats.
   The independent auditor rejects incomplete or malformed new-style phase markers.
   The supplementary baseline reader also rejects incomplete validation before loading inputs.

3. **Unverified prediction origin:** the independent metric audit established
   consistency of saved CSVs, but did not regenerate their scores from estimators.
   `scripts/verify_frozen_predictions.py` now checks input hashes **before loading**
   the trusted local joblib, package versions, feature schema, model settings,
   calibrator identity and exact frozen threshold. It rebuilds the unchanged
   feature pipeline, validates the full saved validation ID/date/label population,
   and predicts a deterministic seed-42 sample selected from sorted string IDs.
   Rows are selected by ID, not row position. It compares calibrated probabilities
   with `atol=rtol=1e-12` and compares classes separately using the exact
   `probability >= threshold` rule with **no classification tolerance**. No fit,
   fit_transform, search or threshold-selection method is called. It opens no
   test metrics and makes no predictions on test records. The upstream builder
   still constructs all chronological partitions; its returned test frame is
   never scored or used for selection. The verifier writes only to stdout.
   Original LR manifests lack a complete parameter snapshot; the verifier checks
   the unchanged C/class-weight/solver/iteration settings in addition to the
   serialized artifact hash. It does not reconstruct an immutable fitting history.

4. **Invalid chronological dates:** the splitter previously counted NaT as an
   additional unique date. Missing components are now rejected before grouping;
   nonnumeric, non-integral or invalid calendar components produce clear
   ValueErrors. The original valid-data date fractions, rounding, record order
   and whole-date partitioning are unchanged. Full-population raw-ID/date checks
   and synthetic exact-window comparisons support this preservation.

### Recovery policy

Do not remove a start marker, edit completion markers, clear staging, or invoke a
retry to get around the safeguards. Preserve the failed directory and inspect its
traceback, stage contents and phase markers. After an interrupted **validation**,
use a new empty directory if a new training experiment is explicitly authorized.
After an interrupted **final-test** attempt, stop and manually review whether any
test scoring occurred; there is no automatic retry/unlock mode. Previously saved
results may be audited without scoring a model. If only plotting failed after
publication, inspect/recreate figures from the preserved saved predictions in a
separate location rather than repeating final-test scoring. This contribution
adds no automatic recovery/rescoring command.

These guards protect cooperative local process interruption/concurrency; they
are not an immutable remote execution ledger or a claim of fully transactional
filesystem behavior during hardware/power failure. Deleting files manually can
bypass any local file-based guard. Legacy unmarked runs cannot retroactively
prove their execution history. Those limitations are not converted into PASSes.

### Actual verification and regression coverage

- Full suite: **209 passed**, zero failed/skipped.
- CI-equivalent synthetic suite: **202 passed**, zero failed/skipped; the same
  seven real-data loader/preparation tests remain excluded by the existing CI.
- New regression cases cover round-trip threshold equality and repeated isotonic
  scores, leading-zero IDs, missing/invalid prediction fields, partial validation,
  completed and partial legacy final outputs, artifact-writing and publication
  failures, exclusive concurrent claims, preservation of existing destinations,
  marker validation, corrupt hashes/settings/labels/scores, deterministic sampling
  and a sub-tolerance score difference that still changes its class.
- Frozen-model replay on **100 validation flights**: both LR and XGBoost matched
  their saved calibrated probabilities exactly (maximum absolute error **0.0**)
  and matched every sampled class under their original thresholds. Model methods
  were called for validation predictions only; no model was fitted.
- Synthetic monkeypatch tests explicitly make model/calibrator fitting and
  preprocessing fit_transform fail if called during verification, forbid opening
  test_metrics.json, and check byte preservation on success and failure.
- The original independent audit is rerun against saved artifacts only. Source
  and artifact hash checks accompany the final review; original prediction CSVs,
  joblib bytes, test metrics and published score tables are not edited.

### Local verification commands

```bash
cd ~/Documents/flight-delay-prediction
source .venv/bin/activate
export DYLD_LIBRARY_PATH="$PWD/.venv/libomp:${DYLD_LIBRARY_PATH:-}"
python -m pytest -q
python -m pytest -q --ignore=tests/test_load.py --ignore=tests/test_pipeline.py --ignore=tests/test_prepare.py
python scripts/verify_frozen_predictions.py \
  --data data/raw/flights.csv --output-dir reports/evaluation \
  --partition validation --sample-size 100
python scripts/audit_evaluation.py --data data/raw/flights.csv --output-dir reports/evaluation
git diff --check
git status --short
```

Do **not** rerun the real final-test command to validate these fixes. No published
score, threshold, map, hyperparameter or validation choice is improved or changed
here. Historical outcome availability remains unverified with the current clock-
valued CSV, prior hyperparameter selection is not fully nested with respect to the
calibration block, and already-evaluated final-test results must not be used for
new optimization. This remains a Phase 1 academic evaluation, not a claim of
production readiness or complete leakage freedom.

The verifier checks a deterministic sample, not every stored probability. It does
not assert predictive performance on a new test set or establish immutable model
provenance. Full frozen test prediction replay is intentionally not performed.

Final preservation check: all **124** pre-existing local evaluation files (including
the earlier reproduction) remained byte-identical. GitHub-hosted CI for this
uncommitted hardening branch has not run; its unchanged Python 3.12 workflow and
local synthetic equivalent were reviewed. Publication is left for the explicit
post-review Git step rather than performed automatically.
