# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Cross-sell propensity modeling: identify non-wealth bank customers (loan/deposit
holders) likely to buy a wealth product, from their product-acquisition sequence
and the dollars tied to each product.

## Layout

```
nbo_data_preprocessing/   preprocess.py + config.yaml     raw rows -> modeling datasets
nbo_data_modeling/        model.py + model_config.yaml    fit + evaluate
                          make_synthetic.py               synthetic raw file (smoke test)
.claude/rules/            deeper rules per area (see "Rules files" below)
```

Both scripts default `--config` to a bare filename, so **run each from its own
directory** or pass an explicit path. `make_synthetic.py` lives under
`nbo_data_modeling/` even though it feeds preprocessing.

## Key commands

```bash
# smoke-test data (writes a raw-schema file; --seed, --base-rate also available)
cd nbo_data_modeling && python make_synthetic.py --customers 25000 --out synthetic.csv

cd nbo_data_preprocessing && python preprocess.py --config config.yaml
cd nbo_data_modeling      && python model.py      --config model_config.yaml
```

**Before any run, fix the `io:` paths.** Both configs point at
`/mnt/user-data/outputs/...`, a sandbox path from where this code was authored.
It does not exist on this machine, and `config.yaml`'s `input_path`
(`sample_raw_acquisitions.csv`) is not in the repo. Repoint `io.input_path`,
`io.output_dir`, and `io.artifact_dir` before running anything.

No test suite, linter config, `requirements.txt`, or lockfile exists. Verified
working locally: Python 3.10.7, pandas 2.1.4, numpy 1.26.4, pyyaml 6.0.2,
scikit-learn 1.7.2. Preprocessing needs only pandas/numpy/pyyaml; `model.py`
adds scikit-learn.

## Architecture

```
load -> split outcome rows -> pick anchor -> build features -> attach labels
```

Steps 4 and 5 are independent, so **one feature builder serves both designs**.
The designs differ only in the anchor rule and the label shape, and
`preprocess.py` emits both from a single pass over the config:

- `flat_dataset.csv` — one row per eligible customer, fixed-horizon label
- `hazard_dataset.csv` — one row per customer per interval at risk
- `exclusions.csv` — every dropped customer with a reason, per design
- `artifacts/percentile_grids.json` — fitted rank grids, **reloaded at scoring
  time, never refit**

`model.py` reads either file; `design: flat|hazard` in `model_config.yaml`
switches everything else with no code change. Hazard runs predict a
per-interval hazard and combine them into `P(convert by H) = 1 - prod(1 - h_k)`,
so both designs land on the same customer-level footing and their `metrics.csv`
numbers are directly comparable. It writes `metrics.csv`,
`metrics_by_repeat.csv`, `calibration.csv`, `features.csv`, `predictions.csv`.

**Anchor rules.** Converters anchor at their last acquisition *before* the
outcome, in both designs. Non-converters differ: *fixed* anchors at the last
acquisition at least `horizon_months` before the extract date so both classes
get equal time at risk (converters past the horizon are **excluded, never
relabelled 0**); *hazard* anchors at the last acquisition, with censoring
handled by interval expansion instead.

## Invariants — do not relax without asking

- **Outcome rows are split out before any aggregation.** They carry only id,
  date, and target; every other column is null. A groupby that still sees them
  silently corrupts counts and last-row lookups.
- **No feature may use a row dated after its anchor**, or encode the distance
  from the anchor to the outcome or extract date. Both leak the label. Gaps are
  measured *between* acquisitions only, never anchor-forward.
- Customers who keep buying after the anchor still have those rows in the source
  file; `build_features` filters by date. Do not swap that for a plain "last N
  rows per customer".
- **Percentile grids are fit once on training rows and saved.** Refitting at
  scoring time makes a feature depend on who else is in that batch.
- **Splits and sampling happen on the whole customer, never the row** —
  `StratifiedGroupKFold` on `columns.group`, and
  `sampling.negative_customer_fraction` drops whole customers. Dropping some of
  a customer's intervals corrupts the survival product.
- No oversampling or SMOTE: `class_weight="balanced"` plus post-hoc calibration.

## Conventions

- **Config-driven.** `preprocess.py` holds no column names, dates, or
  thresholds. Point at a new dataset by editing `config.yaml` only. A new
  numeric feature is an entry under `numeric_columns` (flows into aggregates
  automatically); add it to `sequence.carry_columns` /
  `carry_percentiles` to carry it per sequence position.
- **The per-customer `groupby` loop in `preprocess.py` is deliberate** — slower
  than vectorized, far easier for a human to read and modify without
  foundation-model help. Don't "optimize" it into a vectorized rewrite unless
  asked. `build_features` is the only hot spot. Expect ~120s for 25k customers,
  scaling roughly linearly (20–25 min at 250k), not a couple of minutes.
- **Rare-event metrics:** ROC-AUC is near-useless below ~1% prevalence and is
  written for reference only — read PR-AUC and lift-at-top-K. Row count is not
  sample size; the *event* count limits how many features the data supports.
- Lift is measured against the prevalence of the *sampled* data. Set
  `negative_customer_fraction: 1.0` before reporting final numbers (it ships
  at 0.25).
- `features.csv` sign-consistency (logistic only) is the stability check: below
  ~0.7 is unproven, not a finding. Never report coefficients from a
  non-converged fit — `saga` is the only elastic-net solver and is slow on wide
  hazard files, so lower `logistic.max_iter` or the sampling fraction before
  reaching for a different model.

## Ask rather than decide alone

- Any change to the leakage invariants or anchor cutoffs above.
- `min_cell_size`, `horizon_months`, `min_exposure_frac`, `min_acquisitions`, or
  other thresholds affecting eligibility and labels — these are modeling
  decisions, not tuning.
- Switching `model.type` (`logistic` vs `gbm`) or the CV scheme.
- Any dependency beyond pandas / numpy / pyyaml / scikit-learn — and confirm
  whether the environment is air-gapped for the task before assuming a package
  can be installed at all.
- Whether a change must stay explainable for MRM (model risk management)
  review. This code is maintained by an engineer without foundation-model
  support for troubleshooting; favor traceable and self-explanatory over clever,
  and keep outputs directly actionable for real cross-sell decisions.

## Skills

Deeper per-area detail lives in `.claude/skills/`, loaded on relevance rather
than every session: `nbo-feature-engineering` (preprocessing, anchors, leakage,
percentile grids), `nbo-modeling` (CV, sampling, metrics, solver), and
`nbo-synthetic-data` (the smoke-test generator and its limits).

## Repo state

Pipeline outputs are `.gitignore`d — the CSVs currently in the working tree are
synthetic-run artifacts, not fixtures. `preprocess.py` is untracked and should
be committed. Each subdirectory `README.md` now covers only its own stage and
cross-links to the other.
