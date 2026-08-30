---
name: nbo-feature-engineering
description: Rules for the cross-sell preprocessing stage — anchor selection, the two leakage invariants, percentile grids, and how to add or change features via config. Use when editing nbo_data_preprocessing/preprocess.py or config.yaml, adding a numeric feature, changing product-family rollups, or debugging flat_dataset.csv / hazard_dataset.csv output.
---

# Feature engineering / preprocessing rules

Scope: `nbo_data_preprocessing/preprocess.py` + `nbo_data_preprocessing/config.yaml`.

## Two rules that must not be relaxed

1. **Outcome rows are removed before any aggregation.** They carry an id, a
   date, and a target flag; every other column is null. A groupby that still
   sees them will silently corrupt counts and "last row" lookups.
2. **No feature may use a row dated after the anchor**, and no feature may
   encode the distance from the anchor to the outcome or to the extract date.
   Both leak the label. Gaps are measured *between* acquisitions only.

## Anchor rules (differ by design)

- **fixed** (`flat_dataset.csv`): non-converters anchor at their last
  acquisition at least `horizon_months` before the extract date, so both
  classes get equal time-at-risk. Converters whose gap to the outcome exceeds
  the horizon are **excluded, never relabelled 0**.
- **hazard** (`hazard_dataset.csv`): everyone anchors at their last
  acquisition; censoring is handled by interval expansion, so no horizon is
  needed and nobody is dropped for timing.
- Converters anchor at their last acquisition *before* the outcome in both
  designs.

## Config-only changes

`preprocess.py` contains no column names, dates, or thresholds — all of that
lives in `config.yaml`. To point the pipeline at a new dataset or add a
feature, edit the config, not the script:

- New numeric feature → add an entry under `numeric_columns` (aggregations,
  `log1p`, `percentile_group`). It flows into the aggregate block
  automatically.
- To also carry it in the sequence block, add it to `sequence.carry_columns`
  (and `sequence.carry_percentiles` for the ranked version).
- To change product rollups, edit `product_families.map`. Unlisted
  `product_type` values fall into `default_family` — keep the family count
  small, since it drives the width of count/one-hot blocks downstream.

Note that `config.yaml` ships with `io:` paths under `/mnt/user-data/outputs/`,
which do not exist locally. Repoint `input_path`, `output_dir`, and
`artifact_dir` before running.

## Percentile grids

- Fit once on training (acquisition) rows, saved to
  `artifacts/percentile_grids.json`. **Never refit at scoring time** — that
  makes a customer's feature value depend on whoever else is in that day's
  batch.
- `percentiles.min_cell_size` controls fallback: a grouping level is used only
  if its cell held at least that many rows at fit time, else the next level in
  `percentile_group` is tried. A null result when every level is too thin is
  intended behavior, not a bug.

## Other known pitfalls

- Customers who keep buying after the anchor still have those later rows in
  the source file — `build_features` filters them out by date. Don't replace
  that filter with a plain "last N rows per customer"; a customer can have
  rows after the anchor that must stay excluded.
- `agg_tenure_months` requires full history back to the first-ever
  acquisition. Drop it from the config if the scoring payload can't supply a
  first-acquisition date, or training and serving will disagree.
- Hazard row count is not sample size — a large customer base still produces
  only a few hundred events, and the event count (not row count) governs how
  many features the data can support.
- The final partial interval is dropped when exposure falls below
  `hazard.min_exposure_frac` — but never when it contains the event.
- The per-customer logic is an explicit `groupby` loop by design (readability
  over vectorized performance, since the maintaining engineer works without
  foundation-model support). Don't rewrite it to be "faster" without asking.
  `build_features` is the only hot spot: ~120s for 25k customers, scaling
  roughly linearly to 20–25 minutes at 250k.
