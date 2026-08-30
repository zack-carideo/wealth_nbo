# Cross-sell preprocessing

Turns raw product-acquisition rows into two modelling datasets. For fitting and
evaluating a model on those datasets, see
[`../nbo_data_modeling/README.md`](../nbo_data_modeling/README.md).

```
python preprocess.py --config config.yaml
```

Dependencies: `pandas`, `numpy`, `pyyaml`. Nothing else.

`config.yaml` ships with `io:` paths under `/mnt/user-data/outputs/`, which is a
sandbox path from where this was authored. Repoint `input_path`, `output_dir`
and `artifact_dir` before the first run.

## Outputs

| File | Shape |
|---|---|
| `model_data/flat_dataset.csv` | one row per eligible customer, fixed-horizon label |
| `model_data/hazard_dataset.csv` | one row per customer per interval at risk |
| `model_data/exclusions.csv` | every dropped customer, with the reason, per design |
| `artifacts/percentile_grids.json` | fitted rank grids — **load this at scoring time, never refit** |

## Pointing it at a new dataset

Edit `config.yaml` only. `preprocess.py` contains no column names, no dates
and no thresholds.

To add a numeric feature, add an entry under `numeric_columns`:

```yaml
numeric_columns:
  my_new_column:
    aggregations: ["sum", "max"]
    log1p: true
    percentile_group: [["product_type", "year"], ["product_type"]]
```

It will flow into the aggregate block automatically. To also carry it at each
sequence position, add it to `sequence.carry_columns` (and to
`sequence.carry_percentiles` if you want the ranked version too).

To change the product rollup, edit `product_families.map`. Unlisted types fall
into `default_family`.

## How it works

```
load  ->  split outcome rows  ->  pick anchor  ->  build features  ->  attach labels
```

Steps 4 and 5 are independent, so the same feature builder serves both
datasets. The two designs differ only in the anchor rule and the label shape.

**Anchor rules.** Converters anchor at their last acquisition before the
outcome, in both designs. Non-converters differ:

- *fixed*: anchor at the last acquisition at least `horizon_months` before the
  extract date, so both classes get the same time at risk. Converters whose gap
  exceeds the horizon are **excluded, never relabelled 0**.
- *hazard*: anchor at the last acquisition. Censoring is handled by the interval
  expansion, so no horizon is needed and nobody is dropped for timing.

## Two rules that must not be relaxed

1. **Outcome rows are removed before any aggregation.** They carry an id, a date
   and a target flag; every other column is null. Any groupby that still sees
   them will silently corrupt counts and last-row lookups.
2. **No feature may use a row dated after the anchor**, and no feature may
   encode the distance from the anchor to the outcome or to the extract date.
   Both leak the label. Gaps are measured *between* acquisitions only.

## Things that will bite you

- **Percentile grids are fit once and saved.** Recomputing at scoring time makes
  a customer's feature value depend on whoever else is in that day's batch.
- **`min_cell_size` controls the fallback.** A grouping level is used only if its
  cell held at least that many rows at fit time; otherwise the next level in
  `percentile_group` is tried. If every level is too thin the value comes back
  null, which is the intended behaviour, not a failure.
- **Hazard row count is not sample size.** 250k customers produce millions of
  rows and still only a few hundred events. The event count governs how many
  features the data supports. Cluster standard errors by customer and split
  cross-validation folds by customer, never by row.
- **The final partial interval** is dropped when exposure falls below
  `min_exposure_frac` — but never when it contains the event.
- **Customers who keep buying after the anchor** still have those rows in the
  source file. The feature builder filters them out. Do not replace that filter
  with a plain "last N rows per customer".
- **`agg_tenure_months`** needs full history. Drop it from the config if the
  scoring payload cannot supply a first-acquisition date, or training and
  serving will disagree.

## Performance

`preprocess.py` takes ~120s to write both datasets for 25k synthetic customers.
Scale roughly linearly: expect **20-25 minutes** at 250k customers, not the
couple of minutes you might assume.

The per-customer logic runs as an explicit `groupby` loop. Slower than a
vectorised version and much easier to read and modify, which is the right trade
for a batch job. `build_features` is the only hot spot.
