# Cross-sell preprocessing

Turns raw product-acquisition rows into two modelling datasets.

```
python preprocess.py --config config.yaml
```

Dependencies: `pandas`, `numpy`, `pyyaml`. Nothing else.

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

---

# Model evaluation

```
python model.py --config model_config.yaml
```

Reads whichever file `preprocess.py` wrote. Flip `design` between `flat` and
`hazard`; nothing else needs to change.

| File | Contents |
|---|---|
| `model_output/metrics.csv` | headline metrics, mean and spread across repeats |
| `model_output/metrics_by_repeat.csv` | the same, per repeat |
| `model_output/calibration.csv` | predicted vs observed rate by score bin |
| `model_output/features.csv` | coefficients + sign consistency, or GBM importances |
| `model_output/predictions.csv` | out-of-fold score per customer |

## How the two designs are compared

Under `flat` the model scores customers directly. Under `hazard` it predicts a
per-interval hazard, which the script combines into

```
P(convert by H) = 1 - product over k of (1 - h_k)
```

so both designs end up scored on the same customer-level footing and the numbers
in `metrics.csv` are directly comparable. Customers censored before the horizon
are dropped from evaluation, because their outcome is genuinely unknown.

## Things that are easy to get wrong

- **Folds split on the customer, never on the row.** In the hazard design a
  customer owns many rows; splitting by row leaks them across the boundary.
  `StratifiedGroupKFold` with `columns.group` handles this.
- **ROC-AUC is close to useless below 1% prevalence.** Read PR-AUC and lift at
  the top of the ranking. ROC-AUC is written out for reference only.
- **Row count is not sample size.** Millions of hazard rows still carry only a
  few hundred events, and the event count is what limits how many features the
  data supports.
- **No oversampling or SMOTE.** Class weights, then check `calibration.csv`.
- **`sampling.negative_customer_fraction` samples whole customers, never rows.**
  Dropping some of a customer's intervals would corrupt the survival product.
  Retained negatives are weighted by `1/fraction`. Ranking is unaffected either
  way; the weight is what keeps predicted probabilities near the true base rate.
- **Lift is measured against the prevalence of the sampled data.** If you
  downsample negatives, lift figures are not comparable to the full population.
  Set the fraction to 1.0 for the final numbers you report.
- **`features.csv` sign consistency is the stability check.** A logistic feature
  whose coefficient flips sign across folds is noise, not a finding. Treat
  anything below ~0.7 consistency as unproven.

## Smoke testing

`make_synthetic.py` writes a file in the same schema as the real extract, with
signal injected deliberately so the metrics come out non-degenerate:

```
python make_synthetic.py --customers 25000 --out synthetic.csv
```

It is not a simulation of the real population and says nothing about achievable
lift. Use it to check that a config change still runs end to end.

## Performance

Measured on 25k synthetic customers:

| Step | Time |
|---|---|
| `preprocess.py` (both datasets) | ~120s |
| one `saga` logistic fit, 52k hazard rows | ~8s |
| full CV, 2 repeats x 5 folds | ~2min |

Scale roughly linearly: expect **20-25 minutes** for preprocessing at 250k
customers, not the couple of minutes you might assume.

The per-customer logic in `preprocess.py` runs as an explicit `groupby` loop.
Slower than a vectorised version and much easier to read and modify, which is
the right trade for a batch job. `build_features` is the only hot spot.

`saga` is the only sklearn solver that supports elastic net and it is slow on
wide hazard files. If fits drag, lower `logistic.max_iter` or reduce
`sampling.negative_customer_fraction` before reaching for a different model. A
convergence warning at low `max_iter` is expected and does not materially move
the ranking, but do not report final coefficients from a non-converged fit.
