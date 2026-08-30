# wealth_nbo

Cross-sell propensity modelling: find non-wealth bank customers likely to buy a
wealth product, from the sequence of products they have already acquired and the
dollars tied to each one.

## Layout

```
nbo_data_preprocessing/   preprocess.py + config.yaml        raw rows -> modelling datasets
nbo_data_modeling/        model.py + model_config.yaml       fit + evaluate
                          make_synthetic.py                  synthetic raw file (smoke test)
nbo_report/               reporting layer (charts, HTML, sequences, profiling, targeting)
report_config.yaml        every knob the reporting layer has
eda.py                    entry point -> eda_report.html
main.py                   entry point -> model_report.html + next_best_customers.csv
e2e_walkthrough.ipynb     interactive component-by-component tour of the whole chain
```

Each stage directory has its own README covering that stage. `CLAUDE.md` carries
the invariants — read it before changing anything about anchors, leakage or
splits.

## Running it

Both entry points do the whole chain, gated by `steps:` in `report_config.yaml`:

```bash
python eda.py      # synthetic -> preprocess -> eda_report.html
python main.py     # synthetic -> preprocess -> model -> model_report.html + CSV
```

Add `--no-run` to skip every pipeline stage and re-render the reports from
whatever is already on disk — the fast loop while iterating on a report.
`--config other.yaml` points the whole thing at a different setup.

**`e2e_walkthrough.ipynb`** is the interactive version of the same chain: every
component's inputs and outputs on screen, one traced customer followed from raw
rows to reason codes. It defaults to synthetic data; set `USER_DATA_PATH` in
its first cell to run it on your own extract (which must match the schema your
preprocessing config describes). It writes only under `outputs/walkthrough/`,
so it never disturbs a real run.

The stage scripts still run standalone, exactly as before:

```bash
cd nbo_data_modeling      && python make_synthetic.py --customers 25000 --out ../outputs/synthetic.csv
cd nbo_data_preprocessing && python preprocess.py --config config.yaml
cd nbo_data_modeling      && python model.py      --config model_config.yaml
```

Everything lands under `outputs/` (gitignored):

```
outputs/synthetic.csv               raw extract
outputs/model_data/                 flat_dataset.csv, hazard_dataset.csv, exclusions.csv
outputs/artifacts/                  percentile_grids.json
outputs/model_output/               metrics.csv, predictions.csv, features.csv, model.joblib
outputs/reports/                    eda_report.html, model_report.html,
                                    next_best_customers.csv, segment_profiles.csv
```

### The path rule

A relative path inside a config file resolves against **the directory holding
that config file**. That is why the stage configs say `../outputs/...` and
`report_config.yaml` says `outputs/...` for the same place, and it is what lets
a config live outside the repo entirely.

## The two reports

**`eda_report.html`** — before any model exists, for a marketing audience.
Headline volumes, the raw file's quality and dynamics, n-order Markov profiling
of the event runs that precede the target event, the raw→engineered lineage
table, per-feature distributions and Information Value, and who was excluded
from each design and why. The insights block at the top is derived from the
run's own numbers, not written by hand.

**`model_report.html`** — model risk notes first, then performance (PR-AUC and
lift foregrounded, ROC-AUC with its low-prevalence caveat), coefficient
stability, the score distribution and decile/gains tables, k-means segments over
the high-propensity population, the call list with per-customer reason codes,
and the same sequence profiling restricted to the customers the model likes.

Both are single self-contained files — charts are base64 PNGs, CSS is inline, no
network needed — and every chart has the table of numbers beside it.

### Two scores, kept apart

The campaign list is **ranked on out-of-fold scores**, so no customer is ranked
by a model that trained on them. `outputs/model_output/model.joblib` is a
separate model refit on everything, for scoring customers who were not in this
extract. Ranking today's population with the refit model looks better and means
less.

## Pointing it at a different dataset

No column name appears in `eda.py`, `main.py` or anything under `nbo_report/`.
To move to another dataset with the same shape (one row per event per entity,
plus outcome rows), edit config only:

- `nbo_data_preprocessing/config.yaml` — `columns:` roles, `product_families:`
  rollup, `numeric_columns:`, the sequence window and the labelling thresholds.
- `report_config.yaml` — `schema:` (the structural columns preprocess.py
  writes), `labels:` (report titles and the vocabulary used in generated prose,
  e.g. converter/conversion → churner/churn), `sequences.event_column` for the
  event alphabet, and the profiling/targeting knobs.

This has been exercised end to end against an unrelated telco-churn extract with
entirely different column names; the reports pick up the new columns, families
and vocabulary with no code change.

## Environment

No `requirements.txt` or lockfile. Verified on Python 3.10.7 with pandas 2.1.4,
numpy 1.26.4, pyyaml 6.0.2, scikit-learn 1.7.2, matplotlib 3.10.9, joblib 1.5.2,
scipy 1.15.3. Preprocessing needs only pandas/numpy/pyyaml; `model.py` adds
scikit-learn; the reporting layer adds matplotlib and joblib.

Expect roughly 2 minutes of preprocessing and 2 minutes of cross-validation at
25k customers, scaling roughly linearly.
