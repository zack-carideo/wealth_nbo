# Task: implement `eda.py` and `main.py` with HTML reporting

Working plan for the wealth_nbo project. Both entry scripts are currently
comment-only stubs. Read `CLAUDE.md` and the three skills in `.claude/skills/`
before starting — the leakage and customer-level-split invariants are binding.

## Decisions already made (do not re-ask)

| Question | Decision |
|---|---|
| Chart/report stack | **matplotlib static PNGs**, base64-embedded into hand-built HTML. `matplotlib 3.10.9` is already installed. Do **not** add plotly or jinja2 — build HTML with f-strings. |
| Scoring basis for target list | **Refit the final model on all data and persist to `model_output/model.joblib`** (the deliverable "trained model"), but **rank the target list on out-of-fold scores** — unbiased for the current population. |
| Config layout | **One new root-level `report_config.yaml`** for all EDA/reporting knobs. `nbo_data_preprocessing/config.yaml` and `nbo_data_modeling/model_config.yaml` stay authoritative for their stages. |
| Reason codes | **Per-customer top-3 drivers, plus segment-level narrative** (cluster the top-scoring population into named segments with a profile for each). |

Everything must be driven from config so it generalizes to any dataset the
pipeline is pointed at. No hardcoded column names in the new code.

## Stub intent (translated from the comments)

The stubs name `preprocessing.py` / `modeling.py` / `postprocessing.py`, which
do not exist. Real mapping:

- "preprocessing.py" (synthetic gen) → `nbo_data_modeling/make_synthetic.py`
- "preprocessing.py" (feature build) → `nbo_data_preprocessing/preprocess.py`
- "modeling.py" → `nbo_data_modeling/model.py`
- "postprocessing.py" → **new reporting code** (does not exist yet)

`eda.py`
1. Generate synthetic raw data (optional, config-gated).
2. Generic **n-order Markov sequence profiling** — subset to customers with a
   historical target event, surface the event sequences that lead to it. Must
   be general to any event/target, not just cross-sale.
3. Generic profiling of `preprocess.py` output: **3.1** map original variables
   → engineered variables; **3.2** profile each engineered variable's
   distribution and its relationship to the target.

`main.py`
1. Synthetic data → 2. preprocess → 3. model + evaluate → 4. subset to
   high-propensity customers and profile the sequences leading to the target
   event, **reusing the same functions as `eda.py`**.

## Files to create

```
report_config.yaml          all new knobs (paths, markov order, event column,
                            top-N, segment k, output filenames)
nbo_report/
  __init__.py
  config.py                 config loading + sys.path shim to import
                            preprocess.py / model.py across directories
  charts.py                 matplotlib -> base64 <img> helpers (Agg backend)
  html.py                   self-contained HTML assembly: KPI cards, tables,
                            sections, insight callouts, embedded CSS
  sequences.py              generic n-order Markov profiling + transition matrix
  profiling.py              raw->engineered lineage map, distributions,
                            target relationship, Information Value
  targeting.py              refit + persist, reason codes, KMeans segments
eda.py                      entry point -> eda_report.html
main.py                     entry point -> model_report.html + target list CSV
```

Reuse over reimplementation: import `load_data`, `split_outcome_rows`, and
`compute_anchors` from `preprocess.py` rather than duplicating anchor logic —
this keeps the sequence analysis on the correct side of the leakage rule.

## Report contents

**`eda_report.html`** — audience: marketing.
- Headline KPIs: customers, acquisitions, conversion rate, date span.
- Raw data overview: volume by product family, balance distributions, data quality.
- Acquisition dynamics: acquisitions per customer, gaps between acquisitions.
- **Markov section**: top n-grams preceding conversion ranked by lift vs base
  rate, converter vs non-converter transition matrices, minimum support filter.
- Engineered feature profiling: raw→engineered lineage table, distributions,
  event rate by decile, missingness, IV ranking.
- Exclusions breakdown (who was dropped and why, per design).
- Actionable insights block — derived programmatically, not hand-written prose.

**`model_report.html`** — model insights + population profiling + target list.
- Run summary and the config actually used.
- Performance: PR-AUC and lift@K foregrounded, ROC-AUC shown with its
  low-prevalence caveat, calibration curve, spread across repeats.
- Feature insights: coefficients with sign consistency, flagging anything
  below 0.7 as unproven.
- Score profiling over the population: distribution, decile table with observed
  rate, lift, and cumulative capture.
- Segments: named clusters of the high-propensity population with profiles.
- **Next best customers**: top-N with score, decile, top-3 reason codes, and
  segment — plus `next_best_customers.csv` for campaign upload.
- Sequence profile of the high-propensity subset (shared `eda.py` functions).
- Model risk notes: sampling fraction, convergence, sign consistency.

## Blocking issue to fix first

Both configs ship `io:` paths under `/mnt/user-data/outputs/`, a sandbox path
that does not exist on this machine. Repoint them to repo-relative `outputs/`
before the first run. `outputs/` is already covered by `.gitignore`.

## Execution order

1. Repoint the `io:` paths; create `report_config.yaml`.
2. Build `nbo_report/` modules.
3. Build `eda.py`, run it, inspect the HTML.
4. Build `main.py` (full chain + refit/persist + reports), run it.
5. **Have FABLE review the work** (`Agent` tool, `model: fable`) and revise.
6. Full end-to-end run on 25k synthetic customers; verify both HTML reports and
   the CSV. Expect ~120s preprocessing and ~2min CV at that size.

Environment verified: Python 3.10.7, pandas 2.1.4, numpy 1.26.4, pyyaml 6.0.2,
scikit-learn 1.7.2, joblib 1.5.2, scipy 1.15.3, matplotlib 3.10.9.
