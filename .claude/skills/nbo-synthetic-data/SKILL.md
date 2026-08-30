---
name: nbo-synthetic-data
description: Purpose, limits, and editing rules for make_synthetic.py, the generator that writes a raw-schema file for smoke-testing the cross-sell pipeline end to end. Use when running a smoke test, editing make_synthetic.py, adding a product type, or interpreting metrics that came from a synthetic run.
---

# Synthetic data generation rules

Scope: `nbo_data_modeling/make_synthetic.py` (it lives beside the modeling
code even though its output feeds preprocessing).

## Purpose and limits

- Writes a raw file in the same schema as the real extract, for smoke-testing
  `preprocess.py` and `model.py` end to end at realistic volume.
- Signal is injected **deliberately** (larger deposits and savings/time
  products raise conversion odds) so metrics come out non-degenerate on a
  test run.
- **It is not a simulation of the real population and tells you nothing about
  achievable lift.** Never cite its metrics as evidence of real-world model
  performance — use it only to confirm a config or code change still runs.

## Usage

```bash
cd nbo_data_modeling && python make_synthetic.py --customers 25000 --out synthetic.csv
```

Flags: `--customers` (default 25000), `--out` (default `synthetic.csv`),
`--seed` (default 7, reproducibility), `--base-rate` (default 0.02, the target
conversion rate baked into the injected signal).

Point `io.input_path` in `nbo_data_preprocessing/config.yaml` at the file this
writes before running preprocessing against it.

## When editing this script

- Keep the injected wealth-first / early-outcome edge case (a small fraction
  of customers get an outcome row dated before their first acquisition) — it
  exists specifically to exercise the eligibility filter in
  `preprocess.py`, don't remove it as "unrealistic."
- If you add a new product type, add it to the `TYPES` table here *and* to
  `config.yaml`'s `product_families.map`, and extend the `WEIGHTS` array to
  match the new length — these are not auto-synced, and a mismatched
  `WEIGHTS` length raises at `rng.choice`.
