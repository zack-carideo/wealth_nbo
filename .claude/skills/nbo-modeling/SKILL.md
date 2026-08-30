---
name: nbo-modeling
description: Rules for fitting and evaluating the cross-sell model — customer-level CV splits, negative downsampling and reweighting, rare-event metric choice, and saga solver behavior. Use when editing nbo_data_modeling/model.py or model_config.yaml, switching between the flat and hazard designs, or interpreting metrics.csv / features.csv / calibration.csv.
---

# Modeling / evaluation rules

Scope: `nbo_data_modeling/model.py` + `nbo_data_modeling/model_config.yaml`.

## Design switch

Input is whatever `preprocess.py` wrote. `design: flat` or `design: hazard`
in `model_config.yaml` controls everything else — no code change needed to
switch.

- `flat`: one row per customer, direct binary classification.
- `hazard`: one row per customer-interval; the model predicts a per-interval
  hazard. Those are combined into a customer-level score via
  `P(convert by H) = 1 - prod(1 - h_k)` so both designs are evaluated on the
  same customer-level footing and are directly comparable.
- Customers censored before the horizon are dropped from evaluation — their
  outcome is genuinely unknown, not a negative.

The label column differs by design: flat writes `label`, hazard writes
`event`. Set `columns.label` accordingly, along with `columns.weight`
(`exposure_frac` for hazard, null for flat).

## Splitting and sampling — customer only, never row

- CV folds split on the customer (`columns.group`), via
  `StratifiedGroupKFold`. In the hazard design a customer owns many rows;
  splitting by row leaks them across the fold boundary.
- `sampling.negative_customer_fraction` downsamples whole non-converting
  customers, never individual rows — dropping some of a customer's intervals
  would corrupt the survival product. Retained negatives are reweighted by
  `1/fraction` so totals still reflect the full population; ranking is
  unaffected, the weight just keeps predicted probabilities near the true
  base rate.
- Lift figures are only comparable across runs at the same sampling fraction.
  The config ships at `0.25`; set `negative_customer_fraction: 1.0` before
  reporting final numbers.

## Metrics

- **ROC-AUC is close to useless below ~1% prevalence.** Read PR-AUC and lift
  at the top of the ranking instead; ROC-AUC is written for reference only.
- Row count is not sample size — millions of hazard rows can still carry only
  a few hundred events, and the event count is what limits how many features
  the data can support.
- No oversampling or SMOTE anywhere in this pipeline — class weights
  (`class_weight="balanced"`) plus post-hoc calibration (`calibration.csv`).
- `features.csv` sign-consistency (logistic only) is the stability check: a
  coefficient that flips sign across folds is noise, not a finding. Treat
  anything below ~0.7 consistency as unproven.

## Solver / performance notes

- `saga` is the only sklearn solver supporting elastic net
  (`logistic.l1_ratio`), and it is slow on wide hazard files. If fits drag,
  lower `logistic.max_iter` or reduce `sampling.negative_customer_fraction`
  before switching models.
- A convergence warning at low `max_iter` is expected and doesn't materially
  move the ranking — but don't report final coefficients from a
  non-converged fit.
- Measured on 25k synthetic customers: one saga logistic fit over 52k hazard
  rows ~8s; full CV at 2 repeats x 5 folds ~2min.
