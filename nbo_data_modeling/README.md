# Model evaluation

Fits and evaluates a cross-sell model on the datasets written by
[`../nbo_data_preprocessing/README.md`](../nbo_data_preprocessing/README.md).

```
python model.py --config model_config.yaml
```

Dependencies: `pandas`, `numpy`, `pyyaml`, `scikit-learn`.

Reads whichever file `preprocess.py` wrote. Flip `design` between `flat` and
`hazard`; nothing else needs to change. As with the preprocessing config, the
`io:` paths ship pointing at `/mnt/user-data/outputs/` and need repointing
before the first run.

The label column differs by design — flat writes `label`, hazard writes
`event` — so `columns.label` and `columns.weight` (`exposure_frac` on hazard,
null on flat) move together with the `design` switch.

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
  The config ships at `0.25`; set the fraction to 1.0 for the final numbers you
  report.
- **`features.csv` sign consistency is the stability check.** A logistic feature
  whose coefficient flips sign across folds is noise, not a finding. Treat
  anything below ~0.7 consistency as unproven.

## Smoke testing

`make_synthetic.py` writes a file in the same schema as the real extract, with
signal injected deliberately so the metrics come out non-degenerate:

```
python make_synthetic.py --customers 25000 --out synthetic.csv
```

It lives here rather than beside the preprocessing code, but its output is the
*input* to `preprocess.py` — point `io.input_path` in
`../nbo_data_preprocessing/config.yaml` at it. Other flags: `--seed` (default 7)
and `--base-rate` (default 0.02).

It is not a simulation of the real population and says nothing about achievable
lift. Use it to check that a config change still runs end to end.

## Performance

Measured on 25k synthetic customers:

| Step | Time |
|---|---|
| one `saga` logistic fit, 52k hazard rows | ~8s |
| full CV, 2 repeats x 5 folds | ~2min |

`saga` is the only sklearn solver that supports elastic net and it is slow on
wide hazard files. If fits drag, lower `logistic.max_iter` or reduce
`sampling.negative_customer_fraction` before reaching for a different model. A
convergence warning at low `max_iter` is expected and does not materially move
the ranking, but do not report final coefficients from a non-converged fit.
