# Lookalike prospecting for rare targets

Finds the customers who most resemble the (very few) customers who already bought a product,
and tells a sales rep *why* and *how much more likely* they are to buy. No supervised model is fit,
so it works with a handful of buyers. Everything is driven by `config.yaml`.

```
pip install pandas numpy scipy scikit-learn pyyaml plotly      # umap-learn optional
python make_synthetic.py data/customers.csv 200000 80          # optional smoke-test data
python lookalike.py config.yaml
```

## Inputs

One row per customer: aggregated metrics + demographics, any mix of numeric / categorical /
boolean / date columns, missing values allowed. A binary target column = 1 for existing holders
of the product. Optional: a rep/book column (filter for reps), an eligibility flag.

## Outputs (`output.dir`)

| file | what it is |
|---|---|
| `leads.csv` | ranked prospects: score percentile, band (A = top 1% …), archetype, nearest existing buyer, plain-language reasons, historical lift of that band |
| `buyer_profile.csv` | the features that separate buyers from everyone else, with buyer-typical vs population-typical values — the "buyer signature" |
| `validation_lift.csv` | leave-one-buyer-out lift by band with bootstrap CIs — the number to quote to sales ("customers in band A bought at ~45x the average rate") |
| `prospect_map.html` | interactive map for reps: stars = buyers, dots = prospects; click a dot for reasons + its most similar buyers; filter by rep; search an id. Self-contained, opens in any browser |
| `run_summary.json` | feature weights, counts, engines used |

## How it works

1. **Represent** — types are inferred per column. Numerics are rank-transformed to [0,1] (whales don't
   dominate distance). Low-cardinality categoricals stay categorical (rare levels pooled). High-cardinality
   columns (zip, branch…) become a leave-one-out, shrunken buyer-rate. Missing indicators are added
   where informative. None of this uses the label except the (heavily shrunk, LOO) rate encoding.
2. **Profile** — for every feature, a separation statistic between buyers and the population
   (KS distance for numerics, max shrunken log-odds over levels for categoricals). The same statistic is
   computed for hundreds of *random* pseudo-buyer sets of the same size; a feature only earns weight by
   how much it beats the 99th percentile of that null. This is what stops 30 buyers and 300 columns from
   producing a metric built on noise.
3. **Archetypes** — buyers are clustered on the surviving features; a split is kept only if it is
   well separated (silhouette gate). Used for labelling and the map, not for scoring.
4. **Score** — two label-free similarity engines, rank-blended:
   * *Weighted Gower kNN*: mean distance to the k nearest buyers, feature-weighted from step 2. Fully
     decomposable, which is where the per-lead reasons come from.
   * *Random-forest leaf density*: a forest learns the joint structure of the data (real vs
     column-shuffled rows — no label), and each leaf's buyer concentration relative to base rate is the
     score. Catches interactions Gower can't.
5. **Validate** — leave-one-buyer-out: for each buyer, re-derive the profile without them, score them
   against the remaining buyers, and record where they rank against a population sample. The share of
   buyers landing in the top x% divided by x% is the historical lift for that band. If it is ~1, the
   method found nothing and the lead list should not ship.
6. **Explain** — each lead's reasons are the features on which it most matches *its own* nearest
   buyers ("card_spend = 4,120 (similar buyers: 3,500 to 6,200)"), not abstract averages.

## Knobs worth knowing

* `profile.null_quantile` — raise toward 0.995 with many columns and very few buyers; lower to 0.95 if
  nothing survives. `min_features` guarantees the pipeline always runs.
* `profile.exclude_features` — drop scale proxies (total balance, tenure) if the lead list is just
  "your biggest customers"; the profile will then find what *else* separates buyers.
* `similarity.engines` — `[gower]` alone is fastest and fully explainable; add `rf_density` for
  interaction effects. `blend_weights` sets the mix.
* `validation.population_sample` / `n_permutations` — the two runtime drivers.
* `map.n_prospects` — how many top leads are plotted (default 300); the lead CSV is separate and can be
  larger. `map.show_background` / `map.n_background` control the grey context sample; turn it off for the
  cleanest picture.
* `map.extra_fields` — extra raw columns shown (greyed) in the click-panel profile table alongside the
  indicative features, e.g. fields reps care about that the score doesn't use.
* `output.n_reasons` — how many "why similar" bullets per lead, in both `leads.csv` and the map panel.
* `map.method` — `tsne` (default, sklearn), `umap` if installed, `pca` for speed.

## The click panel

Clicking a prospect shows its band and score percentile, the "why similar" bullets, and a profile table
with one row per trait: the prospect's value, the range across its k most similar buyers, the median and
middle-50% range across all buyers, and the same across all customers. Numerics get a bar showing where
the prospect sits among buyers; categoricals show how many of the similar buyers share the value and what
share of buyers / all customers do. Indicative features come first, `extra_fields` follow in grey.

## Runtime

~2 minutes for 200k rows × 30 columns × 80 buyers on a laptop, most of it in the leave-one-out loop
and the forest. Scales linearly in rows; the Gower distance is computed N × buyers, never N × N.

## Files

`lookalike.py` — pipeline. `prospect_map.py` — the HTML map. `make_synthetic.py` — planted-signal test data.
`config.yaml` — every parameter.
