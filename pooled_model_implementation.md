# Pooled cross-product model: what changes and how to build it

Companion to `result_ranking_approaches.md`. This describes the "upstream" option: replace the
seven per-product XGBoost models plus the association-rule / look-alike score combination with
**one supervised model trained on all seven products at once**, so there is no combination step
left to defend.

Written against the code in `wealth_nbo/` (preprocess.py, model.py, adhoc_utils/). Where the real
pipeline differs from what the repo shows, treat the file names as placeholders and the logic as the
point.

---

## 0. One-paragraph summary of the change

Today: for each product p, a model is fit on the customers who bought p versus everyone else, then
three per-product scores are combined into a rank. Tomorrow: build one table with **one row per
(customer, product) pair**, label = 1 if that customer bought that product, product as a single
categorical feature, and fit **one** model. Each customer gets seven scores (one per row). Rank
within product exactly as today. The association-rule and look-alike signals either disappear from
scoring or come back in as ordinary feature columns of that one model.

---

## 1. What the source data looks like

### 1a. Raw extract (input to `preprocess.py`)

One row per **product acquisition** per customer, plus one **outcome row** per wealth purchase.
Columns the pipeline relies on (names from `nbo_data_preprocessing/config.yaml`):

| column | on acquisition rows | on outcome rows |
|---|---|---|
| `customer_id` | set | set |
| `origination_date` | date the product was opened | date the wealth product was bought |
| `product_type` | checking_basic, savings_standard, auto_loan, mortgage_fixed, ... | **the wealth product bought (A..G)** |
| `target` | null | non-null (today: 1) |
| `deposit_balance`, `loan_balance`, `credit_exposure` | set where applicable | null |

Example (one customer, two prior products, one wealth purchase):

```
customer_id, origination_date, product_type,     target, deposit_balance, loan_balance
10001,       2019-03-14,       checking_basic,    ,       4200,
10001,       2021-07-02,       auto_loan,         ,       ,                18500
10001,       2023-11-20,       product_C,         1,      ,
```

**The one thing to check first.** Today `target` is a single flag meaning "bought a wealth
product." For the pooled model the outcome row must say *which* product. If `product_type` on
outcome rows already holds A..G, nothing changes. If it does not, the extract needs one extra column
(or `target` becomes the product code). Without this, nothing below is possible.

Scale: ~500k customers, roughly 2-6 acquisition rows each, ~419 outcome rows across the seven
products (160 / 110 / 60 / 40 / 20 / 18 / 11).

### 1b. Current modelling table (`flat_dataset.csv`, output of `preprocess.py`)

One row per **eligible customer**, anchored at a date, with features computed only from rows dated
on or before the anchor:

| block | examples | width |
|---|---|---|
| id / bookkeeping | `customer`, `anchor_date` | dropped before fitting |
| sequence (last 4 acquisitions) | `pos1_product_type`, `pos1_product_family`, `pos1_deposit_balance_pct`, `gap_1_2`, ... | ~25 |
| aggregates over full history | `deposit_balance_sum`, `deposit_balance_max_log1p`, `count_TXN_DEPOSIT`, `count_CARD`, `tenure_months` | ~15 |
| label | `label` = 1 if a wealth product was bought within `horizon_months` of the anchor | 1 |

This table is what the per-product XGBoost models are trained on today (one copy per product, with
`label` meaning "bought product p").

### 1c. The other two inputs (association rules, look-alikes)

Both consume a **one-row-per-customer** file with the same features plus a binary target for one
product. Association rules run on a matched case-control sample; look-alikes run on the full file.
Each produces a per-customer score for that product. These two files are *derived* from 1b, not
separate sources.

---

## 2. What the new training table looks like

**Long format: one row per (customer, product).** Same feature columns as 1b, plus two new columns.

```
customer, product, <all feature columns from flat_dataset>, holds_product, label
10001,    A,       ...,                                      0,             0
10001,    B,       ...,                                      0,             0
10001,    C,       ...,                                      0,             1
10001,    D,       ...,                                      0,             0
10001,    E,       ...,                                      0,             0
10001,    F,       ...,                                      0,             0
10001,    G,       ...,                                      0,             0
10002,    A,       ...,                                      1,             0   <- already holds A: excluded from A's lead list
...
```

| column | meaning |
|---|---|
| `product` | categorical, 7 levels. The only "new" model feature. |
| `holds_product` | 1 if the customer already held this product at the anchor. Used to exclude, not to predict. |
| `label` | 1 if the customer bought *this* product within the horizon after the anchor. |

Sizes: 500k customers x 7 = 3.5M rows, ~40-45 columns, ~419 positives. The **feature column count
barely moves**; only the row count multiplies. Multi-label is automatic: a customer who bought C and
later F has `label = 1` on two of their seven rows.

---

## 3. Step-by-step implementation

### Step 1. Confirm the outcome rows identify the product (data check, no code)

```python
raw = pd.read_csv(RAW)
out = raw[raw["target"].notna()]
print(out["product_type"].value_counts())      # must show A..G with counts ~160, 110, 60, 40, 20, 18, 11
print(out.groupby("customer_id").size().value_counts())   # how many customers bought >1 product
```

If `product_type` is blank on outcome rows, stop and fix the extract.

### Step 2. Change `preprocess.py`'s label step to emit one label per product

Everything up to "attach labels" stays as it is. Today the label step writes one `label` column.
Change it to write seven, then melt.

```python
PRODUCTS = ["A", "B", "C", "D", "E", "F", "G"]

# outcome rows, already split out earlier in the script
outcomes = outcome_rows[["customer_id", "origination_date", "product_type"]]

def label_for(features, product):
    """1 if the customer bought `product` within horizon_months after their anchor."""
    bought = outcomes[outcomes["product_type"] == product]
    m = features[["customer", "anchor_date"]].merge(bought, left_on="customer",
                                                    right_on="customer_id", how="left")
    months = (m["origination_date"] - m["anchor_date"]).dt.days / 30.44
    return ((months > 0) & (months <= HORIZON_MONTHS)).astype(int).values

wide = features.copy()
for p in PRODUCTS:
    wide[f"label_{p}"] = label_for(features, p)
    wide[f"holds_{p}"]  = held_at_anchor(features, p)   # 1 if p appears in acquisition rows dated <= anchor

long = wide.melt(id_vars=[c for c in wide.columns if not c.startswith(("label_", "holds_"))],
                 value_vars=[f"label_{p}" for p in PRODUCTS],
                 var_name="product", value_name="label")
long["product"] = long["product"].str.replace("label_", "")
long["holds_product"] = [wide.loc[i, f"holds_{p}"] for i, p in zip(long.index % len(wide), long["product"])]
long.to_csv(OUT / "long_dataset.csv", index=False)
```

Keep the existing anchor rule unchanged (one anchor per customer). Two consequences to write down
for MRM, not to engineer around yet:

- A customer who bought C in 2023 and F in 2025 has one anchor (before C). F's label is 1 only if F
  fell inside the horizon from that anchor. This is the simplest consistent rule.
- `holds_*` must only look at acquisition rows dated on or before the anchor. Looking later leaks.

### Step 3. Point `model.py` at the long table (config only, plus one line)

In `model_config.yaml`:

```yaml
io:
  input_path: "../outputs/model_data/long_dataset.csv"
design: "flat"
columns:
  id:     "customer"
  group:  "customer"            # already the rule: folds split on customer, never row
  label:  "label"
  weight: null
  drop:   ["anchor_date", "holds_product"]
  force_categorical: ["product"]  # <- the one new feature
model:
  type: "gbm"
sampling:
  negative_customer_fraction: 0.25   # samples whole customers, so all 7 rows go together
```

Nothing in the fitting code needs to know about products. `StratifiedGroupKFold` on `customer`
already keeps a customer's seven rows in the same fold, which is the leakage rule here.

The one code change: **report lift per product, not overall.** After out-of-fold predictions exist:

```python
pred = pd.read_csv("predictions.csv")            # customer, product, score, label
for p, g in pred.groupby("product"):
    g = g.sort_values("score", ascending=False)
    base = g["label"].mean()
    for frac in (0.01, 0.05, 0.10):
        top = g.head(int(len(g) * frac))
        print(p, frac, "lift =", top["label"].mean() / base, "n_pos =", top["label"].sum())
```

### Step 4. Build the lead lists

```python
pred = pred[pred["holds_product"] == 0]                      # never recommend what they already own
pred["rank"] = pred.groupby("product")["score"].rank(ascending=False, method="first")
leads = pred.sort_values(["product", "rank"])
```

Seven ranked lists, one per product, from one model. There is no weighted rank and no 3-tier sort.

### Step 5. Prove it beat the old approach (this is the MRM deliverable)

On the same per-product holdout used today, compute lift@N for:

1. the current per-product XGBoost alone
2. the current 3-tier sort
3. the pooled model

with bootstrap confidence intervals (resample customers, 500 draws). The decision rule, product by
product: adopt the pooled model where it is at least as good as (2) within the interval, and
it is expected to be clearly better for D-G because those products now share the structure learned
from A-C. If it is worse for A or B, that says product-specific interactions matter and the fix is
more tree depth or a few explicit `product x feature` columns, not going back to seven models.

Also run the stability check already in the repo (rank overlap of the top-N across seeds and
repeats). Pooled fits are more stable than 11-positive fits; show it.

### Step 6 (optional, phase 2). Fold the other two signals in as features

Only after Step 5 is done and the pooled model is accepted. Add two or three columns to the long
table, each computed **inside training folds only**:

| new feature | from | how |
|---|---|---|
| `rule_hits` | association rules | number of product-p rules (mined on the training fold) the customer's row satisfies |
| `rule_max_lift` | association rules | max lift among the rules it satisfies, 0 if none |
| `lookalike_dist` | look-alikes | Gower distance to the k nearest training-fold buyers of product p |

These are per-(customer, product) numbers, so they slot straight into the long table. The model
then decides how much they matter, which is the "data-estimated weights" MRM said it accepts, with
no separate stacker. If feature importance for them is ~0, drop them and retire the two utilities
from scoring (they stay useful as explanation tools for reps).

---

## 4. What gets retired, what stays

| today | after |
|---|---|
| 7 per-product XGBoost fits | 1 pooled fit |
| `Lead_Rank = .6 s1 + .3 s2 + .1 s3` | gone |
| 3-tier lexicographic sort | gone (keep as a documented fallback) |
| association rules as a *score* | optional feature (Step 6), otherwise explanation only |
| look-alikes as a *score* | optional feature (Step 6), otherwise explanation / rep map only |
| per-product lead CSVs | unchanged format, produced by Step 4 |

---

## 5. Pitfalls, in the order you will hit them

1. **Outcome rows without a product code.** Step 1. Nothing works without it.
2. **Splitting folds by row.** The repo already groups by customer. Do not change `columns.group`.
3. **Features that peek past the anchor.** `holds_product` and any cross-ownership feature must use
   acquisition rows dated on or before the anchor only. Buying B after the anchor is a label for B,
   not a feature for A.
4. **The product indicator eating the model.** With 160 vs 11 positives the first split will be on
   `product`. That is fine (it is the base rate), but keep `min_samples_leaf` high (200+) and depth
   low (3-4) so later splits still have to generalise across products.
5. **Reading overall metrics.** Overall PR-AUC on the long table is dominated by product A. Only the
   per-product lift table in Step 3 means anything.
6. **Probabilities.** They are not calibrated per product and should not be shown as conversion
   probabilities. Report rank, decile, and lift, as the repo already does.
7. **Negative downsampling.** Keep `negative_customer_fraction` at 1.0 for the numbers you report.
   0.25 is for iteration speed only.

---

## 6. If MRM wants it in statistical form

The same long table fits a **hierarchical (partially pooled) logistic regression**: global
coefficients plus product-level deviations shrunk toward zero. Set `model.type: logistic` in the
same config, add `product` one-hot plus a handful of `product x feature` interactions with a strong
L2 penalty. That is the textbook answer to "many small groups," is fully transparent, and produces
exactly the same seven ranked lists. Use gradient boosting to find out how much signal there is,
and the penalised logistic as the version you take to review if explainability wins.
