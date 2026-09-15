"""
Synthetic one-row-per-customer dataset with a planted, rare buyer signal.

    python make_synthetic.py data/customers.csv 200000 80

Buyers are drawn from two archetypes so the pipeline has structure to find:
  type 1: mid-age, high card spend, mortgage, mobile-heavy, ~60% of buyers
  type 2: older, high deposit balance, long tenure, branch-heavy, ~40%
Plenty of pure-noise columns are included so the permutation null has work to do.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int, n_buyers: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(46, 14, n).clip(18, 90)
    tenure = rng.exponential(6, n).clip(0, 40)
    deposits = np.exp(rng.normal(8.5, 1.4, n))
    card_spend = np.exp(rng.normal(6.8, 1.1, n)) * (rng.random(n) > 0.25)
    n_products = rng.poisson(1.8, n).clip(1, 8)
    has_mortgage = (rng.random(n) < 0.22).astype(int)
    channel = rng.choice(["mobile", "online", "branch", "mixed"], n, p=[.4, .25, .2, .15])
    segment = rng.choice(["mass", "affluent", "small_biz", "student"], n, p=[.6, .2, .12, .08])
    state = rng.choice([f"S{i:02d}" for i in range(48)], n)          # high-cardinality, no signal
    logins_90d = rng.poisson(12, n)
    noise_cols = {f"noise_{i}": rng.normal(size=n) for i in range(12)}
    noise_cat = {f"noise_cat_{i}": rng.choice(list("ABCDE"), n) for i in range(4)}
    rep_id = rng.choice([f"rep_{i:03d}" for i in range(60)], n)

    y = np.zeros(n, int)
    b = rng.choice(n, n_buyers, replace=False)
    t1 = rng.random(n_buyers) < 0.6
    b1, b2 = b[t1], b[~t1]
    age[b1] = rng.normal(42, 6, len(b1)); card_spend[b1] = np.exp(rng.normal(8.6, .5, len(b1)))
    has_mortgage[b1] = (rng.random(len(b1)) < .8); channel[b1] = rng.choice(["mobile", "mixed"], len(b1))
    age[b2] = rng.normal(63, 6, len(b2)); deposits[b2] = np.exp(rng.normal(11.2, .6, len(b2)))
    tenure[b2] = rng.normal(18, 5, len(b2)).clip(3, 40); channel[b2] = rng.choice(["branch", "mixed"], len(b2))
    segment[b] = rng.choice(["affluent", "mass"], n_buyers, p=[.7, .3])
    y[b] = 1

    df = pd.DataFrame(dict(customer_id=[f"C{i:07d}" for i in range(n)], rep_id=rep_id, age=age.round(0),
                           tenure_years=tenure.round(1), deposit_balance=deposits.round(0),
                           card_spend_monthly=card_spend.round(0), n_products=n_products,
                           has_mortgage=has_mortgage, primary_channel=channel, segment=segment, state=state,
                           logins_90d=logins_90d, **noise_cols, **noise_cat, bought_product=y))
    df.loc[rng.random(n) < .05, "card_spend_monthly"] = np.nan
    df["email"] = "x@example.com"
    df["open_date"] = "2020-01-01"
    return df


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/customers.csv")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    out.parent.mkdir(parents=True, exist_ok=True)
    make(n, k).to_csv(out, index=False)
    print(f"wrote {out}: {n:,} rows, {k} buyers")
