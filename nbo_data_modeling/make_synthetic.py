"""
Generate a synthetic raw file in the same schema as the real extract.

For smoke-testing the pipeline at realistic volume. Signal is injected
deliberately (large deposits and savings/time products raise conversion
odds) so the metrics come out non-degenerate. It is NOT a simulation of
the real population and tells you nothing about achievable lift.

    python make_synthetic.py --customers 25000 --out synthetic.csv
"""

import argparse

import numpy as np
import pandas as pd

TYPES = {
    "checking_basic": ("TXN_DEPOSIT", "dep", 200, 0.6),
    "checking_premium": ("TXN_DEPOSIT", "dep", 2000, 0.8),
    "savings_standard": ("SAV_TIME", "dep", 8000, 1.4),
    "cd_standard": ("SAV_TIME", "dep", 30000, 1.6),
    "money_market": ("SAV_TIME", "dep", 90000, 1.8),
    "jumbo_cd": ("SAV_TIME", "dep", 600000, 2.0),
    "credit_card_rewards": ("CARD", "card", 12000, 0.7),
    "credit_card_secured": ("CARD", "card", 4000, 0.4),
    "auto_loan": ("SEC_INSTALL", "loan", 25000, 0.8),
    "mortgage_fixed": ("MORTGAGE", "loan", 280000, 1.1),
    "heloc": ("MORTGAGE", "loan", 70000, 1.3),
}
NAMES = list(TYPES)
WEIGHTS = np.array([0.20, 0.06, 0.18, 0.08, 0.05, 0.01,
                    0.11, 0.03, 0.12, 0.11, 0.05])
WEIGHTS = WEIGHTS / WEIGHTS.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=25000)
    ap.add_argument("--out", default="synthetic.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base-rate", type=float, default=0.02)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start, end = pd.Timestamp("2012-01-01"), pd.Timestamp("2026-06-30")
    span_days = (end - start).days
    rows = []

    for i in range(args.customers):
        cid = "C%06d" % i
        n_acq = int(np.clip(rng.poisson(3.5) + 1, 1, 12))
        first = start + pd.Timedelta(days=int(rng.uniform(0, span_days * 0.7)))

        dates, chosen, score = [], [], 0.0
        current = first
        for _ in range(n_acq):
            name = rng.choice(NAMES, p=WEIGHTS)
            family, kind, scale, weight = TYPES[name]
            amount = float(np.round(scale * np.exp(rng.normal(0, 0.7)), 2))
            dates.append(current)
            chosen.append((name, family, kind, amount))
            score += weight * np.log1p(amount) / 12.0
            current = current + pd.Timedelta(days=int(rng.uniform(60, 900)))
            if current > end:
                break

        for date, (name, family, kind, amount) in zip(dates, chosen):
            rows.append({
                "customer_id": cid,
                "origination_date": date.date(),
                "product_type": name,
                "deposit_balance": amount if kind == "dep" else None,
                "loan_balance": amount if kind == "loan" else (0.0 if kind == "card" else None),
                "credit_exposure": amount if kind == "card" else None,
                "rate": round(float(rng.uniform(0.0001, 0.22)), 4),
                "account_status": "OPEN" if rng.random() > 0.15 else "CLOSED",
                "target": None,
            })

        # Conversion hazard rises with the score. A wealth-first customer
        # is injected occasionally so the eligibility filter gets tested.
        prob = 1.0 / (1.0 + np.exp(-(np.log(args.base_rate / (1 - args.base_rate))
                                     + 1.6 * (score - 2.2))))
        if rng.random() < prob:
            anchor = dates[-1]
            gap = int(rng.gamma(2.2, 9.0)) + 1
            when = anchor + pd.DateOffset(months=gap)
            if when <= end:
                rows.append({"customer_id": cid, "origination_date": when.date(),
                             "product_type": None, "deposit_balance": None,
                             "loan_balance": None, "credit_exposure": None,
                             "rate": None, "account_status": None, "target": True})
        elif rng.random() < 0.01:
            rows.append({"customer_id": cid,
                         "origination_date": (first - pd.Timedelta(days=200)).date(),
                         "product_type": None, "deposit_balance": None,
                         "loan_balance": None, "credit_exposure": None,
                         "rate": None, "account_status": None, "target": True})

    df = pd.DataFrame(rows).sort_values(["customer_id", "origination_date"],
                                        kind="mergesort")
    df.to_csv(args.out, index=False)
    n_conv = df["target"].notna().sum()
    print("%d rows, %d customers, %d outcome rows (%.2f%%)"
          % (len(df), df.customer_id.nunique(), n_conv,
             100.0 * n_conv / df.customer_id.nunique()))


if __name__ == "__main__":
    main()
