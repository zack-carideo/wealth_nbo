"""
associations.py — mixed-type DataFrame in, association rules with the target as the only consequent out.

    pip install mlxtend pyyaml
    python associations.py matched.csv                       # the case-control sample, not the full 515k
    python associations.py matched.csv --config other.yaml   # a different config file

Pipeline:  binarize -> mine -> to_target_rules -> dedupe -> validate -> rank_rules
Item columns are opaque ids (i0001, i0002, ...). A `catalog` maps id -> (attribute, human label).
Rules carry antecedents as tuples of ids; the readable string is derived for display only and is
never parsed, so column names and category levels may contain any characters.

CONFIG
------
Every knob lives in the `associations:` block of report_config.yaml at the repo root, beside the
blocks driving eda.py and main.py. Nothing here falls back to a hardcoded tuning value:
load_config() reads the block and checks it against PARAMS, main() splats it into run(), and run()
threads each value down to the function that uses it. So the command line above and

    run(df, **load_config())

are the same run, and every function is callable on its own with explicit arguments — none of them
reaches back up to module state.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from mlxtend.frequent_patterns import fpgrowth

# ------------------------------------------------------------------ CONFIG
# Paths and a block name, not tuning values. Relative to this file the repo
# root -- the directory holding report_config.yaml -- is two levels up.
DEFAULT_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "report_config.yaml"))
DEFAULT_SECTION = "associations"

# The exact set of keys run() takes. Named here so a typo in the yaml fails at
# load time instead of silently falling back to some default.
PARAMS = ("target", "exclude", "n_bins", "max_levels", "max_item_cov", "min_events",
          "max_len", "min_lift", "min_conf_ev", "seed", "top_n", "discover_frac",
          "ci_alpha", "fdr", "max_overlap", "tid")


def load_config(path=None, section=DEFAULT_SECTION):
    """Read the config block and return it as run()'s keyword arguments.

    A missing or unknown key raises rather than being filled in, so the yaml stays the single place
    any of these numbers is written down."""
    path = path or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if section not in cfg:
        raise KeyError("%s has no '%s:' block" % (path, section))
    block = cfg[section] or {}
    missing = [k for k in PARAMS if k not in block]
    unknown = [k for k in block if k not in PARAMS]
    if missing or unknown:
        raise KeyError("%s '%s:' block is wrong%s%s" % (
            path, section,
            "; missing: " + ", ".join(missing) if missing else "",
            "; unknown: " + ", ".join(unknown) if unknown else ""))
    return {k: block[k] for k in PARAMS}


# ------------------------------------------------------------------ 1. binarize
def binarize(df, target, exclude, n_bins, max_levels, tid):
    """Returns (items, catalog). items: bool DataFrame with opaque column ids plus the target item
    `tid`. catalog: {id: (attribute, label)}. Numeric -> quantile bins, categorical -> one-hot
    (rare -> OTHER), 0/1 & bool -> as is, NaN -> own item."""
    cols, catalog = {}, {}
    def add(attr, label, mask):
        iid = f"i{len(catalog) + 1:04d}"
        catalog[iid] = (attr, label)
        cols[iid] = mask.astype(bool)
    for c in df.columns:
        if c in exclude or c == target:
            continue
        s = df[c]
        if s.dropna().isin([0, 1, True, False]).all():
            add(c, f"{c} = 1", s.fillna(0))
        elif pd.api.types.is_numeric_dtype(s) and s.nunique() > n_bins:
            b = pd.qcut(s.rank(method="first"), n_bins, labels=False, duplicates="drop")
            for k, (lo, hi) in s.groupby(b).agg(["min", "max"]).iterrows():
                add(c, f"{c} in [{lo:.3g}, {hi:.3g}]", b == k)
        else:
            s = s.astype("object")
            keep = s.value_counts().index[: max_levels - 1]
            s = s.where(s.isin(keep) | s.isna(), "OTHER")
            for lvl in s.dropna().unique():
                add(c, f"{c} = {lvl!r}", s == lvl)
        if s.isna().any():
            add(c, f"{c} is NA", s.isna())
    items = pd.DataFrame(cols, index=df.index)
    items[tid] = df[target].astype(bool)
    return items, catalog


def label(ante, catalog):
    return " AND ".join(catalog[i][1] for i in ante)


# ------------------------------------------------------------------ 2. mine
def mine(items, min_events, max_len, max_item_cov, tid):
    keep = [c for c in items.columns if c == tid or items[c].mean() <= max_item_cov]
    fi = fpgrowth(items[keep], min_support=min_events / len(items), use_colnames=True, max_len=max_len)
    return fi[fi["itemsets"].apply(lambda s: tid in s)]


# ------------------------------------------------------------------ 3. rules, target as sole consequent
def _poisson_ci(k, alpha):
    lo = stats.chi2.ppf(alpha / 2, 2 * k) / 2 if k > 0 else 0.0
    return lo, stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2

def _cover(items, ante, tid):
    m = items[list(ante)].all(axis=1).values
    return m, int(m.sum()), int(items[tid].values[m].sum())

def to_target_rules(freq, items, catalog, min_lift, ci_alpha, tid):
    base = items[tid].mean(); rows = []
    for s in freq["itemsets"]:
        ante = tuple(sorted(s - {tid}))
        if not ante:
            continue
        _, n, k = _cover(items, ante, tid)
        lo, hi = _poisson_ci(k, ci_alpha)
        rows.append(dict(ante=ante, antecedent=label(ante, catalog), n_ante=len(ante), cover=n, events=k,
                         confidence=k / n, lift=(k / n) / base, lift_lo=lo / n / base, lift_hi=hi / n / base))
    rules = pd.DataFrame(rows)
    return rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ 4. dedupe
def dedupe(rules, catalog):
    """`ante` is already a sorted tuple, so order variants are identical. Drop rules that use one attribute
    twice, then exact duplicates."""
    twice = rules["ante"].apply(lambda a: len({catalog[i][0] for i in a}) < len(a))
    return rules[~twice].drop_duplicates("ante").reset_index(drop=True)


# ------------------------------------------------------------------ 5. validate on holdout
def validate(rules, items_hold, min_conf_ev, tid):
    base = items_hold[tid].mean(); res = []
    for ante in rules["ante"]:
        _, n, k = _cover(items_hold, ante, tid)
        lift = (k / n) / base if n else np.nan
        parent = np.nan
        if len(ante) > 1:
            # Best lift any one-item-shorter version reaches. A rule has to beat all
            # of them, or the extra item is buying nothing.
            parents = []
            for d in ante:
                _, p_n, p_k = _cover(items_hold, tuple(x for x in ante if x != d), tid)
                parents.append((p_k / max(p_n, 1)) / base)
            parent = max(parents)
        res.append(dict(cover_conf=n, events_conf=k, lift_conf=lift, parent_lift_conf=parent))
    out = pd.concat([rules, pd.DataFrame(res)], axis=1)
    out["replicates"] = (out["lift_conf"] > 1) & (out["events_conf"] >= min_conf_ev)
    out["adds_over_parent"] = out["lift_conf"] > out["parent_lift_conf"].fillna(-np.inf)
    return out


# ------------------------------------------------------------------ 6. shortlist
def rank_rules(rules, items_hold, top_n, ci_alpha, fdr, max_overlap, tid):
    """Gates: replicates, adds_over_parent, BH-adjusted one-sided Fisher on the holdout. Score: lower Poisson
    CI of holdout lift. Diversity: greedy pick skipping rules whose covered holdout events overlap a chosen
    rule by Jaccard > max_overlap."""
    y = items_hold[tid].values; N, K = len(y), int(y.sum())
    cand = rules[rules["replicates"] & rules["adds_over_parent"]].copy()
    if cand.empty:
        return cand
    ev, p, lo = {}, [], []
    for idx, ante in cand["ante"].items():
        m, n_, k = _cover(items_hold, ante, tid)
        ev[idx] = m & y
        p.append(stats.fisher_exact([[k, n_ - k], [K - k, N - n_ - (K - k)]], alternative="greater")[1])
        lo.append(_poisson_ci(k, ci_alpha)[0] / n_ / (K / N))
    cand["p_conf"], cand["lift_lo_conf"] = p, lo
    cand["q_conf"] = stats.false_discovery_control(p, method="bh")
    cand = cand[cand["q_conf"] <= fdr].sort_values("lift_lo_conf", ascending=False)
    chosen = []
    for idx in cand.index:
        if all((ev[idx] & ev[c]).sum() / max((ev[idx] | ev[c]).sum(), 1) <= max_overlap for c in chosen):
            chosen.append(idx)
        if len(chosen) == top_n:
            break
    return cand.loc[chosen].reset_index(drop=True)


# ------------------------------------------------------------------ 7. end to end
def run(df, target, exclude, n_bins, max_levels, max_item_cov, min_events, max_len,
        min_lift, min_conf_ev, seed, top_n, discover_frac, ci_alpha, fdr, max_overlap, tid):
    """Mine on a stratified `discover_frac` of the rows, validate and shortlist on the rest.

    The signature is exactly the config block, so `run(df, **load_config())` is the whole pipeline
    and a caller who wants one value different passes it instead of editing the yaml."""
    rng = np.random.default_rng(seed)
    disc = np.zeros(len(df), bool)
    for cls in (0, 1):
        idx = np.flatnonzero(df[target].values == cls); rng.shuffle(idx)
        disc[idx[: int(discover_frac * len(idx))]] = True
    items, catalog = binarize(df, target, exclude, n_bins, max_levels, tid)
    freq = mine(items[disc], min_events, max_len, max_item_cov, tid)
    rules = dedupe(to_target_rules(freq, items[disc], catalog, min_lift, ci_alpha, tid), catalog)
    if rules.empty:
        print("no rules met min_events=%s / min_lift=%s on the discover half" % (min_events, min_lift))
        return rules, rules, catalog
    rules = validate(rules, items[~disc], min_conf_ev, tid)
    top = rank_rules(rules, items[~disc], top_n, ci_alpha, fdr, max_overlap, tid)
    return rules, top, catalog


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("input", help="csv holding the case-control sample")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="yaml carrying the config block (default: %(default)s)")
    ap.add_argument("--section", default=DEFAULT_SECTION,
                    help="which block of that yaml to read (default: %(default)s)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.section)
    pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 80)
    out, top, _ = run(pd.read_csv(args.input), **cfg)
    if out.empty:
        return 1
    cols = ["antecedent", "cover", "events", "lift", "lift_lo", "lift_hi", "events_conf", "lift_conf", "replicates", "adds_over_parent"]
    print(f"== all {len(out)} deduped rules"); print(out[cols].round(2).to_string())
    print(f"\n== shortlist (top {cfg['top_n']}, FDR {cfg['fdr']}, overlap <= {cfg['max_overlap']})")
    print(top[["antecedent", "events", "events_conf", "lift_conf", "lift_lo_conf", "q_conf"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
