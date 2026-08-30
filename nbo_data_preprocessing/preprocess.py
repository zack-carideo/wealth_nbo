"""
Cross-sell preprocessing: raw acquisition rows -> two modelling datasets.

    flat_dataset.csv     one row per eligible customer, fixed-horizon label
    hazard_dataset.csv   one row per customer per interval at risk
    exclusions.csv       every dropped customer with a reason
    percentile_grids.json  fitted rank artifact, reused at scoring time

Every input comes from config.yaml. This file contains no column names,
no dates and no thresholds.

Run:
    python preprocess.py --config config.yaml

--------------------------------------------------------------------
HOW IT FITS TOGETHER
--------------------------------------------------------------------
    1. load + validate
    2. split outcome rows away from acquisition rows      <- do this first
    3. pick an anchor per customer (differs by design)
    4. build features from rows at or before the anchor
    5. attach labels: one row per customer, or one per interval

Steps 4 and 5 are independent, so the same feature builder serves both
datasets. The only difference between the two designs is the anchor
rule in step 3 and the label shape in step 5.

--------------------------------------------------------------------
TWO RULES THAT MUST NOT BE RELAXED
--------------------------------------------------------------------
    * Outcome rows are removed before any aggregation. They carry a
      date and nothing else, so any groupby that still sees them will
      silently corrupt counts and "last row" lookups.
    * No feature may use a row dated after the anchor, and no feature
      may encode the distance from the anchor to the outcome or to the
      extract date. Both leak the label.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml


# =====================================================================
# small helpers
# =====================================================================

def months_between(start, end):
    """Whole months from start to end. Negative if end precedes start."""
    return (end.year - start.year) * 12 + (end.month - start.month)


def add_months(ts, n):
    return ts + pd.DateOffset(months=n)


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def require_columns(df, names, where):
    missing = [c for c in names if c and c not in df.columns]
    if missing:
        raise ValueError("config %s names columns absent from input: %s" % (where, missing))


# =====================================================================
# 1. load and validate
# =====================================================================

def load_data(cfg):
    col = cfg["columns"]
    df = pd.read_csv(cfg["io"]["input_path"])

    require_columns(df, [col["id"], col["date"], col["target"], col["product_type"],
                         col["tiebreak"]], "columns")
    require_columns(df, list(cfg["numeric_columns"].keys()), "numeric_columns")

    df[col["date"]] = pd.to_datetime(df[col["date"]])
    for name in cfg["numeric_columns"]:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    # Stable sort so that same-day ties keep input order when no
    # tiebreak column is configured.
    sort_keys = [col["id"], col["date"]]
    if col["tiebreak"]:
        sort_keys.append(col["tiebreak"])
    df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    # product_family is derived here once and used everywhere after.
    fam = cfg["product_families"]
    df["product_family"] = (df[col["product_type"]]
                            .map(fam["map"])
                            .fillna(fam["default_family"]))
    # Outcome rows have no product type, so blank their derived family.
    df.loc[df[col["target"]].notna(), "product_family"] = np.nan
    return df


def split_outcome_rows(df, cfg):
    """Return (acquisitions, outcomes). Must run before any aggregation."""
    tgt = cfg["columns"]["target"]
    is_outcome = df[tgt].notna()
    return df[~is_outcome].copy(), df[is_outcome].copy()


# =====================================================================
# 2. anchors and eligibility
# =====================================================================

def compute_anchors(acq, out, cfg, mode):
    """
    One row per customer describing what to do with them.

    mode="fixed"   negatives anchor at their last acquisition that is at
                   least horizon_months before the extract date, so both
                   classes get the same time at risk. Converters whose
                   gap exceeds the horizon are excluded.
    mode="hazard"  everybody anchors at their last acquisition. Censoring
                   is handled by the interval expansion instead, so no
                   horizon is needed and nobody is dropped for timing.

    Columns out: customer, anchor_date, keep, reason, event, event_month.
    """
    col = cfg["columns"]
    lab = cfg["labeling"]
    extract = pd.Timestamp(lab["extract_date"])
    horizon = lab["horizon_months"]
    cutoff = add_months(extract, -horizon)

    first_outcome = out.groupby(col["id"])[col["date"]].min()
    rows = []

    for cid, grp in acq.groupby(col["id"], sort=True):
        dates = grp[col["date"]]
        outcome_date = first_outcome.get(cid, pd.NaT)

        if len(grp) < lab["min_acquisitions"]:
            rows.append((cid, pd.NaT, False, "too few acquisitions", np.nan, np.nan))
            continue

        # Originated inside the target product line: not in the population.
        if pd.notna(outcome_date) and outcome_date < dates.min():
            rows.append((cid, pd.NaT, False, "outcome precedes first acquisition",
                         np.nan, np.nan))
            continue

        if pd.notna(outcome_date):
            prior = dates[dates < outcome_date]
            if prior.empty:
                rows.append((cid, pd.NaT, False, "no acquisition before outcome",
                             np.nan, np.nan))
                continue
            anchor = prior.max()
            gap = months_between(anchor, outcome_date)
            if mode == "fixed" and gap > horizon:
                rows.append((cid, anchor, False, "outcome %dm beyond horizon" % gap,
                             np.nan, np.nan))
            else:
                rows.append((cid, anchor, True, "event", 1, gap))
        else:
            if mode == "fixed":
                eligible = dates[dates <= cutoff]
                if eligible.empty:
                    rows.append((cid, pd.NaT, False, "censored: no acquisition before cutoff",
                                 np.nan, np.nan))
                    continue
                anchor = eligible.max()
            else:
                anchor = dates.max()
            rows.append((cid, anchor, True, "censored", 0,
                         months_between(anchor, extract)))

    return pd.DataFrame(rows, columns=["customer", "anchor_date", "keep", "reason",
                                       "event", "event_month"])


# =====================================================================
# 3. percentile grids
#
# Ranking a balance within its own product type and vintage makes values
# comparable across products, damps the long right tail, and absorbs
# inflation and the rate environment in one step. The grid is FIT ONCE on
# training rows and saved. Recomputing it at scoring time would make a
# customer's feature value depend on whoever else is in that day's batch.
# =====================================================================

def _group_values(df, keys, date_col):
    """Build the group label for each row from a list of key names."""
    parts = []
    for key in keys:
        if key == "year":
            parts.append(df[date_col].dt.year.astype(str))
        else:
            parts.append(df[key].astype(str))
    return parts[0].str.cat(parts[1:], sep="|") if len(parts) > 1 else parts[0]


def fit_percentile_grids(acq, cfg):
    """{column: {level_index: {group_label: {"n": int, "edges": [...]}}}}"""
    if not cfg["percentiles"]["enable"]:
        return {}

    date_col = cfg["columns"]["date"]
    probs = np.linspace(0, 100, 101)
    grids = {}

    for name, spec in cfg["numeric_columns"].items():
        levels = spec.get("percentile_group")
        if not levels:
            continue
        present = acq[acq[name].notna()]
        grids[name] = {}
        for idx, keys in enumerate(levels):
            labels = _group_values(present, keys, date_col)
            per_level = {}
            for label, sub in present.groupby(labels):
                per_level[label] = {
                    "n": int(len(sub)),
                    "edges": np.percentile(sub[name].values, probs).tolist(),
                }
            grids[name][str(idx)] = per_level
    return grids


def apply_percentiles(acq, grids, cfg):
    """Add <column>_pct in 0-100. Structural nulls stay null."""
    if not grids:
        return acq

    date_col = cfg["columns"]["date"]
    min_n = cfg["percentiles"]["min_cell_size"]

    for name, spec in cfg["numeric_columns"].items():
        if name not in grids:
            continue
        levels = spec["percentile_group"]
        label_by_level = [_group_values(acq, keys, date_col) for keys in levels]
        result = np.full(len(acq), np.nan)
        values = acq[name].values

        for i in range(len(acq)):
            if np.isnan(values[i]):
                continue
            for idx in range(len(levels)):
                cell = grids[name].get(str(idx), {}).get(label_by_level[idx].iloc[i])
                if cell and cell["n"] >= min_n:
                    result[i] = float(np.searchsorted(cell["edges"], values[i]))
                    break
        acq[name + "_pct"] = np.clip(result, 0, 100)
    return acq


# =====================================================================
# 4. feature construction
#
# Called once per design, with that design's anchors. Only rows at or
# before the anchor are visible.
# =====================================================================

def _family_list(cfg):
    fams = sorted(set(cfg["product_families"]["map"].values()))
    return fams + [cfg["product_families"]["default_family"]]


def build_features(acq, anchors, cfg):
    col = cfg["columns"]
    seq_cfg = cfg["sequence"]
    agg_cfg = cfg["aggregates"]
    window = seq_cfg["window"]
    families = _family_list(cfg)

    anchor_by_id = anchors.set_index("customer")["anchor_date"].to_dict()
    records = []

    for cid, grp in acq.groupby(col["id"], sort=True):
        anchor = anchor_by_id.get(cid)
        if anchor is None or pd.isna(anchor):
            continue

        # Everything after the anchor is invisible. A customer who kept
        # buying products after the anchor still has those rows in the
        # source file; using them would leak.
        hist = grp[grp[col["date"]] <= anchor]
        if hist.empty:
            continue

        rec = {"customer": cid}

        # ---- full-history aggregates ---------------------------------
        rec["agg_n_acquisitions"] = len(hist)
        if agg_cfg["emit_tenure"]:
            rec["agg_tenure_months"] = months_between(hist[col["date"]].min(), anchor)

        if agg_cfg["count_by_family"]:
            counts = hist["product_family"].value_counts()
            for fam in families:
                rec["agg_n_" + fam] = int(counts.get(fam, 0))

        for name, spec in cfg["numeric_columns"].items():
            for how in spec["aggregations"]:
                value = _aggregate(hist[name], how)
                rec["agg_%s_%s" % (name, how)] = value
                if spec.get("log1p") and how in ("sum", "max", "mean"):
                    rec["agg_%s_%s_log1p" % (name, how)] = _safe_log1p(value)
            if agg_cfg["numerics_by_family"]:
                for fam in families:
                    sub = hist.loc[hist["product_family"] == fam, name]
                    for how in spec["aggregations"]:
                        rec["agg_%s_%s_%s" % (name, fam, how)] = _aggregate(sub, how)

        # ---- sequence block, position 1 = anchor ---------------------
        recent = hist.tail(window).iloc[::-1].reset_index(drop=True)
        rec["n_acquisitions_in_window"] = len(recent)

        for pos in range(1, window + 1):
            has_row = pos <= len(recent)
            row = recent.iloc[pos - 1] if has_row else None
            for name in seq_cfg["carry_columns"]:
                rec["pos%d_%s" % (pos, name)] = row[name] if has_row else np.nan
            for name in seq_cfg["carry_percentiles"]:
                key = name + "_pct"
                rec["pos%d_%s" % (pos, key)] = (row[key] if has_row and key in recent.columns
                                                else np.nan)
            rec["pos%d_present" % pos] = int(has_row)

        # ---- gaps ----------------------------------------------------
        # Distances BETWEEN acquisitions only. The distance from the
        # anchor forward is the label and never appears here.
        if seq_cfg["emit_gaps"]:
            gaps = []
            for pos in range(1, window):
                if pos + 1 <= len(recent):
                    g = months_between(recent[col["date"]].iloc[pos],
                                       recent[col["date"]].iloc[pos - 1])
                else:
                    g = np.nan
                rec["gap_%d_%d" % (pos, pos + 1)] = g
                gaps.append(g)
            observed = [g for g in gaps if not pd.isna(g)]
            rec["gap_mean"] = float(np.mean(observed)) if observed else np.nan
            rec["gap_min"] = float(np.min(observed)) if observed else np.nan
            # Recent gap minus oldest gap. Negative means accelerating.
            rec["gap_trend"] = (observed[0] - observed[-1]) if len(observed) > 1 else np.nan
            rec["bundle_count"] = int(sum(1 for g in observed if g == 0))

        records.append(rec)

    return pd.DataFrame(records)


def _aggregate(series, how):
    clean = series.dropna()
    if how == "count_nonnull":
        return int(len(clean))
    if clean.empty:
        return np.nan
    return float(getattr(clean, how)())


def _safe_log1p(value):
    if pd.isna(value) or value < 0:
        return np.nan
    return float(np.log1p(value))


# =====================================================================
# 5a. flat dataset
# =====================================================================

def build_flat_dataset(features, anchors):
    kept = anchors[anchors["keep"]][["customer", "anchor_date", "event"]]
    out = kept.merge(features, on="customer", how="inner")
    return out.rename(columns={"event": "label"})


# =====================================================================
# 5b. hazard dataset
#
# One row per customer per interval at risk. event=1 lands only in the
# interval containing the outcome. Time-fixed features repeat down a
# customer's rows, which is expected and correct.
#
# Row count is not sample size: the number of events still governs how
# many features the data can support.
# =====================================================================

def build_hazard_dataset(features, anchors, cfg):
    hz = cfg["hazard"]
    width = hz["bin_months"]
    cap = hz["max_followup_months"]
    min_exposure = hz["min_exposure_frac"]

    kept = anchors[anchors["keep"]]
    rows = []

    for _, spell in kept.iterrows():
        months = min(int(spell["event_month"]), cap)
        months = max(months, 0)
        # An outcome at or past the cap becomes administratively censored.
        event_in_spell = int(spell["event"]) if months < cap else 0

        # Intervals cover (low, high], so month m sits in interval
        # ceil(m / width). Indexing this way keeps a spell that ends
        # exactly on a boundary out of a zero-exposure interval.
        n_intervals = max(1, int(np.ceil(months / float(width))))

        for k in range(1, n_intervals + 1):
            low = (k - 1) * width
            is_last = (k == n_intervals)
            exposure = min(months - low, width) / float(width) if is_last else 1.0
            exposure = max(exposure, 1.0 / width)   # a spell shorter than one
                                                    # month still gets exposure
            # Never drop the interval holding the event: events are the
            # scarce quantity and the partial-exposure filter exists only
            # to trim thin censored tails.
            if is_last and not event_in_spell and exposure < min_exposure:
                continue
            rows.append({
                "customer": spell["customer"],
                "interval": k,
                "months_low": low,
                "months_high": k * width,
                "exposure_frac": round(exposure, 3),
                "event": 1 if (is_last and event_in_spell) else 0,
            })

    skeleton = pd.DataFrame(rows)
    if skeleton.empty:
        return skeleton
    return skeleton.merge(features, on="customer", how="inner")


# =====================================================================
# main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["io"]["output_dir"], exist_ok=True)
    os.makedirs(cfg["io"]["artifact_dir"], exist_ok=True)

    df = load_data(cfg)
    acq, out = split_outcome_rows(df, cfg)
    print("loaded %d rows: %d acquisitions, %d outcomes, %d customers"
          % (len(df), len(acq), len(out), df[cfg["columns"]["id"]].nunique()))

    # Fit percentile grids on acquisition rows only, then save them. The
    # scoring job must load this file rather than refit.
    grids = fit_percentile_grids(acq, cfg)
    grid_path = os.path.join(cfg["io"]["artifact_dir"], "percentile_grids.json")
    with open(grid_path, "w") as fh:
        json.dump(grids, fh)
    acq = apply_percentiles(acq, grids, cfg)

    all_exclusions = []
    for mode, filename in [("fixed", "flat_dataset.csv"), ("hazard", "hazard_dataset.csv")]:
        anchors = compute_anchors(acq, out, cfg, mode=mode)
        features = build_features(acq, anchors, cfg)

        if mode == "fixed":
            data = build_flat_dataset(features, anchors)
        else:
            data = build_hazard_dataset(features, anchors, cfg)

        path = os.path.join(cfg["io"]["output_dir"], filename)
        data.to_csv(path, index=False)

        dropped = anchors[~anchors["keep"]].copy()
        dropped["design"] = mode
        all_exclusions.append(dropped[["design", "customer", "reason"]])

        events = int(data["label"].sum()) if mode == "fixed" else int(data["event"].sum())
        print("%-7s -> %-19s %6d rows, %5d customers, %4d events, %3d features"
              % (mode, filename, len(data), data["customer"].nunique(), events,
                 data.shape[1]))

    excl_path = os.path.join(cfg["io"]["output_dir"], "exclusions.csv")
    pd.concat(all_exclusions).to_csv(excl_path, index=False)
    print("artifacts -> %s" % grid_path)


if __name__ == "__main__":
    main()
