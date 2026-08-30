"""
Profiling the engineered dataset: lineage, distributions, and the
relationship each feature has with the target.

THREE JOBS
----------
1. Lineage. Every column preprocess.py writes is NAMED from the config,
   so the mapping raw -> engineered can be reconstructed from the config
   rather than maintained by hand. `lineage_map` rebuilds the columns
   the config implies and diffs them against the columns actually on
   disk. A row marked "unmapped" means preprocess.py emitted something
   the config does not explain -- which is a real finding, not a
   cosmetic gap in the report.

2. Distributions. Count, missingness, and quantiles per feature.
   Missingness matters more than usual here: a null in a sequence
   position is STRUCTURAL (the customer had no fourth acquisition), not
   a data-quality defect, and the report says so rather than filing it
   as dirty data.

3. Relationship to the target. Event rate by decile, plus Information
   Value. IV is the right summary for this problem because it is
   monotone-agnostic and reads the same for numeric and categorical
   columns, so one ranking covers the whole feature set.

A NOTE ON WHICH DATASET TO PROFILE
----------------------------------
Profile the flat dataset by default. In the hazard file one customer
owns many rows, so a "distribution" there is a distribution over
customer-intervals: customers with long exposure are counted more
often, and IV is inflated accordingly. Both are legitimate as long as
the report says which it did, which it does.
"""

import numpy as np
import pandas as pd

# Standard credit-scoring bands. The top band is a leak detector: a
# single feature carrying IV above 0.5 on a rare event is far more
# likely to be an accident than a discovery.
IV_BANDS = [(0.02, "no signal"), (0.1, "weak"), (0.3, "medium"),
            (0.5, "strong"), (float("inf"), "suspiciously strong")]


# =====================================================================
# 1. lineage: raw -> engineered
# =====================================================================

def _families(cfg_pre):
    fams = sorted(set(cfg_pre["product_families"]["map"].values()))
    return fams + [cfg_pre["product_families"]["default_family"]]


def expected_columns(cfg_pre, design, schema):
    """
    Rebuild the engineered column list the config implies, mirroring
    build_features in preprocess.py. Returns a list of dicts.

    `schema` supplies the structural output names (customer, label,
    the hazard bookkeeping block) so nothing here is a literal.
    """
    col = cfg_pre["columns"]
    seq = cfg_pre["sequence"]
    agg = cfg_pre["aggregates"]
    numeric = cfg_pre["numeric_columns"]
    rows = []

    def add(name, block, source, transform):
        rows.append({"feature": name, "block": block, "source": source,
                     "transform": transform})

    add(schema["customer"], "key", col["id"], "carried through unchanged")
    if design == "flat":
        add(schema["anchor_date"], "key", col["date"],
            "last acquisition at or before the anchor rule")
        add(schema["flat_label"], "target", col["target"],
            "1 if the outcome falls within horizon_months of the anchor")
    else:
        bookkeeping = list(schema["hazard_bookkeeping"])
        descriptions = [
            "index of the %d-month interval at risk"
            % cfg_pre["hazard"]["bin_months"],
            "interval lower edge, months from anchor",
            "interval upper edge, months from anchor",
            "share of the interval actually observed; the row weight",
        ]
        for name, text in zip(bookkeeping, descriptions):
            add(name, "time", col["date"], text)
        for name in bookkeeping[len(descriptions):]:
            add(name, "time", col["date"], "hazard bookkeeping column")
        add(schema["hazard_label"], "target", col["target"],
            "1 only in the interval containing the outcome")

    # ---- aggregates over full history to the anchor -----------------
    add("agg_n_acquisitions", "aggregate", "(all rows)",
        "count of acquisitions at or before the anchor")
    if agg["emit_tenure"]:
        add("agg_tenure_months", "aggregate", col["date"],
            "months from first-ever acquisition to the anchor")
    if agg["count_by_family"]:
        for fam in _families(cfg_pre):
            add("agg_n_" + fam, "aggregate", col["product_type"],
                "count of acquisitions rolled up to family %s" % fam)

    for name, spec in numeric.items():
        for how in spec["aggregations"]:
            add("agg_%s_%s" % (name, how), "aggregate", name,
                "%s over full history to the anchor" % how)
            if spec.get("log1p") and how in ("sum", "max", "mean"):
                add("agg_%s_%s_log1p" % (name, how), "aggregate", name,
                    "log1p of the %s, to damp the right tail" % how)
        if agg["numerics_by_family"]:
            for fam in _families(cfg_pre):
                for how in spec["aggregations"]:
                    add("agg_%s_%s_%s" % (name, fam, how), "aggregate",
                        "%s, %s" % (name, col["product_type"]),
                        "%s of %s within family %s" % (how, name, fam))

    # ---- sequence block ---------------------------------------------
    add("n_acquisitions_in_window", "sequence", "(all rows)",
        "how many of the %d window slots are filled" % seq["window"])
    for pos in range(1, seq["window"] + 1):
        where = "position %d (1 = the anchor itself)" % pos
        for name in seq["carry_columns"]:
            add("pos%d_%s" % (pos, name), "sequence", name,
                "raw value at %s" % where)
        for name in seq["carry_percentiles"]:
            group = numeric.get(name, {}).get("percentile_group")
            add("pos%d_%s_pct" % (pos, name), "sequence", name,
                "0-100 rank of %s within %s, from the saved grid"
                % (name, _describe_group(group)))
        add("pos%d_present" % pos, "sequence", "(all rows)",
            "1 when %s exists" % where)

    if seq["emit_gaps"]:
        for pos in range(1, seq["window"]):
            add("gap_%d_%d" % (pos, pos + 1), "timing", col["date"],
                "months between acquisition %d and %d" % (pos, pos + 1))
        add("gap_mean", "timing", col["date"], "mean of the observed gaps")
        add("gap_min", "timing", col["date"], "shortest observed gap")
        add("gap_trend", "timing", col["date"],
            "most recent gap minus the oldest; negative means accelerating")
        add("bundle_count", "timing", col["date"],
            "count of zero-month gaps, i.e. products opened the same day")

    return rows


def _describe_group(levels):
    if not levels:
        return "(no grouping)"
    return " else ".join("+".join(map(str, keys)) for keys in levels)


def lineage_map(cfg_pre, actual_columns, design, schema):
    """
    Expected columns joined against what is really in the file. Columns
    present but not explained by the config are appended as `unmapped`,
    so drift between preprocess.py and its config surfaces here.
    """
    expected = pd.DataFrame(expected_columns(cfg_pre, design, schema))
    actual = set(actual_columns)
    expected["in_dataset"] = expected["feature"].isin(actual)

    explained = set(expected["feature"])
    extra = [c for c in actual_columns if c not in explained]
    if extra:
        expected = pd.concat([expected, pd.DataFrame([{
            "feature": c, "block": "unmapped", "source": "(unknown)",
            "transform": "present in the file but not implied by the config",
            "in_dataset": True} for c in extra])], ignore_index=True)
    return expected


# =====================================================================
# 2. distributions
# =====================================================================

def split_feature_types(df, exclude):
    """Numeric vs categorical, excluding keys/labels/bookkeeping."""
    numeric, categorical = [], []
    for name in df.columns:
        if name in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[name]):
            numeric.append(name)
        else:
            categorical.append(name)
    return numeric, categorical


def numeric_summary(df, columns):
    rows = []
    n = len(df)
    for name in columns:
        series = df[name]
        clean = series.dropna()
        rows.append({
            "feature": name,
            "non_null": int(clean.size),
            "missing_pct": 1.0 - (clean.size / float(n) if n else 0.0),
            "distinct": int(clean.nunique()),
            "mean": float(clean.mean()) if clean.size else np.nan,
            "std": float(clean.std()) if clean.size else np.nan,
            "p10": float(clean.quantile(0.10)) if clean.size else np.nan,
            "median": float(clean.median()) if clean.size else np.nan,
            "p90": float(clean.quantile(0.90)) if clean.size else np.nan,
            "max": float(clean.max()) if clean.size else np.nan,
        })
    table = pd.DataFrame(rows)
    return (table.sort_values("missing_pct", ascending=False)
            if len(table) else table)


def categorical_summary(df, columns, top_levels=3):
    rows = []
    n = len(df)
    for name in columns:
        series = df[name]
        clean = series.dropna()
        counts = clean.value_counts()
        top = ", ".join("%s (%.0f%%)" % (idx, 100.0 * val / n)
                        for idx, val in counts.head(top_levels).items())
        rows.append({
            "feature": name,
            "non_null": int(clean.size),
            "missing_pct": 1.0 - (clean.size / float(n) if n else 0.0),
            "levels": int(counts.size),
            "most_common": top or "—",
        })
    table = pd.DataFrame(rows)
    return (table.sort_values("missing_pct", ascending=False)
            if len(table) else table)


def structural_missing_note(cfg_pre):
    """
    Sequence positions beyond a customer's history are null by design.
    Naming them keeps a reader from reading the missingness table as a
    data-quality problem.
    """
    window = cfg_pre["sequence"]["window"]
    return ("Nulls in pos2..pos%d are structural: the customer simply had "
            "fewer than %d acquisitions before the anchor. pos*_present "
            "carries that as an explicit flag, and the model gets a "
            "was-missing indicator beside every imputed numeric column."
            % (window, window))


# =====================================================================
# 3. relationship to the target
# =====================================================================

def bin_feature(series, bins, min_bin_count):
    """
    Quantile bins for numerics, levels for categoricals. Missing is
    always its own bin -- for these features it carries meaning.
    Returns a Series of bin labels aligned to the input index.
    """
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > bins:
        try:
            binned = pd.qcut(series, bins, duplicates="drop")
            labels = binned.astype(object)
        except (ValueError, IndexError):
            labels = series.astype(object)
    else:
        labels = series.astype(object)

    labels = labels.where(series.notna(), other="(missing)")
    labels = labels.map(lambda v: str(v))

    # Pool anything too thin to interpret rather than reporting it.
    counts = labels.value_counts()
    rare = set(counts[counts < min_bin_count].index) - {"(missing)"}
    if rare:
        labels = labels.map(lambda v: "(other, thin bins)" if v in rare else v)
    return labels


def event_rate_by_bin(df, feature, label_col, bins=10, min_bin_count=25):
    """Per-bin count, event rate and lift against the overall rate."""
    labels = bin_feature(df[feature], bins, min_bin_count)
    target = df[label_col].astype(float)
    base = target.mean()

    grouped = pd.DataFrame({"bin": labels, "y": target}).groupby("bin", sort=False)
    table = grouped.agg(customers=("y", "size"),
                        events=("y", "sum")).reset_index()
    table["event_rate"] = table["events"] / table["customers"]
    table["lift"] = table["event_rate"] / base if base else np.nan
    table["share"] = table["customers"] / float(len(df))
    return table.sort_values("bin", key=lambda s: s.map(_bin_sort_key)), base


def _bin_sort_key(value):
    """Sort numeric interval labels by their left edge, text alphabetically."""
    text = str(value)
    if text.startswith(("(", "[")) and "," in text:
        try:
            return (0, float(text[1:].split(",")[0]))
        except ValueError:
            return (2, 0.0)
    if text == "(missing)":
        return (3, 0.0)
    if text.startswith("(other"):
        return (4, 0.0)
    return (1, 0.0)


def information_value(df, feature, label_col, bins=10, min_bin_count=25):
    """
    IV with a 0.5 continuity correction so an all-one-class bin does not
    send WoE to infinity. Returns (iv, per-bin table).
    """
    labels = bin_feature(df[feature], bins, min_bin_count)
    target = df[label_col].astype(float)
    total_events = target.sum()
    total_non = len(target) - total_events
    if total_events == 0 or total_non == 0:
        return np.nan, pd.DataFrame()

    grouped = pd.DataFrame({"bin": labels, "y": target}).groupby("bin", sort=False)
    table = grouped.agg(customers=("y", "size"),
                        events=("y", "sum")).reset_index()
    table["non_events"] = table["customers"] - table["events"]
    ev_share = (table["events"] + 0.5) / (total_events + 0.5 * len(table))
    non_share = (table["non_events"] + 0.5) / (total_non + 0.5 * len(table))
    table["woe"] = np.log(ev_share / non_share)
    table["contribution"] = (ev_share - non_share) * table["woe"]
    table["event_rate"] = table["events"] / table["customers"]
    iv = float(table["contribution"].sum())
    return iv, table.sort_values("bin", key=lambda s: s.map(_bin_sort_key))


def iv_band(value):
    if value is None or not np.isfinite(value):
        return "not computable"
    for edge, name in IV_BANDS:
        if value < edge:
            return name
    return "suspiciously strong"


def standardised_separation(df, features, label_col):
    """
    Cohen's d per feature: how many pooled standard deviations separate
    converters from everyone else. Sign is kept, so a feature can be
    compared across two datasets rather than just ranked within one.
    """
    rows = []
    target = df[label_col].astype(float)
    for name in features:
        series = pd.to_numeric(df[name], errors="coerce")
        pos = series[target == 1].dropna()
        neg = series[target == 0].dropna()
        if len(pos) < 2 or len(neg) < 2:
            continue
        pooled = np.sqrt(((len(pos) - 1) * pos.var() +
                          (len(neg) - 1) * neg.var()) /
                         float(len(pos) + len(neg) - 2))
        rows.append({"feature": name,
                     "mean_converter": float(pos.mean()),
                     "mean_other": float(neg.mean()),
                     "separation": float((pos.mean() - neg.mean()) / pooled)
                     if pooled else np.nan})
    return pd.DataFrame(rows)


def anchor_asymmetry(flat, hazard, features, schema, top_n=10):
    """
    Compare each feature's converter/non-converter separation under the
    two anchor rules.

    WHY THIS CHECK EXISTS
    ---------------------
    The two designs anchor non-converters differently. Fixed anchors
    them at their last acquisition at least horizon_months before the
    extract date, so that both classes get equal time at risk -- but
    that also throws away any acquisition a non-converter made inside
    the horizon, while a converter keeps their history right up to the
    outcome. History-length features (counts, tenure, window occupancy)
    therefore separate the classes MORE under the fixed design than
    under hazard, and some of that extra separation is the anchor rule
    rather than customer behaviour.

    This is a property of the documented design, not a defect. But a
    feature whose separation collapses when the cutoff is removed is one
    to be careful about, so the report names it instead of letting a
    reader take an inflated Information Value at face value.

    Returns the features whose separation shrinks most from flat to
    hazard, worst first.
    """
    customer_col = schema["customer"]
    hazard_label = schema["hazard_label"]
    flat_label = schema["flat_label"]

    one_row = hazard.drop_duplicates(subset=[customer_col], keep="first").copy()
    event_by_customer = hazard.groupby(customer_col)[hazard_label].max()
    one_row["_ever"] = one_row[customer_col].map(event_by_customer)

    shared = [f for f in features
              if f in flat.columns and f in one_row.columns
              and pd.api.types.is_numeric_dtype(flat[f])]
    if not shared:
        return pd.DataFrame()

    a = standardised_separation(flat, shared, flat_label).set_index("feature")
    b = standardised_separation(one_row, shared, "_ever").set_index("feature")
    joined = a.join(b, lsuffix="_fixed", rsuffix="_hazard", how="inner")
    if joined.empty:
        return pd.DataFrame()

    joined["shrinkage"] = (joined["separation_fixed"] -
                           joined["separation_hazard"])
    joined = joined[joined["separation_fixed"] > 0]
    return (joined.reset_index()
                  .sort_values("shrinkage", ascending=False)
                  .head(top_n)[["feature", "mean_converter_fixed",
                                "mean_other_fixed", "mean_other_hazard",
                                "separation_fixed", "separation_hazard",
                                "shrinkage"]])


def iv_ranking(df, features, label_col, bins=10, min_bin_count=25):
    rows = []
    for name in features:
        try:
            iv, _ = information_value(df, name, label_col, bins, min_bin_count)
        except Exception as exc:                       # noqa: BLE001
            # One awkward column must not take the whole report down.
            rows.append({"feature": name, "iv": np.nan,
                         "strength": "failed: %s" % type(exc).__name__,
                         "missing_pct": float(df[name].isna().mean())})
            continue
        rows.append({"feature": name, "iv": iv, "strength": iv_band(iv),
                     "missing_pct": float(df[name].isna().mean())})
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["feature", "iv", "strength",
                                     "missing_pct"])
    return table.sort_values("iv", ascending=False,
                             na_position="last").reset_index(drop=True)
