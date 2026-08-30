"""
From scores to a call list: refit, reason codes, segments, deciles.

THE TWO SCORES, AND WHY THEY ARE DIFFERENT
------------------------------------------
There are two legitimate scores for a customer and this module keeps
them apart on purpose.

  * The RANKING uses out-of-fold scores from model.py. Every customer
    was scored by a model that never saw them in training, so the
    ranking is unbiased for the population as it stands today. This is
    what the campaign list is sorted by.

  * The ARTIFACT is a model refit and dumped to model.joblib. It is
    what a scoring job would load for customers who were not in this
    extract. It is never used to rank the current population, because a
    model that has seen a customer in training scores them
    optimistically. Note that cross-validation runs on the DOWNSAMPLED
    frame while the artifact should be fit on the whole population --
    otherwise its probabilities sit above the true base rate. The
    bundle records which frame it actually got.

Using the refit model to rank the same customers it trained on is the
single easiest way to publish a target list that looks excellent and
performs at chance. Hence the split.

REASON CODES
------------
For a logistic model each customer's score decomposes exactly:

    log-odds = intercept + sum over features of (coef * transformed value)

so a per-customer contribution is a real decomposition, not an
approximation. Numeric features are standardised before the model sees
them, so a contribution reads as "this customer sits N standard
deviations from the mean on this feature, and that is worth this much
log-odds". Contributions are reported against the population average
customer, which is where every standardised numeric sits at zero.

For a tree model no such decomposition exists without adding a
dependency, so the fallback describes where the customer sits on the
globally most important features instead, and the report says which of
the two it used.
"""

import datetime
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =====================================================================
# refit the deliverable model
# =====================================================================

def refit_and_persist(model_module, df, numeric, categorical, cfg_mdl, path,
                      downsampled=False):
    """
    Fit one pipeline on `df` and dump it with enough metadata to be
    auditable later: what it was trained on, when, and with which
    library versions.

    `downsampled` records whether the caller handed us the sampled
    frame or the whole population, because the difference matters to
    anyone who loads this file later and it must not be guessed from
    the row count.
    """
    import sklearn

    col = cfg_mdl["columns"]
    features = numeric + categorical
    X = df[features]
    y = df[col["label"]].values

    weights = df["_sample_weight"].values.astype(float)
    if col["weight"]:
        weights = weights * df[col["weight"]].values.astype(float)

    pipe = model_module.build_pipeline(numeric, categorical, cfg_mdl)
    pipe = model_module.fit_fold(pipe, X, y, weights, cfg_mdl)

    bundle = {
        "pipeline": pipe,
        "numeric": numeric,
        "categorical": categorical,
        "design": cfg_mdl["design"],
        "label_column": col["label"],
        "model_config": cfg_mdl,
        "n_rows": int(len(df)),
        "n_customers": int(df[col["group"]].nunique()),
        "n_events": int(y.sum()),
        "sklearn_version": sklearn.__version__,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "downsampled_training_frame": bool(downsampled),
        "note": ("Refit on %s. Use for scoring NEW customers. Do not use it "
                 "to rank the customers it was trained on -- the out-of-fold "
                 "scores in predictions.csv are for that."
                 % ("a DOWNSAMPLED negative population; predicted "
                    "probabilities are shifted above the true base rate"
                    if downsampled else
                    "every eligible customer, negatives included")),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)
    return pipe, bundle


def converged(pipe):
    """
    Whether an iterative LINEAR fit converged, as (flag, iterations).

    Returns (None, None) for anything else. This guard matters: a
    gradient-booster also exposes n_iter_ and max_iter, and without
    early stopping they are equal by construction -- so a naive check
    reports "did not converge" and prints saga-specific advice for a
    model that has no convergence criterion at all.
    """
    model = pipe.named_steps.get("model")
    if not hasattr(model, "coef_"):
        return None, None
    n_iter = getattr(model, "n_iter_", None)
    max_iter = getattr(model, "max_iter", None)
    if n_iter is None or max_iter is None:
        return None, None
    reached = int(np.max(n_iter))
    return reached < max_iter, reached


# =====================================================================
# out-of-fold customer scores
# =====================================================================

def customer_scores(predictions, cfg_mdl):
    """
    predictions.csv holds one row per customer per repeat. Average
    across repeats so each customer has one score, and keep the spread
    so instability is visible rather than averaged away.
    """
    id_col = cfg_mdl["columns"]["id"]
    grouped = predictions.groupby(id_col)
    scores = grouped.agg(score=("score", "mean"),
                         score_min=("score", "min"),
                         score_max=("score", "max"),
                         actual=("actual", "max"),
                         repeats=("score", "size")).reset_index()
    return scores.sort_values("score", ascending=False).reset_index(drop=True)


def decile_table(scores, n_deciles=10):
    """
    Deciles of the ranking with observed rate, lift and cumulative
    capture -- the three numbers a campaign is actually planned from.
    """
    if scores.empty:
        return pd.DataFrame()
    ranked = scores.sort_values("score", ascending=False).reset_index(drop=True)
    n = len(ranked)
    # Rank-based cut, not qcut: qcut on tied scores produces uneven
    # deciles and a campaign brief that does not add up to 100%.
    ranked["decile"] = np.minimum(
        (np.arange(n) * n_deciles) // n + 1, n_deciles)

    base = ranked["actual"].mean()
    total_events = ranked["actual"].sum()

    table = (ranked.groupby("decile")
             .agg(customers=("actual", "size"),
                  mean_score=("score", "mean"),
                  events=("actual", "sum"))
             .reset_index())
    table["observed_rate"] = table["events"] / table["customers"]
    table["lift"] = table["observed_rate"] / base if base else np.nan
    table["cumulative_capture"] = (table["events"].cumsum() / total_events
                                   if total_events else np.nan)
    return table


# =====================================================================
# reason codes
# =====================================================================

def _pretty_feature(raw_name, numeric, categorical):
    """
    Turn a ColumnTransformer output name back into something a human
    reads. Names arrive as 'num__<col>', 'cat__<col>_<level>' or
    'num__missingindicator_<col>'.
    """
    name = raw_name
    for prefix in ("num__", "cat__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    if name.startswith("missingindicator_"):
        return name[len("missingindicator_"):], None, True

    # One-hot: find the longest configured column that prefixes it, so a
    # column name containing underscores still splits correctly.
    best = None
    for column in categorical:
        if name.startswith(column + "_") and (best is None or len(column) > len(best)):
            best = column
    if best is not None:
        return best, name[len(best) + 1:], False
    return name, None, False


def contribution_matrix(pipe, X):
    """
    Exact per-row log-odds decomposition for a linear model.
    Returns (contributions, feature_names) or (None, names) if the
    estimator is not linear.
    """
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    names = list(prep.get_feature_names_out())
    if not hasattr(model, "coef_"):
        return None, names
    transformed = prep.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return transformed * model.coef_[0], names


def reason_codes(pipe, X, numeric, categorical, top_k=3,
                 exclude_columns=(), ranked_features=None):
    """
    Top-k drivers pushing each row's score UP, as readable text.

    Only positive contributions are reported: the question a campaign
    asks is "why is this customer worth calling", not "what held them
    back". Rows with fewer than k positive contributions get fewer
    reasons rather than padded ones.

    `exclude_columns` names source columns that must never appear as a
    reason -- the hazard interval index above all. It is bookkeeping,
    it takes the same value for every customer being scored, and
    "interval = 1" is not a reason to call anybody.
    """
    contributions, names = contribution_matrix(pipe, X)
    if contributions is None:
        return (_deviation_reasons(X, numeric, top_k, ranked_features),
                "deviation")

    pretty = [_pretty_feature(n, numeric, categorical) for n in names]
    prep = pipe.named_steps["prep"]
    transformed = prep.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    excluded = set(exclude_columns)
    keep = np.array([pretty[i][0] not in excluded for i in range(len(names))])
    # Mask rather than drop, so column indices still line up with `pretty`.
    masked = np.where(keep, contributions, -np.inf)
    order = np.argsort(-masked, axis=1)[:, :top_k]
    out = []
    for row in range(contributions.shape[0]):
        parts = []
        for column in order[row]:
            value = contributions[row, column]
            if not np.isfinite(value) or value <= 0:
                break
            column_name, level, is_missing = pretty[column]
            if is_missing:
                text = "%s not populated" % column_name
            elif level is not None:
                text = "%s = %s" % (column_name, level)
            else:
                z = transformed[row, column]
                text = "%s %s" % (column_name,
                                  "well above average" if z > 0.5 else
                                  "above average" if z > 0 else
                                  "below average")
            parts.append("%s (%+.2f)" % (text, value))
        out.append("; ".join(parts) if parts
                   else "no single driver stands out")
    return out, "linear"


def _deviation_reasons(X, numeric, top_k, ranked_features=None):
    """
    Fallback for tree models: where the customer sits on the features
    the model leans on globally. Weaker than a real decomposition, and
    the report labels it as such.

    `ranked_features` is the feature ordering from features.csv (for a
    tree model, permutation importance). Without it there is no basis
    for choosing which columns to talk about -- sklearn's boosters
    expose no feature_importances_ -- so the fallback would silently
    describe whichever columns happen to come first in the file. When it
    is missing, say so rather than implying a ranking that does not
    exist.
    """
    if ranked_features:
        columns = [c for c in ranked_features if c in X.columns][:top_k]
    else:
        columns = []
    if not columns:
        return ["no per-customer decomposition available for this model type"
                ] * len(X)
    frame = X[columns].apply(pd.to_numeric, errors="coerce")
    medians = frame.median()
    out = []
    for _, row in frame.iterrows():
        parts = []
        for name in columns:
            value = row[name]
            if pd.isna(value):
                continue
            parts.append("%s %s median" % (name,
                                           "above" if value >= medians[name]
                                           else "below"))
        out.append("; ".join(parts[:top_k]) or "no single driver stands out")
    return out


# =====================================================================
# segments
# =====================================================================

def segment_population(frame, numeric, cfg_seg, scores=None):
    """
    Cluster the high-propensity population and describe each cluster by
    what makes it different from that population's average, not from
    zero.

    Names are generated from the two most distinguishing features, so
    they stay meaningful when the pipeline is pointed at a different
    dataset. A cluster with nothing distinguishing is named for its
    size rather than given a false characterisation.
    """
    usable = [c for c in numeric
              if c in frame.columns and frame[c].notna().sum() > 0]
    if len(frame) < cfg_seg["k"] or not usable:
        return None

    values = frame[usable].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median(numeric_only=True))
    values = values.loc[:, values.std(numeric_only=True) > 0]
    if values.empty:
        return None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)

    k = int(min(cfg_seg["k"], len(frame)))
    kmeans = KMeans(n_clusters=k, n_init=10,
                    random_state=cfg_seg["random_state"])
    assignment = kmeans.fit_predict(scaled)

    scaled_frame = pd.DataFrame(scaled, columns=values.columns,
                                index=values.index)
    scaled_frame["_segment"] = assignment

    profiles, rows = {}, []
    for index, segment in enumerate(sorted(set(assignment))):
        members = scaled_frame[scaled_frame["_segment"] == segment]
        deviation = members[values.columns].mean().sort_values(
            key=np.abs, ascending=False)
        strong = deviation[deviation.abs() >= cfg_seg["min_zscore"]]
        top = strong.head(cfg_seg["max_profile_features"])

        label = chr(ord("A") + index)
        if len(top) == 0:
            name = "Segment %s — no strong profile" % label
        else:
            name = "Segment %s — %s" % (label, ", ".join(
                "%s %s" % ("high" if v > 0 else "low", n)
                for n, v in top.head(2).items()))

        raw_means = frame.loc[members.index, list(top.index)].mean() \
            if len(top) else pd.Series(dtype="float64")
        profiles[segment] = {
            "name": name,
            "size": int(len(members)),
            "drivers": [(n, float(v), float(raw_means.get(n, np.nan)))
                        for n, v in top.items()],
        }
        row = {"segment": name, "customers": int(len(members)),
               "share": len(members) / float(len(frame))}
        if scores is not None:
            row["mean_score"] = float(scores.loc[members.index].mean())
            row["observed_rate"] = np.nan
        row["distinguishing_features"] = ", ".join(
            "%s %s (%.2f vs population, mean %.4g)"
            % ("high" if v > 0 else "low", n, v, raw_means.get(n, np.nan))
            for n, v in top.items()) or "nothing above the z-score floor"
        rows.append(row)

    return {"assignment": pd.Series(assignment, index=frame.index),
            "profiles": profiles,
            "table": pd.DataFrame(rows),
            "features_used": list(values.columns)}


def segment_names(result):
    """segment index -> display name, for joining onto the target list."""
    if result is None:
        return {}
    return {k: v["name"] for k, v in result["profiles"].items()}
