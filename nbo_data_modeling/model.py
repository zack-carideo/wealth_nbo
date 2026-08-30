"""
Fit and evaluate a cross-sell model on the output of preprocess.py.

    python model.py --config model_config.yaml

Handles both designs:

    flat    one row per customer. Straightforward binary classification.
    hazard  one row per customer-interval. The model predicts a
            per-interval hazard; those are combined into
            P(convert by H) = 1 - prod(1 - h_k) so that both designs are
            scored on the same customer-level footing and can be
            compared directly.

Outputs
    metrics.csv          headline metrics, mean and spread across folds
    calibration.csv      predicted vs observed rate by probability bin
    features.csv         coefficients (logistic) or importances (gbm),
                         with sign consistency across folds
    predictions.csv      out-of-fold score per customer

--------------------------------------------------------------------
THINGS THAT ARE EASY TO GET WRONG HERE
--------------------------------------------------------------------
  * Folds split on the customer, never on the row. In the hazard design
    a customer owns many rows and splitting by row leaks them across
    the fold boundary.
  * ROC-AUC is close to useless below 1% prevalence. Read PR-AUC and
    lift at the top of the ranking instead.
  * Row count is not sample size. Millions of hazard rows still carry
    only a few hundred events, and the event count is what limits how
    many features the data supports.
  * No oversampling or SMOTE. Class weights plus calibration afterwards.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =====================================================================
# setup
# =====================================================================

def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_data(cfg):
    col = cfg["columns"]
    df = pd.read_csv(cfg["io"]["input_path"])

    for role in ["id", "group", "label"]:
        if col[role] not in df.columns:
            raise ValueError("column '%s' (role: %s) is not in the input file"
                             % (col[role], role))

    df = df[df[col["label"]].notna()].copy()
    df[col["label"]] = df[col["label"]].astype(int)

    # A numeric column named in force_categorical (the hazard interval
    # index, typically) must become text, or the categorical imputer
    # will refuse to write its string fill value into an integer column.
    for name in col["force_categorical"]:
        if name in df.columns:
            df[name] = df[name].astype(str)

    return downsample_negatives(df, cfg)


def downsample_negatives(df, cfg):
    """
    Keep every customer who converted, sample the rest.

    Sampling is done on whole CUSTOMERS, never on rows. Dropping some of
    a customer's intervals would corrupt the survival product at
    evaluation time.

    Retained negatives carry weight 1/fraction so totals still reflect
    the full population. Ranking is unaffected either way; the weight
    keeps the predicted probabilities near the true base rate.
    """
    col = cfg["columns"]
    fraction = cfg.get("sampling", {}).get("negative_customer_fraction", 1.0)
    if fraction >= 1.0:
        df["_sample_weight"] = 1.0
        return df

    converted = df.groupby(col["group"])[col["label"]].max()
    positives = set(converted[converted == 1].index)
    negatives = np.array(sorted(set(converted.index) - positives))

    rng = np.random.default_rng(cfg["cv"]["random_state"])
    keep_n = rng.random(len(negatives)) < fraction
    keep = positives.union(set(negatives[keep_n]))

    out = df[df[col["group"]].isin(keep)].copy()
    out["_sample_weight"] = np.where(out[col["group"]].isin(positives), 1.0, 1.0 / fraction)
    print("downsampled negatives to %.0f%%: %d customers -> %d"
          % (fraction * 100, df[col["group"]].nunique(), out[col["group"]].nunique()))
    return out


def split_column_types(df, cfg):
    """Decide which columns are numeric, categorical, or excluded."""
    col = cfg["columns"]
    reserved = {col["id"], col["group"], col["label"], "_sample_weight"}
    if col["weight"]:
        reserved.add(col["weight"])
    reserved.update(col["drop"])

    forced = set(col["force_categorical"])
    numeric, categorical = [], []

    for name in df.columns:
        if name in reserved:
            continue
        if name in forced or df[name].dtype == object:
            categorical.append(name)
        elif pd.api.types.is_numeric_dtype(df[name]):
            numeric.append(name)
        # anything else (dates, mixed types) is skipped on purpose

    return numeric, categorical


# =====================================================================
# model
# =====================================================================

def build_pipeline(numeric, categorical, cfg):
    """Impute, encode, scale, fit. One object so CV cannot leak."""
    enc = cfg["encoding"]

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median",
                                 add_indicator=enc["add_missing_indicators"])),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=enc["min_category_frequency"],
                                 sparse_output=False)),
    ])

    prep = ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("cat", categorical_steps, categorical),
    ])

    return Pipeline([("prep", prep), ("model", make_estimator(cfg))])


def make_estimator(cfg, c_value=None):
    spec = cfg["model"]
    if spec["type"] == "logistic":
        p = spec["logistic"]
        # saga is the only solver supporting elastic net. It is slow on
        # wide hazard files; if fits drag, lower max_iter or reduce
        # sampling.negative_customer_fraction before changing the model.
        return LogisticRegression(
            solver="saga",
            l1_ratio=p["l1_ratio"],
            tol=p["tol"],
            C=c_value if c_value is not None else p["c_grid"][-1],
            max_iter=p["max_iter"],
            class_weight="balanced",
        )
    if spec["type"] == "gbm":
        p = spec["gbm"]
        return HistGradientBoostingClassifier(
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            max_iter=p["max_iter"],
            min_samples_leaf=p["min_samples_leaf"],
            l2_regularization=p["l2_regularization"],
            class_weight="balanced",
        )
    raise ValueError("model.type must be 'logistic' or 'gbm'")


def fit_fold(pipe, X, y, weights, cfg):
    """Fit one pipeline. Weights are passed through to the estimator."""
    if weights is None:
        pipe.fit(X, y)
    else:
        pipe.fit(X, y, model__sample_weight=weights)
    return pipe


# =====================================================================
# cross validation -> out-of-fold predictions
# =====================================================================

def run_cv(df, numeric, categorical, cfg):
    col = cfg["columns"]
    cv_cfg = cfg["cv"]
    features = numeric + categorical

    X = df[features]
    y = df[col["label"]].values
    groups = df[col["group"]].values
    weights = df["_sample_weight"].values
    if col["weight"]:
        weights = weights * df[col["weight"]].values

    all_preds = []      # one frame per repeat
    fold_models = []

    for repeat in range(cv_cfg["n_repeats"]):
        splitter = StratifiedGroupKFold(n_splits=cv_cfg["n_splits"], shuffle=True,
                                        random_state=cv_cfg["random_state"] + repeat)
        oof = np.full(len(df), np.nan)

        for train_idx, test_idx in splitter.split(X, y, groups):
            pipe = build_pipeline(numeric, categorical, cfg)
            w = weights[train_idx] if weights is not None else None
            pipe = fit_fold(pipe, X.iloc[train_idx], y[train_idx], w, cfg)
            oof[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
            fold_models.append(pipe)

        # id and group are usually the same column; keep it once.
        keep_cols = list(dict.fromkeys([col["id"], col["group"], col["label"]]))
        frame = df[keep_cols].copy()
        frame["repeat"] = repeat
        frame["score"] = oof
        if col["weight"]:
            frame[col["weight"]] = df[col["weight"]].values
        for extra in ["interval"]:
            if extra in df.columns:
                frame[extra] = df[extra].values
        all_preds.append(frame)

    return pd.concat(all_preds, ignore_index=True), fold_models


# =====================================================================
# turn predictions into one score per customer
# =====================================================================

def to_customer_level(preds, cfg):
    """
    flat    already one row per customer.
    hazard  combine per-interval hazards into P(convert by horizon):
                P = 1 - prod over k of (1 - h_k)
            Truth is whether the customer had the event inside the
            horizon. Customers censored before the horizon are dropped
            from evaluation, because their outcome is genuinely unknown.
    """
    col = cfg["columns"]
    ev = cfg["evaluation"]

    if cfg["design"] == "flat":
        out = preds.rename(columns={col["label"]: "actual"})
        return out[["repeat", col["id"], "score", "actual"]]

    n_bins = ev["hazard_horizon_months"] // ev["hazard_bin_months"]
    inside = preds[preds["interval"].astype(int) <= n_bins]

    rows = []
    for (repeat, cid), grp in inside.groupby(["repeat", col["id"]]):
        survival = float(np.prod(1.0 - grp["score"].values))
        had_event = int(grp[col["label"]].max())
        observed_full = grp["interval"].astype(int).max() >= n_bins
        if not had_event and not observed_full:
            continue                      # censored before the horizon
        rows.append((repeat, cid, 1.0 - survival, had_event))

    return pd.DataFrame(rows, columns=["repeat", col["id"], "score", "actual"])


# =====================================================================
# metrics
# =====================================================================

def lift_at(scores, actual, fraction):
    """Event rate in the top `fraction` divided by the overall rate."""
    n = max(1, int(round(len(scores) * fraction)))
    order = np.argsort(-scores)
    base = actual.mean()
    if base == 0:
        return np.nan
    return actual[order[:n]].mean() / base


def evaluate(customer_preds, cfg):
    ev = cfg["evaluation"]
    rows = []

    for repeat, grp in customer_preds.groupby("repeat"):
        s = grp["score"].values
        a = grp["actual"].values
        if a.sum() == 0:
            continue
        metrics = {
            "repeat": repeat,
            "n_customers": len(a),
            "n_events": int(a.sum()),
            "prevalence": a.mean(),
            "pr_auc": average_precision_score(a, s),
            "roc_auc": roc_auc_score(a, s) if len(set(a)) > 1 else np.nan,
            "brier": brier_score_loss(a, s),
        }
        for frac in ev["lift_at"]:
            metrics["lift_top_%d%%" % int(frac * 100)] = lift_at(s, a, frac)
        rows.append(metrics)

    per_repeat = pd.DataFrame(rows)
    numeric_cols = [c for c in per_repeat.columns if c != "repeat"]
    summary = pd.DataFrame({
        "metric": numeric_cols,
        "mean": [per_repeat[c].mean() for c in numeric_cols],
        "std": [per_repeat[c].std() for c in numeric_cols],
        "min": [per_repeat[c].min() for c in numeric_cols],
        "max": [per_repeat[c].max() for c in numeric_cols],
    })
    return summary, per_repeat


def calibration_table(customer_preds, cfg):
    """Predicted vs observed rate by score bin. Flat and honest beats clever."""
    n_bins = cfg["evaluation"]["calibration_bins"]
    df = customer_preds.copy()
    try:
        df["bin"] = pd.qcut(df["score"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        df["bin"] = 0
    return (df.groupby("bin")
              .agg(n=("actual", "size"),
                   mean_predicted=("score", "mean"),
                   observed_rate=("actual", "mean"))
              .reset_index())


# =====================================================================
# feature report
#
# For logistic this doubles as the stability check: a feature whose sign
# flips across folds is not a finding, it is noise.
# =====================================================================

def feature_report(fold_models, df, numeric, categorical, cfg):
    names = fold_models[0].named_steps["prep"].get_feature_names_out()

    if cfg["model"]["type"] == "logistic":
        coefs = np.vstack([m.named_steps["model"].coef_[0] for m in fold_models])
        signs = np.sign(coefs)
        dominant = np.where(signs.sum(axis=0) >= 0, 1, -1)
        return pd.DataFrame({
            "feature": names,
            "mean_coef": coefs.mean(axis=0),
            "std_coef": coefs.std(axis=0),
            "sign_consistency": (signs == dominant).mean(axis=0),
            "nonzero_rate": (coefs != 0).mean(axis=0),
        }).sort_values("mean_coef", key=np.abs, ascending=False)

    col = cfg["columns"]
    features = numeric + categorical
    pipe = fold_models[-1]
    result = permutation_importance(
        pipe, df[features], df[col["label"]].values,
        n_repeats=cfg["model"]["gbm"]["permutation_repeats"],
        random_state=cfg["cv"]["random_state"],
        scoring="average_precision",
    )
    return pd.DataFrame({
        "feature": features,
        "mean_importance": result.importances_mean,
        "std_importance": result.importances_std,
    }).sort_values("mean_importance", ascending=False)


# =====================================================================
# main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="model_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["io"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(cfg)
    numeric, categorical = split_column_types(df, cfg)
    print("design=%s  rows=%d  customers=%d  row-events=%d"
          % (cfg["design"], len(df), df[cfg["columns"]["group"]].nunique(),
             int(df[cfg["columns"]["label"]].sum())))
    print("features: %d numeric, %d categorical" % (len(numeric), len(categorical)))

    preds, fold_models = run_cv(df, numeric, categorical, cfg)
    customer_preds = to_customer_level(preds, cfg)
    summary, per_repeat = evaluate(customer_preds, cfg)

    summary.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    per_repeat.to_csv(os.path.join(out_dir, "metrics_by_repeat.csv"), index=False)
    calibration_table(customer_preds, cfg).to_csv(
        os.path.join(out_dir, "calibration.csv"), index=False)
    customer_preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    feature_report(fold_models, df, numeric, categorical, cfg).to_csv(
        os.path.join(out_dir, "features.csv"), index=False)

    print()
    print(summary.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print()
    print("written to %s" % out_dir)


if __name__ == "__main__":
    main()
