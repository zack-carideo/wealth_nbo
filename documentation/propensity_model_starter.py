"""
================================================================================
PROPENSITY MODEL — GENERIC STARTER PIPELINE
================================================================================
Purpose:
    Given a dataset of clients × products × balances, predict the probability
    that a client will adopt a product they do not currently hold.

Assumptions:
    - Input is a long-format table: one row per client-product pair.
    - Minimum columns: client_id, product, balance
    - This is a SINGLE-PRODUCT propensity model. To score multiple products,
      loop this pipeline once per target product.

Call Out:  
    The one thing I'd call out: the most impactful single enhancement is switching from a random split to a temporal out-of-time split, 
    because propensity models degrade fast and you want your validation to reflect that reality from day one.

Author:  Zack (FCB Enterprise AI)
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# STEP 0 — CONFIGURATION
# ==============================================================================

TARGET_PRODUCT = "HELOC"  # <-- change this to whichever product you're modeling

# [ENHANCEMENT] Move config to a YAML/JSON file for pipeline parameterization.
# [ENHANCEMENT] Add date windows for train/test temporal splits instead of random.


# ==============================================================================
# STEP 1 — LOAD RAW DATA
# ==============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Expects a long-format file with at minimum:
        client_id  | product     | balance
        C001       | Checking    | 12500.00
        C001       | Savings     | 45000.00
        C001       | Mortgage    | 210000.00
        C002       | Checking    | 3200.00
        ...
    """
    df = pd.read_csv(filepath)

    required_cols = {"client_id", "product", "balance"}
    assert required_cols.issubset(df.columns), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )

    print(f"Loaded {len(df):,} rows | {df['client_id'].nunique():,} clients "
          f"| {df['product'].nunique()} products")
    return df


# ==============================================================================
# STEP 2 — BUILD CLIENT-LEVEL FEATURE TABLE
# ==============================================================================

def build_features(df: pd.DataFrame, target_product: str) -> pd.DataFrame:
    """
    Pivots the long-format data into one row per client with:
        - Binary flags: has_<product> (0/1)
        - Balance columns: bal_<product>
        - Aggregate features: total_balance, product_count, avg_balance
        - Target: 1 if client holds the target product, else 0

    [ENHANCEMENT] Add balance concentration (HHI across products).
    [ENHANCEMENT] Add balance percentile ranks within each product.
    [ENHANCEMENT] Add tenure, demographics, channel activity if available.
    [ENHANCEMENT] Add time-series features: balance trend, velocity, volatility.
    """
    # --- Product ownership flags ---
    ownership = (
        df.assign(has_product=1)
        .pivot_table(index="client_id", columns="product",
                     values="has_product", fill_value=0)
        .add_prefix("has_")
    )

    # --- Balance by product ---
    balances = (
        df.pivot_table(index="client_id", columns="product",
                       values="balance", aggfunc="sum", fill_value=0)
        .add_prefix("bal_")
    )

    # --- Aggregates ---
    agg = df.groupby("client_id")["balance"].agg(
        total_balance="sum",
        avg_balance="mean",
        max_balance="max",
        product_count="count"          # count of products held
    )

    # --- Combine ---
    features = ownership.join(balances).join(agg).reset_index()

    # --- Target variable ---
    target_col = f"has_{target_product}"
    if target_col not in features.columns:
        raise ValueError(f"Target product '{target_product}' not found in data.")

    features["target"] = features[target_col].astype(int)

    print(f"Feature table: {features.shape[0]:,} clients × "
          f"{features.shape[1]} columns")
    print(f"Target rate ({target_product}): "
          f"{features['target'].mean():.2%}")

    return features


# ==============================================================================
# STEP 3 — PREPARE TRAIN / TEST SPLIT
# ==============================================================================

def prepare_splits(features: pd.DataFrame, target_product: str):
    """
    Simple random stratified split. Drops target-leaking columns.

    [ENHANCEMENT] Use temporal split (train on month M, test on M+1)
                  for realistic out-of-time validation.
    [ENHANCEMENT] Add k-fold cross-validation for more robust estimates.
    [ENHANCEMENT] For severe class imbalance, consider SMOTE or
                  stratified sampling — but always validate on
                  the natural distribution.
    """
    # Remove columns that leak the target
    leak_cols = [f"has_{target_product}", f"bal_{target_product}"]
    drop_cols = ["client_id", "target"] + [c for c in leak_cols if c in features.columns]

    X = features.drop(columns=drop_cols)
    y = features["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"Train target rate: {y_train.mean():.2%}  |  "
          f"Test target rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# STEP 4 — TRAIN MODEL
# ==============================================================================

def train_model(X_train, y_train):
    """
    Logistic regression as the transparent baseline.

    [ENHANCEMENT] Swap in XGBoost / LightGBM for better lift once
                  governance allows non-linear models.
    [ENHANCEMENT] Add hyperparameter tuning (GridSearchCV / Optuna).
    [ENHANCEMENT] Add regularization path analysis (L1 for feature
                  selection, L2 for stability).
    [ENHANCEMENT] Add fairness constraints or post-hoc bias audit
                  (disparate impact ratio, equal opportunity).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",   # handles class imbalance
        max_iter=1000,
        random_state=42
    )
    model.fit(X_scaled, y_train)

    # --- Coefficient inspection (interpretability) ---
    coef_df = (
        pd.DataFrame({
            "feature": X_train.columns,
            "coefficient": model.coef_[0]
        })
        .sort_values("coefficient", key=abs, ascending=False)
    )
    print("\nTop feature coefficients:")
    print(coef_df.head(10).to_string(index=False))

    return model, scaler


# ==============================================================================
# STEP 5 — EVALUATE
# ==============================================================================

def evaluate(model, scaler, X_test, y_test):
    """
    Basic discrimination and calibration check.

    [ENHANCEMENT] Add calibration curve (reliability diagram).
    [ENHANCEMENT] Add lift/gain charts and decile analysis — these
                  are what marketing actually uses to set cutoffs.
    [ENHANCEMENT] Add KS statistic, Gini coefficient.
    [ENHANCEMENT] Add Population Stability Index (PSI) for monitoring.
    [ENHANCEMENT] Add fairness metrics by protected class.
    """
    X_scaled = scaler.transform(X_test)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    print(f"\nAUC-ROC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    return y_prob


# ==============================================================================
# STEP 6 — SCORE & OUTPUT
# ==============================================================================

def score_clients(features, model, scaler, target_product):
    """
    Score ALL clients and output a ranked list.

    [ENHANCEMENT] Filter to only clients who do NOT already hold the
                  target product (true prospects only).
    [ENHANCEMENT] Add a multi-product loop: score each product,
                  then stack into a next-best-product matrix.
    [ENHANCEMENT] Write scores to a database / feature store for
                  downstream campaign systems.
    """
    leak_cols = [f"has_{target_product}", f"bal_{target_product}"]
    drop_cols = ["client_id", "target"] + [c for c in leak_cols if c in features.columns]

    X_all = features.drop(columns=drop_cols)
    X_scaled = scaler.transform(X_all)

    features = features.copy()
    features["propensity_score"] = model.predict_proba(X_scaled)[:, 1]

    # Rank order
    scored = (
        features[["client_id", "target", "propensity_score"]]
        .sort_values("propensity_score", ascending=False)
        .reset_index(drop=True)
    )
    scored["rank"] = range(1, len(scored) + 1)
    scored["decile"] = pd.qcut(scored["propensity_score"], 10,
                               labels=False, duplicates="drop") + 1

    print(f"\nScored {len(scored):,} clients")
    print("\nTop 10 prospects:")
    print(scored.head(10).to_string(index=False))

    return scored


# ==============================================================================
# MAIN — RUN THE PIPELINE
# ==============================================================================

def main(filepath: str):
    """End-to-end execution."""

    # 1. Load
    raw = load_data(filepath)

    # 2. Features
    features = build_features(raw, target_product=TARGET_PRODUCT)

    # 3. Split
    X_train, X_test, y_train, y_test = prepare_splits(features, TARGET_PRODUCT)

    # 4. Train
    model, scaler = train_model(X_train, y_train)

    # 5. Evaluate
    y_prob = evaluate(model, scaler, X_test, y_test)

    # 6. Score
    scored = score_clients(features, model, scaler, TARGET_PRODUCT)

    # 7. Export
    scored.to_csv("propensity_scores.csv", index=False)
    print("\nScores exported to propensity_scores.csv")

    return scored


# --- Entry point ---
if __name__ == "__main__":
    # Replace with your actual file path
    scored = main("client_product_data.csv")


# ==============================================================================
# ENHANCEMENT ROADMAP (not in code above — add incrementally)
# ==============================================================================
#
# PRIORITY 1 — Data & Features
#   • Temporal features: balance trends over 3/6/12 months
#   • Interaction features: product pairs, balance ratios
#   • External enrichment: demographics, credit bureau, digital activity
#   • Segmented models: mass market vs. affluent vs. commercial
#
# PRIORITY 2 — Modeling
#   • Gradient boosted trees (XGBoost/LightGBM) for lift improvement
#   • Hyperparameter optimization (Optuna, Bayesian search)
#   • Stacked ensemble (logistic + GBM blend)
#   • Calibration (Platt scaling or isotonic regression)
#
# PRIORITY 3 — Governance & Fairness
#   • Fair lending variable audit (WoE, IV by protected class)
#   • Disparate impact testing at each score threshold
#   • SR 11-7 aligned documentation template
#   • Champion-challenger framework for model updates
#
# PRIORITY 4 — Operationalization
#   • Multi-product scoring loop → next-best-product matrix
#   • Score decay monitoring (PSI, CSI)
#   • Automated retraining triggers
#   • Integration with campaign management / CRM
# ==============================================================================
