"""
Lookalike prospecting for very rare targets.

    python lookalike.py config.yaml

Given one row per customer (aggregates + demographics) and a rare binary target
(customers who already bought the product), this pipeline:

  1. infers column types and builds a distance-ready representation (no label used)
  2. profiles the buyers: which features separate them from everyone else,
     validated against a permutation null so tiny buyer counts can't overfit
  3. optionally splits buyers into archetypes
  4. scores every non-buyer by similarity to buyers using two label-free engines
     (weighted Gower kNN distance, random-forest leaf buyer-density) and rank-blends them
  5. validates with leave-one-buyer-out: where would each real buyer have ranked
     if we hadn't known they bought?  -> historical lift per score band
  6. writes a lead list with plain-language reasons, a buyer profile, validation
     table, and (optionally) an interactive map for sales reps

Nothing supervised is fit: the only label-dependent decision is which features
get weight, and that decision is gated by the permutation null.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

OTHER, MISSING = "__other__", "__missing__"


# ----------------------------------------------------------------------------
# config / io
# ----------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    p = Path(d["path"])
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)
    if d.get("max_rows"):
        df = df.head(int(d["max_rows"]))
    df = df.drop(columns=[c for c in d.get("drop_cols", []) or [] if c in df.columns])
    df = df.reset_index(drop=True)
    if df[d["id_col"]].duplicated().any():
        raise ValueError(f"{d['id_col']} must be unique: one row per customer expected")
    df[d["target_col"]] = df[d["target_col"]].fillna(0).astype(int)
    return df


def log(msg: str) -> None:
    print(f"[lookalike] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 1. type inference + preprocessing  (label-free except high-card encoding,
#    which is shrunk hard and never a strong feature on its own)
# ----------------------------------------------------------------------------
def infer_types(df: pd.DataFrame, cfg: dict) -> dict[str, str]:
    d, p = cfg["data"], cfg["preprocess"]
    skip = {d["id_col"], d["target_col"], d.get("eligible_col"), d.get("rep_col")}
    types = {}
    for c in df.columns:
        if c in skip:
            continue
        if c in d.get("type_overrides", {}) or {}:
            types[c] = d["type_overrides"][c]
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            types[c] = "categorical"
        elif pd.api.types.is_numeric_dtype(s):
            types[c] = "categorical" if s.nunique(dropna=True) <= p["categorical_max_unique"] else "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            types[c] = "numeric"
        else:
            types[c] = "categorical"
    return types


class Preprocessor:
    """Turns a mixed-type frame into (num in [0,1], cat integer codes)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.p = cfg["preprocess"]
        self.types: dict[str, str] = {}
        self.feature_info: dict[str, dict] = {}   # feature -> {kind, source}
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._cat_levels: dict[str, list] = {}
        self._hc_maps: dict[str, dict] = {}
        self._base_rate = 0.0

    def fit_transform(self, df: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        p = self.p
        self.types = infer_types(df, self.cfg)
        self._base_rate = y.mean()
        num, cat = {}, {}
        for c, t in self.types.items():
            s = df[c]
            miss = s.isna()
            if miss.mean() >= p["missing_indicator_min_rate"]:
                cat[f"{c}__missing"] = miss.astype(int).values
                self.feature_info[f"{c}__missing"] = {"kind": "categorical", "source": c, "levels": ["present", "missing"]}
            if t == "numeric":
                if pd.api.types.is_datetime64_any_dtype(s):
                    s = (s.max() - s).dt.days
                v = pd.to_numeric(s, errors="coerce").astype(float)
                if p["rank_transform_numeric"]:
                    r = rankdata(v.fillna(v.median()).values, method="average") / len(v)
                else:
                    lo, hi = v.min(), v.max()
                    r = ((v - lo) / (hi - lo if hi > lo else 1)).fillna(0.5).values
                num[c] = r
                self.feature_info[c] = {"kind": "numeric", "source": c}
            else:
                sv = s.astype(object).where(~miss, MISSING).astype(str)
                nlev = sv.nunique()
                if nlev > p["high_cardinality_threshold"]:
                    # shrunken buyer-rate encoding -> numeric
                    # leave-one-out: each row's own label is excluded from its level's rate,
                    # otherwise buyers' rows would carry their own label into the feature
                    g = pd.DataFrame({"l": sv.values, "y": y}).groupby("l")["y"].agg(["sum", "count"])
                    m = p["high_cardinality_shrinkage"]
                    lsum, lcnt = sv.map(g["sum"]).values - y, sv.map(g["count"]).values - 1
                    v = (lsum + m * self._base_rate) / (lcnt + m)
                    self._hc_maps[c] = ((g["sum"] + m * self._base_rate) / (g["count"] + m)).to_dict()
                    num[f"{c}__rate"] = rankdata(v, method="average") / len(v)
                    self.feature_info[f"{c}__rate"] = {"kind": "numeric", "source": c, "note": "buyer-rate encoded"}
                else:
                    vc = sv.value_counts()
                    keep = vc[vc >= p["min_level_count"]].index.tolist()
                    sv = sv.where(sv.isin(keep), OTHER)
                    levels = sorted(sv.unique().tolist())
                    self._cat_levels[c] = levels
                    cat[c] = pd.Categorical(sv, categories=levels).codes.astype(int)
                    self.feature_info[c] = {"kind": "categorical", "source": c, "levels": levels}
        num_df = pd.DataFrame(num, index=df.index)
        cat_df = pd.DataFrame(cat, index=df.index)
        self._num_cols, self._cat_cols = list(num_df.columns), list(cat_df.columns)
        return num_df, cat_df


# ----------------------------------------------------------------------------
# 2. profiling with a permutation null
# ----------------------------------------------------------------------------
class Profiler:
    """
    Separation statistic per feature for an arbitrary buyer index set, computed
    from precomputed ranks / codes so the permutation null and leave-one-out are cheap.
      numeric:     KS distance buyers vs population (default) or |2*AUC - 1|
      categorical: max over levels of |shrunken log-odds(level | buyer) vs population|
    """

    def __init__(self, num: pd.DataFrame, cat: pd.DataFrame, cfg: dict, info: dict):
        self.cfg, self.info = cfg["profile"], info
        self.n = len(num)
        self.num_cols, self.cat_cols = list(num.columns), list(cat.columns)
        self.ranks = {c: rankdata(num[c].values, method="average") for c in self.num_cols}
        self.codes = {c: cat[c].values for c in self.cat_cols}
        self.pop_share = {c: np.bincount(self.codes[c]) / self.n for c in self.cat_cols}
        self.rng = np.random.default_rng(cfg["similarity"]["random_state"])

    def stats(self, idx: np.ndarray) -> dict[str, float]:
        b = len(idx)
        out = {}
        ks = self.cfg.get("numeric_stat", "ks") == "ks"
        grid = np.arange(1, b + 1) / b
        for c in self.num_cols:
            r = self.ranks[c][idx]
            if ks:  # KS distance between buyer ECDF and population ECDF: catches shape, not just shift
                rs = np.sort(r) / self.n
                out[c] = float(max((grid - rs).max(), (rs - grid + 1 / b).max()))
            else:
                auc = (r.mean() - (b + 1) / 2) / (self.n - b)  # Mann-Whitney AUC
                out[c] = abs(2 * auc - 1)
        m = self.cfg["categorical_shrinkage"]
        for c in self.cat_cols:
            cnt = np.bincount(self.codes[c][idx], minlength=len(self.pop_share[c]))
            share = (cnt + m * self.pop_share[c]) / (b + m)
            pop = np.clip(self.pop_share[c], 1e-6, 1 - 1e-6)
            share = np.clip(share, 1e-6, 1 - 1e-6)
            lo = np.log(share / (1 - share)) - np.log(pop / (1 - pop))
            out[c] = float(np.abs(lo).max())
        return out

    def null(self, b: int) -> dict[str, float]:
        q, draws = self.cfg["null_quantile"], []
        for _ in range(self.cfg["n_permutations"]):
            draws.append(self.stats(self.rng.choice(self.n, size=b, replace=False)))
        d = pd.DataFrame(draws)
        return d.quantile(q).to_dict()

    def weights(self, idx: np.ndarray, null_q: dict[str, float]) -> pd.Series:
        s = pd.Series(self.stats(idx))
        excl = set(self.cfg.get("exclude_features", []) or [])
        excess = (s - pd.Series(null_q)).clip(lower=0)
        excess[[f for f in excess.index if f in excl or self.info[f]["source"] in excl]] = 0
        w = excess.sort_values(ascending=False)
        w = w.iloc[: self.cfg["max_features"]]
        w = w[w > 0]
        if len(w) < self.cfg["min_features"]:
            ratio = (s / pd.Series(null_q).replace(0, np.nan)).fillna(0)
            ratio = ratio[[f for f in ratio.index if f not in excl]]
            w = ratio.sort_values(ascending=False).iloc[: self.cfg["min_features"]]
        return w / w.sum()


def describe_buyers(df: pd.DataFrame, buyers: np.ndarray, weights: pd.Series, stats: dict,
                    null_q: dict, info: dict) -> pd.DataFrame:
    """Human-readable buyer profile for the surviving features."""
    rows = []
    for f, w in weights.items():
        src = info[f]["source"]
        raw = df[src]
        if info[f]["kind"] == "numeric" and pd.api.types.is_numeric_dtype(raw):
            b, p = raw.iloc[buyers].dropna(), raw.dropna()
            rows.append(dict(feature=f, weight=w, separation=stats[f], null_threshold=null_q.get(f),
                             direction="higher" if b.median() > p.median() else "lower",
                             buyers_typical=f"{fmt(b.quantile(.25))} to {fmt(b.quantile(.75))}",
                             everyone_typical=f"{fmt(p.quantile(.25))} to {fmt(p.quantile(.75))}"))
        else:
            if f.endswith("__missing"):
                b_top = f"missing {raw.iloc[buyers].isna().mean():.0%}"
                p_top = f"missing {raw.isna().mean():.0%}"
            else:
                bs, ps = raw.iloc[buyers].astype(str).value_counts(normalize=True), raw.astype(str).value_counts(normalize=True)
                lift = ((bs / ps.reindex(bs.index).fillna(1e-6))).sort_values(ascending=False)
                top = lift.index[0]
                b_top, p_top = f"{top}: {bs[top]:.0%} of buyers", f"{top}: {ps.get(top, 0):.0%} of everyone"
            rows.append(dict(feature=f, weight=w, separation=stats[f], null_threshold=null_q.get(f),
                             direction="over-represented", buyers_typical=b_top, everyone_typical=p_top))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. archetypes (optional)
# ----------------------------------------------------------------------------
def buyer_archetypes(Xb: np.ndarray, cfg: dict, seed: int) -> np.ndarray:
    a = cfg["archetypes"]
    if not a["enabled"] or len(Xb) < a["min_buyers"]:
        return np.zeros(len(Xb), dtype=int)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best_k, best_s, best_lab = 1, -1, np.zeros(len(Xb), dtype=int)
    for k in range(2, a["max_k"] + 1):
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Xb)
        if np.bincount(lab).min() < 5:
            continue
        s = silhouette_score(Xb, lab)
        if s > best_s:
            best_k, best_s, best_lab = k, s, lab
    if best_s < a["min_silhouette"]:
        return np.zeros(len(Xb), dtype=int)
    log(f"buyers split into {best_k} archetypes (silhouette {best_s:.2f})")
    return best_lab


# ----------------------------------------------------------------------------
# 4a. weighted Gower kNN similarity
# ----------------------------------------------------------------------------
class GowerScorer:
    def __init__(self, num: pd.DataFrame, cat: pd.DataFrame, weights: pd.Series, k: int, chunk: int):
        self.w = weights
        self.num_f = [f for f in weights.index if f in num.columns]
        self.cat_f = [f for f in weights.index if f in cat.columns]
        self.X = {f: num[f].values.astype(np.float32) for f in self.num_f}
        self.X.update({f: cat[f].values for f in self.cat_f})
        self.k, self.chunk = k, chunk

    def _dist_block(self, rows: np.ndarray, buyers: np.ndarray, by_feature: bool = False):
        D = np.zeros((len(rows), len(buyers)), dtype=np.float32)
        per = {}
        for f in self.num_f:
            d = np.abs(self.X[f][rows][:, None] - self.X[f][buyers][None, :])
            D += self.w[f] * d
            if by_feature:
                per[f] = d
        for f in self.cat_f:
            d = (self.X[f][rows][:, None] != self.X[f][buyers][None, :]).astype(np.float32)
            D += self.w[f] * d
            if by_feature:
                per[f] = d
        return (D, per) if by_feature else D

    def score(self, rows: np.ndarray, buyers: np.ndarray, exclude_self: bool = False):
        """Returns similarity (1 - mean distance to k nearest buyers) and nearest-buyer positions."""
        k = min(self.k, len(buyers) - (1 if exclude_self else 0))
        sims, nn = np.empty(len(rows), np.float32), np.empty((len(rows), k), dtype=int)
        for s in range(0, len(rows), self.chunk):
            r = rows[s:s + self.chunk]
            D = self._dist_block(r, buyers)
            if exclude_self:  # rows are themselves buyers: mask the diagonal
                pos = {b: i for i, b in enumerate(buyers)}
                for i, ri in enumerate(r):
                    if ri in pos:
                        D[i, pos[ri]] = np.inf
            part = np.argpartition(D, k - 1, axis=1)[:, :k]
            dk = np.take_along_axis(D, part, axis=1)
            order = np.argsort(dk, axis=1)
            nn[s:s + len(r)] = np.take_along_axis(part, order, axis=1)
            sims[s:s + len(r)] = 1 - np.take_along_axis(dk, order, axis=1).mean(axis=1)
        return sims, nn

    def feature_closeness(self, rows: np.ndarray, buyers: np.ndarray, nn: np.ndarray) -> pd.DataFrame:
        """Weighted per-feature closeness of each row to its own nearest buyers (for reasons)."""
        _, per = self._dist_block(rows, buyers, by_feature=True)
        out = {}
        for f, d in per.items():
            dk = np.take_along_axis(d, nn, axis=1).mean(axis=1)
            out[f] = self.w[f] * (1 - dk)
        return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# 4b. random-forest leaf buyer-density (label-free partition)
# ----------------------------------------------------------------------------
class RFDensityScorer:
    """
    Fit a forest to tell real rows from column-shuffled fakes (learns the joint
    structure of the data, no label).  Each leaf is a region of customer space;
    score = mean over trees of (buyers in leaf / customers in leaf) / base rate.
    """

    def __init__(self, num: pd.DataFrame, cat: pd.DataFrame, weights: pd.Series, cfg: dict, seed: int):
        feats = [f for f in weights.index]
        X = pd.concat([num[[f for f in feats if f in num.columns]],
                       cat[[f for f in feats if f in cat.columns]].astype(np.float32)], axis=1)
        self.X = X[feats].values.astype(np.float32)
        rng = np.random.default_rng(seed)
        n_fit = min(len(X), 100_000)
        real = self.X[rng.choice(len(X), n_fit, replace=False)]
        fake = np.column_stack([rng.permutation(real[:, j]) for j in range(real.shape[1])])
        r = cfg["rf_density"]
        self.forest = RandomForestClassifier(
            n_estimators=r["n_trees"], min_samples_leaf=r["min_leaf"], max_features=r["max_features"],
            n_jobs=-1, random_state=seed).fit(np.vstack([real, fake]), np.r_[np.ones(n_fit), np.zeros(n_fit)])
        self.chunk = cfg["chunk_rows"]
        self.leaf_total: list[np.ndarray] = []
        self.leaf_buyers: list[np.ndarray] = []

    def fit_counts(self, buyers: np.ndarray, base_rate: float):
        T = len(self.forest.estimators_)
        n_nodes = [t.tree_.node_count for t in self.forest.estimators_]
        self.leaf_total = [np.zeros(n, dtype=np.int64) for n in n_nodes]
        for s in range(0, len(self.X), self.chunk):
            L = self.forest.apply(self.X[s:s + self.chunk])
            for t in range(T):
                self.leaf_total[t] += np.bincount(L[:, t], minlength=n_nodes[t])
        Lb = self.forest.apply(self.X[buyers])
        self.leaf_buyers = [np.bincount(Lb[:, t], minlength=n_nodes[t]) for t in range(T)]
        self.base_rate = base_rate

    def score(self, rows: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        out = np.empty(len(rows), np.float32)
        for s in range(0, len(rows), self.chunk):
            L = self.forest.apply(self.X[rows[s:s + self.chunk]])
            acc = np.zeros(len(L), np.float64)
            for t in range(L.shape[1]):
                b, n = self.leaf_buyers[t][L[:, t]].astype(float), self.leaf_total[t][L[:, t]].astype(float)
                if exclude_self:
                    b, n = b - 1, n - 1
                acc += b / np.maximum(n, 1)
            out[s:s + len(L)] = acc / L.shape[1] / self.base_rate
        return out


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def pct_rank(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average") / len(x)


def blend(scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    tot = sum(weights[e] for e in scores)
    return sum(weights[e] * pct_rank(v) for e, v in scores.items()) / tot


def band_label(pct: float, labels: dict) -> str:
    for cut in sorted(labels):
        if pct >= 1 - float(cut):
            return labels[cut]
    return "below top 10%"


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.3g}"
    return str(v)


def make_reasons(lead_rows: np.ndarray, closeness: pd.DataFrame, df: pd.DataFrame, buyers: np.ndarray,
                 nn: np.ndarray, info: dict, n: int) -> list[str]:
    """Top-n features on which the prospect matches its OWN nearest buyers, in plain language."""
    reasons, vals = [], closeness.values
    for i, r in enumerate(lead_rows):
        nb = buyers[nn[i]]
        parts = []
        for j in np.argsort(-vals[i])[:n]:
            f = closeness.columns[j]
            src, raw = info[f]["source"], df[info[f]["source"]]
            v = raw.iloc[r]
            if f.endswith("__missing"):
                parts.append(f"{src} is {'missing' if pd.isna(v) else 'present'}, like similar buyers")
            elif info[f]["kind"] == "numeric" and pd.api.types.is_numeric_dtype(raw):
                b = raw.iloc[nb].dropna()
                parts.append(f"{src} = {fmt(v)} (similar buyers: {fmt(b.min())} to {fmt(b.max())})")
            else:
                same = (raw.iloc[nb].astype(str) == str(v)).sum()
                parts.append(f"{src} = {fmt(v)} (same as {same} of {len(nb)} most similar buyers)")
        reasons.append(" | ".join(parts))
    return reasons


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    d, sim_cfg, val_cfg, out_cfg = cfg["data"], cfg["similarity"], cfg["validation"], cfg["output"]
    seed = sim_cfg["random_state"]
    rng = np.random.default_rng(seed)
    out_dir = Path(out_cfg["dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(cfg)
    y = df[d["target_col"]].values
    buyers = np.flatnonzero(y == 1)
    base_rate = len(buyers) / len(df)
    log(f"{len(df):,} customers, {len(buyers)} buyers (base rate {base_rate:.5%})")
    if len(buyers) < 5:
        raise ValueError("Need at least 5 buyers to profile against.")

    # --- 1. representation
    pre = Preprocessor(cfg)
    num, cat = pre.fit_transform(df, y)
    log(f"{num.shape[1]} numeric + {cat.shape[1]} categorical features after preprocessing")

    # --- 2. profile with permutation null
    prof = Profiler(num, cat, cfg, pre.feature_info)
    null_q = prof.null(len(buyers))
    weights = prof.weights(buyers, null_q)
    stats = prof.stats(buyers)
    profile = describe_buyers(df, buyers, weights, stats, null_q, pre.feature_info)
    profile.to_csv(out_dir / "buyer_profile.csv", index=False)
    log(f"{len(weights)} indicative features: {', '.join(weights.index)}")

    # --- 3. archetypes on the weighted feature matrix of buyers
    feat_mat = build_feature_matrix(num, cat, weights, pre.feature_info)
    arche = buyer_archetypes(feat_mat[buyers], cfg, seed)

    # --- 4. similarity engines over the scoring population
    elig = np.ones(len(df), bool) if not d.get("eligible_col") else df[d["eligible_col"]].fillna(0).astype(bool).values
    prospects = np.flatnonzero((y == 0) & elig)
    engines, scores = sim_cfg["engines"], {}
    gower = GowerScorer(num, cat, weights, sim_cfg["k_neighbors"], sim_cfg["chunk_rows"])
    g_sim, g_nn = gower.score(prospects, buyers)
    if "gower" in engines:
        scores["gower"] = g_sim
    rf = None
    if "rf_density" in engines:
        rf = RFDensityScorer(num, cat, weights, sim_cfg, seed)
        rf.fit_counts(buyers, base_rate)
        scores["rf_density"] = rf.score(prospects)
    final = blend(scores, sim_cfg["blend_weights"])
    pct = pct_rank(final)
    log("scored %d prospects with engines: %s" % (len(prospects), ", ".join(scores)))

    # --- 5. leave-one-buyer-out validation
    validation, loo_pct = pd.DataFrame(), None
    if val_cfg["leave_one_out"]:
        loo_pct = leave_one_out(prof, null_q, num, cat, buyers, prospects, rf, cfg, rng)
        validation = lift_table(loo_pct, val_cfg["bands"], val_cfg["n_bootstrap"], rng)
        validation.to_csv(out_dir / "validation_lift.csv", index=False)
        log("leave-one-out lift by band:\n" + validation.to_string(index=False))

    # --- 6. lead list
    order = np.argsort(-final)[: out_cfg["top_n_leads"]]
    lead_rows = prospects[order]
    closeness = gower.feature_closeness(lead_rows, buyers, g_nn[order])
    reasons = make_reasons(lead_rows, closeness, df, buyers, g_nn[order], pre.feature_info, out_cfg["n_reasons"])
    nn_arche = arche[g_nn[order]]
    leads = pd.DataFrame({
        d["id_col"]: df[d["id_col"]].values[lead_rows],
        "score_percentile": np.round(pct[order] * 100, 2),
        "band": [band_label(p, out_cfg["band_labels"]) for p in pct[order]],
        "archetype": [f"type {int(np.bincount(a).argmax()) + 1}" for a in nn_arche] if arche.max() > 0 else "all buyers",
        "nearest_buyer_id": df[d["id_col"]].values[buyers[g_nn[order, 0]]],
        "why_similar": reasons,
    })
    if d.get("rep_col"):
        leads.insert(1, d["rep_col"], df[d["rep_col"]].values[lead_rows])
    if not validation.empty:
        lift_map = dict(zip(validation["band"], validation["lift"]))
        leads["historical_lift"] = leads["band"].map(lambda b: lift_map.get(b.split(" - ")[0], np.nan) if " - " in b else np.nan)
    leads.to_csv(out_dir / "leads.csv", index=False)

    summary = dict(customers=int(len(df)), buyers=int(len(buyers)), base_rate=base_rate,
                   prospects_scored=int(len(prospects)), features_used=weights.round(4).to_dict(),
                   archetypes=int(arche.max() + 1), engines=list(scores),
                   loo_median_percentile=float(np.median(loo_pct)) if loo_pct is not None else None)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # --- 7. map for reps
    if cfg.get("map", {}).get("enabled"):
        from prospect_map import build_map
        # raw source columns behind the indicative features, in weight order (deduped)
        profile_fields = list(dict.fromkeys(pre.feature_info[f]["source"] for f in weights.index))
        build_map(cfg, df, feat_mat, buyers, arche, lead_rows, leads, prospects, out_dir / "prospect_map.html", rng,
                  profile_fields, pre.types)
        log(f"map written to {out_dir / 'prospect_map.html'}")

    log(f"done. outputs in {out_dir}/")
    return summary


def build_feature_matrix(num: pd.DataFrame, cat: pd.DataFrame, weights: pd.Series, info: dict) -> np.ndarray:
    """Weighted numeric matrix (one-hot categoricals) used for clustering and the map."""
    cols = []
    for f, w in weights.items():
        if f in num.columns:
            cols.append(np.sqrt(w) * num[f].values[:, None])
        else:
            codes = cat[f].values
            oh = np.eye(codes.max() + 1, dtype=np.float32)[codes]
            cols.append(np.sqrt(w / 2) * oh)   # mismatch on one-hot = sqrt(2); scale so it counts once
    return np.hstack(cols).astype(np.float32)


def leave_one_out(prof, null_q, num, cat, buyers, prospects, rf, cfg, rng) -> np.ndarray:
    """For each buyer: re-derive weights without them, score them vs remaining buyers,
    and return their percentile among a population sample scored the same way."""
    sim_cfg, val_cfg = cfg["similarity"], cfg["validation"]
    sample = rng.choice(prospects, size=min(val_cfg["population_sample"], len(prospects)), replace=False)
    pcts = np.empty(len(buyers))
    rf_sample = rf.score(sample) if rf is not None else None   # forest is label-free: score once
    for i, b in enumerate(buyers):
        others = np.delete(buyers, i)
        w = prof.weights(others, null_q)
        g = GowerScorer(num, cat, w, sim_cfg["k_neighbors"], sim_cfg["chunk_rows"])
        sc_pop, sc_b = {}, {}
        if "gower" in sim_cfg["engines"]:
            sc_pop["gower"] = g.score(sample, others)[0]
            sc_b["gower"] = g.score(np.array([b]), others)[0]
        if rf is not None:
            # forest is label-free; only the leaf buyer counts change (drop self)
            sc_pop["rf_density"] = rf_sample
            sc_b["rf_density"] = rf.score(np.array([b]), exclude_self=True)
        # blend held-out buyer into the sample's rank space
        tot, bl = 0.0, 0.0
        for e in sc_pop:
            wt = sim_cfg["blend_weights"][e]
            bl += wt * (sc_pop[e] < sc_b[e][0]).mean()
            tot += wt
        pcts[i] = bl / tot
    return pcts


def lift_table(loo_pct: np.ndarray, bands: list, n_boot: int, rng) -> pd.DataFrame:
    rows = []
    for b in bands:
        hit = (loo_pct >= 1 - b).astype(float)
        boots = [rng.choice(hit, len(hit), replace=True).mean() / b for _ in range(n_boot)]
        letters = {0.01: "A", 0.02: "B", 0.05: "C", 0.10: "D"}  # join keys for the lead list's band label
        rows.append(dict(band=letters.get(b, f"top {b:.1%}".replace(".0%", "%")),
                         band_desc=f"top {b:.1%}".replace(".0%", "%"),
                         buyers_captured=f"{hit.mean():.0%}",
                         lift=round(hit.mean() / b, 1), lift_ci_low=round(float(np.percentile(boots, 2.5)), 1),
                         lift_ci_high=round(float(np.percentile(boots, 97.5)), 1)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run(load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
