"""End-to-end, generalized feature engineering pipeline for a transactional DataFrame.

Produces one entity/id-level feature table suitable for variable selection and
model evaluation, by passing a raw transactional DataFrame through three
layers, each building on the last:

  1. Sequence layer   -- describes where each id currently sits in its own
                          event sequence, and how likely the terminal event
                          is from there: current/previous/first state
                          (schema-stable one-hot), repeat-run length, state
                          change counts, and a leave-one-out population
                          hazard -- P(terminal event next | last k states),
                          with backoff, plus P(terminal event next | number
                          of prior observations). A low-order Markov model
                          over the non-terminal states supplies next-state
                          entropy/concentration when the state column has
                          enough cardinality for that to mean anything.
  2. Rollup layer     -- per-id transactional aggregations: activity/timing
                          (transaction count, inter-event gaps, recency,
                          distinct products), "flow" aggregates over
                          transaction-amount-like columns (sum/mean/std +
                          rolling-window sums), and "level" aggregates over
                          balance-like columns (last value + rolling-window
                          delta/pct-change, i.e. generalized "month over
                          month" style trend, without assuming calendar
                          cadence).
  3. Customer layer   -- most-recent-known-value passthrough of configured
                          static/context columns (e.g. business_type, lob,
                          current_balance), with schema-stable one-hot
                          encoding for categoricals.

Data shape this is tuned for
-----------------------------
Short histories -- single-digit observations per id, frequently just one --
where the modeled outcome is a binary terminal event that occurs AT MOST ONCE
and ends the observation window. That shape rules out most of what
long-sequence feature engineering reaches for. Occurrence counts of a
once-only event are degenerate (0/1, so a count column and its flag are the
same column). Per-id surprisal or perplexity averaged over one or two
transitions is sampling noise, not behavior. Order-3 contexts exist for a
minority of ids, and their availability is a proxy for transaction volume
rather than a signal in itself.

What survives at this shape is the id's CURRENT state plus a POPULATION
estimate of what usually follows it: the id supplies the context, the
population supplies the probability. That is why the hazard features are the
core of this layer and per-id sequence scoring is not.

Label leakage -- read this before building a modeling table
------------------------------------------------------------
When `terminal_event_states` is configured, presence of the terminal event in
an id's history IS the label. Three guards follow from that:

- Terminal-event rows are stripped from every layer before any feature is
  computed. Left in, they would separate the classes perfectly -- not only
  through `seq_last_state` but through `n_transactions` and the flow sums.
- `as_of` becomes required. The default of "each id's own latest observed
  transaction" is definitionally the event date for positives and the
  censoring date for negatives, so it encodes the label. Pass the per-id
  decision-time cutoff that the label was defined against.
- The `as_of` DISTRIBUTION should match across classes. Positives are cut at
  their event date; if negatives are all cut at end-of-window, the classes
  differ by calendar position and any feature with a time trend will look
  predictive. Sample negatives' cutoffs to mirror the positives' rather than
  taking end-of-window for all of them. This module cannot detect that for
  you -- it is a property of how the sample was drawn.

The population hazard is fit including each id's own outcome, so every per-id
hazard feature is scored leave-one-out: the id's own at-risk observations and
events are subtracted from the population counts before the rate is taken.
Without that subtraction a positive sitting in a rare context reads its own
outcome back out of the population rate -- leakage concentrated exactly where
the feature looks most useful.

Point-in-time correctness
--------------------------
Every feature in every layer is computed only from transactions at or before
a per-id `as_of` cutoff (see `build_feature_table`'s `as_of` parameter). With
no terminal event configured, leaving `as_of=None` defaults every id to its
own latest observed transaction -- correct for scoring the current population
"as of now" (e.g. live NBO scoring), though note it makes
`days_since_last_txn` identically zero. If the table will be evaluated
against a historical labeled outcome you MUST pass an `as_of` cutoff per id.

Known limitations
------------------
- Not vectorized for very large populations (pure-Python loops over
  sequences, inherited from markov_chain.py); fine at typical customer-level
  scale.
- The hazard's leave-one-out correction holds per-id contribution counts in
  memory (ids x contexts); fine at customer scale, revisit for very large
  populations or many context orders.
- One-hot encoding uses a `category_universe` you can pass explicitly and
  which is also returned in `FeatureEngineeringArtifacts` -- persist and
  reuse it when scoring new data, otherwise a category present in training
  but absent from a new batch (or vice versa) will silently shift the
  encoded schema. Unseen categories collapse into an always-present
  `__other__` column rather than adding one.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from utils.markov_chain import OrderModel, _build_sequences, fit_markov_chains
except ImportError:  # running as a script from within utils/ rather than as a package
    from markov_chain import OrderModel, _build_sequences, fit_markov_chains

State = Hashable
Context = Tuple[State, ...]
Counts = Tuple[int, int]  # (n_at_risk, n_events)

__all__ = [
    "FeatureEngineeringConfig",
    "FeatureEngineeringArtifacts",
    "HazardModel",
    "build_feature_table",
]


@dataclass
class FeatureEngineeringConfig:
    """Column-role mapping and knobs -- the only thing that changes per dataset."""

    id_col: str
    time_col: str
    event_col: str  # categorical column driving the sequence/Markov layer

    product_col: Optional[str] = None  # for distinct-product-count features

    flow_cols: Sequence[str] = field(default_factory=tuple)   # transaction amounts: sum/mean/std + rolling sums
    level_cols: Sequence[str] = field(default_factory=tuple)  # balances/snapshots: last value + rolling delta/pct-change

    categorical_passthrough_cols: Sequence[str] = field(default_factory=tuple)  # OHE'd, most-recent value per id
    numeric_passthrough_cols: Sequence[str] = field(default_factory=tuple)      # as-is, most-recent value per id

    # -- sequence layer ----------------------------------------------------
    # States that END the observation window (i.e. the modeled outcome).
    # Configuring these switches the layer into hazard mode: the rows are
    # stripped from every layer, `as_of` becomes required, and per-id hazard
    # features are scored leave-one-out. Leave empty for a descriptive table.
    terminal_event_states: Sequence[State] = field(default_factory=tuple)

    # Context lengths for the hazard, tried longest-first and backing off to
    # the next shortest (then to the population base rate) when a context has
    # too few at-risk observations. Orders past 2 are rarely reachable on
    # short histories, and their availability tracks volume, not behavior.
    context_orders: Sequence[int] = (1, 2)

    sequence_max_gap: Optional[pd.Timedelta] = None  # gap starting a new session; None = one session per id
    sequence_min_support: int = 25  # at-risk observations a context needs before its own rate is used
    runlen_cap: int = 12  # observation counts at/above this share one hazard bucket

    rolling_windows_days: Sequence[int] = (30, 90)

    category_universe: Optional[Dict[str, Sequence[Any]]] = None  # fixed OHE categories; inferred + returned if omitted


@dataclass
class FeatureEngineeringArtifacts:
    """Everything build_feature_table produces -- not just the table.

    Keeping the fitted hazard, Markov models, and category universe around
    (rather than only returning `table`) is what lets you reuse the exact
    same population definitions when scoring a new batch later, instead of
    silently refitting on whatever data happens to be passed in next time.
    `hazard` in particular should be passed back into `build_feature_table`
    when scoring: refitting it on a new batch would estimate the outcome rate
    from the batch being scored.
    """

    table: pd.DataFrame
    markov_models: Dict[int, OrderModel]
    hazard: Optional["HazardModel"]
    category_universe: Dict[str, List[Any]]
    feature_columns_by_layer: Dict[str, List[str]]


def _validate_config(df: pd.DataFrame, cfg: FeatureEngineeringConfig) -> None:
    required = [cfg.id_col, cfg.time_col, cfg.event_col]
    if cfg.product_col:
        required.append(cfg.product_col)
    optional_groups = [
        cfg.flow_cols, cfg.level_cols, cfg.categorical_passthrough_cols, cfg.numeric_passthrough_cols,
    ]
    all_cols = required + [c for group in optional_groups for c in group]
    missing = [c for c in dict.fromkeys(all_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in input dataframe: {missing}")


def _resolve_as_of(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    as_of: Optional[Union[str, pd.Timestamp, pd.Series, Dict[Hashable, Any]]],
) -> pd.Series:
    """Per-id cutoff timestamp. None -> each id's own latest observed transaction."""
    max_ts = df.groupby(id_col)[time_col].max()
    max_ts.index.name = id_col

    if as_of is None:
        return max_ts

    if isinstance(as_of, dict):
        as_of = pd.Series(as_of)

    if isinstance(as_of, pd.Series):
        cutoff = as_of.reindex(max_ts.index)
        cutoff.index.name = id_col
        cutoff = pd.to_datetime(cutoff)
        missing = cutoff.isna()
        if missing.any():
            warnings.warn(
                f"{int(missing.sum())} id(s) missing an explicit as_of value; "
                "falling back to their own latest observed transaction."
            )
            cutoff = cutoff.fillna(max_ts)
        return cutoff

    # scalar (str / datetime / Timestamp): same cutoff broadcast to every id
    cutoff_ts = pd.Timestamp(as_of)
    return pd.Series(cutoff_ts, index=max_ts.index)


def _apply_as_of(
    df: pd.DataFrame, cfg: FeatureEngineeringConfig, as_of: Any,
) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    work[cfg.time_col] = pd.to_datetime(work[cfg.time_col])

    cutoff = _resolve_as_of(work, cfg.id_col, cfg.time_col, as_of)
    cutoff_df = cutoff.rename("__as_of__").reset_index()

    merged = work.merge(cutoff_df, on=cfg.id_col, how="left")
    filtered = merged[merged[cfg.time_col] <= merged["__as_of__"]].drop(columns="__as_of__")
    return filtered.reset_index(drop=True), cutoff


def _as_of_last_value(
    df: pd.DataFrame, id_col: str, time_col: str, value_col: str, cutoff: pd.Series,
) -> pd.Series:
    """Each id's most recent non-null value of value_col at/before cutoff[id]. NaN if none exists."""
    cutoff_df = cutoff.rename("__cutoff__").reset_index()
    merged = df[[id_col, time_col, value_col]].merge(cutoff_df, on=id_col, how="inner")
    merged = merged[(merged[time_col] <= merged["__cutoff__"]) & merged[value_col].notna()]
    merged = merged.sort_values([id_col, time_col], kind="mergesort")
    last = merged.groupby(id_col, sort=False)[value_col].last()
    return last.reindex(cutoff.index)


# --------------------------------------------------------------------------
# Shared: schema-stable one-hot encoding
# --------------------------------------------------------------------------

def _one_hot(values: pd.Series, prefix: str, universe: Sequence[Any]) -> pd.DataFrame:
    """One column per `universe` entry, in order, plus an always-present `__other__`.

    Keeping `__other__` even when nothing lands in it is deliberate: the
    encoded schema then depends only on `universe`, so a training table and a
    scoring table built from the same universe have identical columns whether
    or not the new batch happens to contain an unseen category. NaN encodes as
    all zeros -- pair it with an explicit availability flag wherever the
    difference between "absent" and "none of the above" matters.
    """
    cols = list(universe) + ["__other__"]
    known = values.where(values.isin(universe) | values.isna(), other="__other__")
    return pd.DataFrame(
        {f"{prefix}_{cat}": (known == cat).astype(int) for cat in cols},
        index=values.index,
    )


# --------------------------------------------------------------------------
# Layer 1: sequence position + terminal-event hazard
# --------------------------------------------------------------------------

@dataclass
class HazardModel:
    """Population discrete-time hazard of the terminal event.

    `by_order[k][context]` and `by_runlen[n]` hold (n_at_risk, n_events) for
    every context observed in the population. The parallel `own_*` maps hold
    each id's own contribution to those same cells, which is what lets a
    per-id feature be scored leave-one-out. That subtraction is not a nicety:
    the hazard is fit on data that includes each id's own outcome, so a
    positive sitting in a thinly-populated context would otherwise read its
    own label back out of the population rate.

    Keep this alongside the feature table and reuse it when scoring a later
    batch. Ids absent from `own_*` (i.e. anyone not in the fitting population)
    simply subtract nothing, which is the correct behavior for new entities.
    """

    base: Counts
    by_order: Dict[int, Dict[Context, Counts]]
    by_runlen: Dict[int, Counts]
    own_base: Dict[Hashable, Counts]
    own_by_order: Dict[int, Dict[Tuple[Hashable, Context], Counts]]
    own_by_runlen: Dict[Tuple[Hashable, int], Counts]
    runlen_cap: int

    @property
    def base_rate(self) -> float:
        n, e = self.base
        return e / n if n else float("nan")

    def to_frame(self) -> pd.DataFrame:
        """Long-format view of the fitted hazard, for inspection/sanity checks."""
        rows: List[Dict[str, Any]] = []
        for n_obs, (n, e) in sorted(self.by_runlen.items()):
            rows.append({"kind": "runlen", "context": n_obs,
                         "n_at_risk": n, "n_events": e, "rate": e / n if n else np.nan})
        for order in sorted(self.by_order):
            for ctx, (n, e) in self.by_order[order].items():
                rows.append({"kind": f"order{order}", "context": "_".join(str(x) for x in ctx),
                             "n_at_risk": n, "n_events": e, "rate": e / n if n else np.nan})
        if not rows:
            return pd.DataFrame(columns=["kind", "context", "n_at_risk", "n_events", "rate"])
        return (
            pd.DataFrame(rows)
            .sort_values(["kind", "n_at_risk"], ascending=[True, False])
            .reset_index(drop=True)
        )


def _freeze(counter: Dict[Any, List[int]]) -> Dict[Any, Counts]:
    return {key: (val[0], val[1]) for key, val in counter.items()}


def _fit_hazard(
    sequences: List[Any], terminal: set, orders: Sequence[int], runlen_cap: int,
) -> HazardModel:
    """Count at-risk observations and terminal events per context, and per id.

    This is the standard person-period expansion: every observation before the
    terminal event is one at-risk row, carrying a 1 if the terminal event
    followed it and a 0 otherwise. An id that never reached the event
    contributes all of its observations as zeros -- including its last one,
    which is a known non-event rather than a censored row, because the
    observation window runs past it (that is what makes the id a negative).

    Dropping that final observation instead would remove only zeros, and only
    from the ids that never converted, biasing every rate upward by roughly
    one over the mean history length. On single-digit histories that is not a
    rounding error: it inflated the strongest context by ~40% in testing.

    The one case where these rows really are censored is a history truncated
    by the data pull rather than by the label window. Then the last row's
    outcome is genuinely unknown and counting it as a zero is optimistic --
    cut such ids out of the fitting population, or fit the hazard on an
    earlier window and pass it in prefit.
    """
    base = [0, 0]
    own_base: Dict[Hashable, List[int]] = defaultdict(lambda: [0, 0])
    by_runlen: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    own_by_runlen: Dict[Tuple[Hashable, int], List[int]] = defaultdict(lambda: [0, 0])
    by_order: Dict[int, Dict[Context, List[int]]] = {
        k: defaultdict(lambda: [0, 0]) for k in orders
    }
    own_by_order: Dict[int, Dict[Tuple[Hashable, Context], List[int]]] = {
        k: defaultdict(lambda: [0, 0]) for k in orders
    }

    for seq in sequences:
        states = seq.states
        for i in range(len(states)):
            if states[i] in terminal:
                break  # the window ended here; nothing after it is at risk
            event = 1 if i + 1 < len(states) and states[i + 1] in terminal else 0

            base[0] += 1
            base[1] += event
            own = own_base[seq.id_]
            own[0] += 1
            own[1] += event

            bucket = min(i + 1, runlen_cap)  # observations seen so far, capped
            cell = by_runlen[bucket]
            cell[0] += 1
            cell[1] += event
            own_cell = own_by_runlen[(seq.id_, bucket)]
            own_cell[0] += 1
            own_cell[1] += event

            for order in orders:
                if i + 1 < order:
                    continue
                ctx = tuple(states[i - order + 1 : i + 1])
                cell = by_order[order][ctx]
                cell[0] += 1
                cell[1] += event
                own_cell = own_by_order[order][(seq.id_, ctx)]
                own_cell[0] += 1
                own_cell[1] += event

    return HazardModel(
        base=(base[0], base[1]),
        by_order={k: _freeze(v) for k, v in by_order.items()},
        by_runlen=_freeze(by_runlen),
        own_base=_freeze(own_base),
        own_by_order={k: _freeze(v) for k, v in own_by_order.items()},
        own_by_runlen=_freeze(own_by_runlen),
        runlen_cap=runlen_cap,
    )


def _loo_rate(total: Optional[Counts], own: Counts, min_support: int) -> Optional[float]:
    """Population rate for one context with `own`'s contribution removed.

    None means the context is too thin to trust once the id's own rows are
    taken out -- the caller should back off to a shorter context.
    """
    if total is None:
        return None
    n_at_risk = total[0] - own[0]
    n_events = total[1] - own[1]
    if n_at_risk < min_support or n_at_risk <= 0:
        return None
    return n_events / n_at_risk


def _sequence_layer(
    full_df: pd.DataFrame,
    work_df: pd.DataFrame,
    cfg: FeatureEngineeringConfig,
    state_universe: Optional[Sequence[Any]],
    prefit_hazard: Optional[HazardModel] = None,
) -> Tuple[pd.DataFrame, Dict[int, OrderModel], Optional[HazardModel], List[Any]]:
    """Where each id currently sits, and how likely the terminal event is from there.

    `full_df` still contains the terminal-event rows and is used ONLY to fit
    the population hazard. `work_df` has them stripped and is the only frame
    any per-id feature is read from -- see the module docstring on leakage.
    """
    id_col, time_col, event_col = cfg.id_col, cfg.time_col, cfg.event_col
    all_ids = work_df[[id_col]].drop_duplicates().reset_index(drop=True)
    terminal = set(cfg.terminal_event_states)
    orders = sorted({int(k) for k in cfg.context_orders if int(k) >= 1}, reverse=True)

    hazard: Optional[HazardModel] = prefit_hazard
    if hazard is None and terminal and orders:
        fit_seqs = _build_sequences(
            full_df, id_col, time_col, event_col,
            cfg.sequence_max_gap, collapse_repeats=False, min_length=1,
        )
        if fit_seqs:
            hazard = _fit_hazard(fit_seqs, terminal, orders, cfg.runlen_cap)
        else:
            warnings.warn("no id had any usable history; hazard features skipped")

    # min_length=1 (not the default 2): an id with a single observation still
    # has a current state, and at this data shape that is a large share of the
    # population -- the default would drop them from the layer entirely.
    obs_seqs = _build_sequences(
        work_df, id_col, time_col, event_col,
        cfg.sequence_max_gap, collapse_repeats=False, min_length=1,
    )
    latest: Dict[Hashable, Any] = {}
    for seq in obs_seqs:
        current = latest.get(seq.id_)
        if current is None or seq.timestamps[-1] > current.timestamps[-1]:
            latest[seq.id_] = seq

    if state_universe is None:
        state_universe = sorted(work_df[event_col].dropna().unique().tolist(), key=str)

    models: Dict[int, OrderModel] = {}
    n_distinct = int(work_df[event_col].nunique(dropna=True))
    if n_distinct < 2:
        warnings.warn(
            f"'{event_col}' has {n_distinct} distinct non-terminal state(s), so the next-state "
            "distribution is constant and those features were skipped -- the hazard and "
            "run-length features carry the sequence signal at this cardinality."
        )
    else:
        try:
            models = fit_markov_chains(
                work_df, id_col, time_col, event_col,
                orders=sorted(orders), max_gap=cfg.sequence_max_gap,
                collapse_repeats=False, min_support=cfg.sequence_min_support,
            )
        except ValueError as exc:
            warnings.warn(f"next-state distribution features skipped: {exc}")

    rows: List[Dict[str, Any]] = []
    for id_, seq in latest.items():
        states, times = seq.states, seq.timestamps
        n_obs = len(states)

        run = 1  # how long the id has been sitting in its current state
        while run < n_obs and states[-1 - run] == states[-1]:
            run += 1

        row: Dict[str, Any] = {
            id_col: id_,
            "seq_n_distinct_states": len(set(states)),
            "seq_n_state_changes": sum(1 for a, b in zip(states, states[1:]) if a != b),
            "seq_repeat_run_len": run,
            "seq_days_in_last_state": (times[-1] - times[n_obs - run]).total_seconds() / 86400.0,
            "__last": states[-1],
            "__prev": states[-2] if n_obs >= 2 else np.nan,
            "__first": states[0],
        }

        if hazard is not None:
            base_rate = _loo_rate(hazard.base, hazard.own_base.get(id_, (0, 0)), 1)
            if base_rate is None:
                base_rate = hazard.base_rate

            bucket = min(n_obs, hazard.runlen_cap)
            by_runlen = _loo_rate(
                hazard.by_runlen.get(bucket),
                hazard.own_by_runlen.get((id_, bucket), (0, 0)),
                cfg.sequence_min_support,
            )
            row["seq_hazard_runlen"] = base_rate if by_runlen is None else by_runlen

            ctx_rate: Optional[float] = None
            ctx_order = 0
            for order in orders:  # longest context first, backing off when too thin
                if n_obs < order:
                    continue
                ctx = tuple(states[-order:])
                rate = _loo_rate(
                    hazard.by_order[order].get(ctx),
                    hazard.own_by_order[order].get((id_, ctx), (0, 0)),
                    cfg.sequence_min_support,
                )
                if rate is not None:
                    ctx_rate, ctx_order = rate, order
                    break
            row["seq_hazard_context"] = base_rate if ctx_rate is None else ctx_rate
            row["seq_hazard_context_order"] = ctx_order  # 0 = fell back to the base rate
            row["seq_hazard_lift"] = (
                row["seq_hazard_context"] / base_rate if base_rate else np.nan
            )

        for order in orders:  # same backoff for the next-state distribution
            model = models.get(order)
            if model is None or n_obs < order:
                continue
            dist = model.probs.get(tuple(states[-order:]))
            if not dist:
                continue
            row["seq_next_state_entropy_bits"] = model.entropy.get(tuple(states[-order:]), np.nan)
            row["seq_top_next_state_prob"] = max(dist.values())
            row["seq_next_state_order"] = order
            break

        rows.append(row)

    if not rows:
        return all_ids, models, hazard, list(state_universe)

    feats = pd.DataFrame(rows)
    feats["seq_has_prev_state"] = feats["__prev"].notna().astype(int)
    for prefix, src in (
        ("seq_last_state", "__last"), ("seq_prev_state", "__prev"), ("seq_first_state", "__first"),
    ):
        feats = pd.concat([feats, _one_hot(feats[src], prefix, state_universe)], axis=1)
    feats = feats.drop(columns=["__last", "__prev", "__first"])

    return all_ids.merge(feats, on=id_col, how="left"), models, hazard, list(state_universe)


# --------------------------------------------------------------------------
# Layer 2: transactional rollups
# --------------------------------------------------------------------------

def _rollup_activity_features(
    work_df: pd.DataFrame, cfg: FeatureEngineeringConfig, cutoff: pd.Series,
) -> pd.DataFrame:
    id_col, time_col = cfg.id_col, cfg.time_col

    out = work_df.groupby(id_col)[time_col].agg(
        n_transactions="count", first_txn_ts="min", last_txn_ts="max",
    ).reset_index()

    if cfg.product_col:
        nprod = work_df.groupby(id_col)[cfg.product_col].nunique().rename("n_distinct_products").reset_index()
        out = out.merge(nprod, on=id_col, how="left")

    sorted_df = work_df.sort_values([id_col, time_col], kind="mergesort").copy()
    sorted_df["__gap_days__"] = sorted_df.groupby(id_col)[time_col].diff().dt.total_seconds() / 86400.0
    gap_stats = sorted_df.groupby(id_col)["__gap_days__"].agg(
        avg_days_between_txn="mean", median_days_between_txn="median", std_days_between_txn="std",
    ).reset_index()
    out = out.merge(gap_stats, on=id_col, how="left")

    cutoff_df = cutoff.rename("__as_of__").reset_index()
    out = out.merge(cutoff_df, on=id_col, how="left")
    out["tenure_days"] = (out["last_txn_ts"] - out["first_txn_ts"]).dt.total_seconds() / 86400.0
    out["days_since_last_txn"] = (out["__as_of__"] - out["last_txn_ts"]).dt.total_seconds() / 86400.0
    out = out.drop(columns="__as_of__")

    return out


def _rollup_flow_features(
    work_df: pd.DataFrame, cfg: FeatureEngineeringConfig, cutoff: pd.Series,
) -> pd.DataFrame:
    id_col, time_col = cfg.id_col, cfg.time_col
    out = work_df[[id_col]].drop_duplicates().reset_index(drop=True)
    if not cfg.flow_cols:
        return out

    cutoff_df = cutoff.rename("__as_of__").reset_index()
    merged = work_df.merge(cutoff_df, on=id_col, how="left")

    new_zero_cols: List[str] = []
    for col in cfg.flow_cols:
        agg = work_df.groupby(id_col)[col].agg(
            **{f"{col}_sum": "sum", f"{col}_mean": "mean", f"{col}_std": "std",
               f"{col}_min": "min", f"{col}_max": "max"},
        ).reset_index()
        out = out.merge(agg, on=id_col, how="left")
        new_zero_cols.append(f"{col}_sum")

        for window in cfg.rolling_windows_days:
            lower = merged["__as_of__"] - pd.Timedelta(days=window)
            windowed = merged[(merged[time_col] > lower) & (merged[time_col] <= merged["__as_of__"])]
            wagg = windowed.groupby(id_col)[col].agg(
                **{f"{col}_sum_{window}d": "sum", f"{col}_count_{window}d": "count"},
            ).reset_index()
            out = out.merge(wagg, on=id_col, how="left")
            new_zero_cols.extend([f"{col}_sum_{window}d", f"{col}_count_{window}d"])

    # a window with zero qualifying transactions truly moved zero dollars/events -- 0, not missing
    out[new_zero_cols] = out[new_zero_cols].fillna(0.0)
    return out


def _rollup_level_features(
    work_df: pd.DataFrame, cfg: FeatureEngineeringConfig, cutoff: pd.Series,
) -> pd.DataFrame:
    id_col = cfg.id_col
    out = work_df[[id_col]].drop_duplicates().reset_index(drop=True)
    if not cfg.level_cols:
        return out

    for col in cfg.level_cols:
        end_val = _as_of_last_value(work_df, id_col, cfg.time_col, col, cutoff)
        stats = work_df.groupby(id_col)[col].agg(
            **{f"{col}_min": "min", f"{col}_max": "max", f"{col}_std": "std"},
        ).reset_index()
        stats[f"{col}_last"] = stats[id_col].map(end_val)
        out = out.merge(stats, on=id_col, how="left")

        for window in cfg.rolling_windows_days:
            window_cutoff = cutoff - pd.Timedelta(days=window)
            start_val = _as_of_last_value(work_df, id_col, cfg.time_col, col, window_cutoff)
            delta = end_val - start_val
            pct = (delta / start_val.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

            window_df = pd.DataFrame({
                id_col: cutoff.index,
                f"{col}_delta_{window}d": delta.to_numpy(),
                f"{col}_pct_change_{window}d": pct.to_numpy(),
            })
            out = out.merge(window_df, on=id_col, how="left")

    return out


# --------------------------------------------------------------------------
# Layer 3: customer-level passthrough + one-hot encoding
# --------------------------------------------------------------------------

def _customer_layer(
    work_df: pd.DataFrame, cfg: FeatureEngineeringConfig, cutoff: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]:
    id_col = cfg.id_col
    out = work_df[[id_col]].drop_duplicates().reset_index(drop=True)
    category_universe: Dict[str, List[Any]] = dict(cfg.category_universe or {})

    for col in cfg.numeric_passthrough_cols:
        vals = _as_of_last_value(work_df, id_col, cfg.time_col, col, cutoff)
        out[col] = out[id_col].map(vals)

    for col in cfg.categorical_passthrough_cols:
        vals = _as_of_last_value(work_df, id_col, cfg.time_col, col, cutoff)

        universe = category_universe.get(col)
        if universe is None:
            universe = sorted(vals.dropna().unique().tolist(), key=str)
            category_universe[col] = universe

        dummies = _one_hot(vals, col, universe)
        dummies = dummies.reset_index().rename(columns={dummies.index.name or "index": id_col})
        out = out.merge(dummies, on=id_col, how="left")
        ohe_cols = [c for c in dummies.columns if c != id_col]
        out[ohe_cols] = out[ohe_cols].fillna(0).astype(int)

    return out, category_universe


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_feature_table(
    df: pd.DataFrame,
    config: FeatureEngineeringConfig,
    as_of: Optional[Union[str, pd.Timestamp, pd.Series, Dict[Hashable, Any]]] = None,
    hazard: Optional[HazardModel] = None,
) -> FeatureEngineeringArtifacts:
    """Run the full sequence -> rollup -> customer pipeline and return the merged feature table.

    Parameters
    ----------
    df : raw transactional DataFrame, one row per (id, timestamp, event/...).
    config : column-role mapping and knobs, see FeatureEngineeringConfig.
    as_of : point-in-time cutoff -- a per-id Series/dict, or one global
        timestamp. REQUIRED when `config.terminal_event_states` is set,
        because the default ("each id's own latest observed transaction") is
        the event date for positives and the censoring date for negatives,
        which encodes the label. Pass the decision-time cutoff the label was
        defined against, and check that its distribution matches across
        classes -- see the module docstring.
    hazard : a `HazardModel` fit previously. Pass this when scoring a new
        batch, or when the design calls for fitting the hazard on a strictly
        earlier training window; omitted, it is fit from `df` itself.
    """
    _validate_config(df, config)
    id_col, time_col, event_col = config.id_col, config.time_col, config.event_col
    terminal = set(config.terminal_event_states)

    if terminal and as_of is None:
        raise ValueError(
            "as_of is required when terminal_event_states is configured: defaulting each id to "
            "its own latest transaction makes the cutoff the event date for positives and the "
            "censoring date for negatives, which encodes the label. Pass the per-id "
            "decision-time cutoff the label was defined against."
        )

    work_df, cutoff = _apply_as_of(df, config, as_of)
    if work_df.empty:
        raise ValueError("no transactions remain at/before the given as_of cutoff(s)")

    # The hazard is a POPULATION object and has to see outcomes, so it is fit
    # from the unfiltered history rather than from the per-id feature window.
    # Each id's own contribution is subtracted when the feature is read back
    # (see HazardModel); pass a prefit `hazard` when the design needs the
    # population estimated on a strictly earlier window instead.
    hazard_df = df.copy()
    hazard_df[time_col] = pd.to_datetime(hazard_df[time_col])

    if terminal:
        ordered = work_df.sort_values([id_col, time_col], kind="mergesort")
        is_terminal = ordered[event_col].isin(terminal)
        n_after = int(
            (is_terminal.groupby(ordered[id_col]).cumsum() - is_terminal.astype(int) > 0).sum()
        )
        if n_after:
            warnings.warn(
                f"{n_after} row(s) fall after a terminal event inside the same id's window. The "
                "window is supposed to END at the event, so this means either the cutoff is past "
                "it or the event is not actually once-only -- both make the label ambiguous."
            )

        before_ids = work_df[id_col].nunique()
        work_df = work_df[~work_df[event_col].isin(terminal)].reset_index(drop=True)
        if work_df.empty:
            raise ValueError(
                "every row at/before the cutoff is a terminal event; nothing to build features from"
            )
        dropped = before_ids - work_df[id_col].nunique()
        if dropped:
            warnings.warn(
                f"{dropped} id(s) had no history other than the terminal event and are absent "
                "from the feature table. They are not scoreable here, but dropping them changes "
                "the population -- handle them explicitly rather than letting them vanish."
            )
        cutoff = cutoff[cutoff.index.isin(set(work_df[id_col].unique()))]

    state_universe = (config.category_universe or {}).get(event_col)
    seq_feats, markov_models, hazard, state_universe = _sequence_layer(
        hazard_df, work_df, config, state_universe, prefit_hazard=hazard,
    )
    activity_feats = _rollup_activity_features(work_df, config, cutoff)
    flow_feats = _rollup_flow_features(work_df, config, cutoff)
    level_feats = _rollup_level_features(work_df, config, cutoff)
    cust_feats, category_universe = _customer_layer(work_df, config, cutoff)
    category_universe.setdefault(event_col, list(state_universe))

    table = activity_feats
    for frame in (flow_feats, level_feats, seq_feats, cust_feats):
        table = table.merge(frame, on=id_col, how="left")

    feature_columns_by_layer = {
        "activity": [c for c in activity_feats.columns if c != id_col],
        "flow": [c for c in flow_feats.columns if c != id_col],
        "level": [c for c in level_feats.columns if c != id_col],
        "sequence": [c for c in seq_feats.columns if c != id_col],
        "customer": [c for c in cust_feats.columns if c != id_col],
    }

    return FeatureEngineeringArtifacts(
        table=table,
        markov_models=markov_models,
        hazard=hazard,
        category_universe=category_universe,
        feature_columns_by_layer=feature_columns_by_layer,
    )


def _make_dummy_transactions(
    n_ids: int = 4_000, seed: int = 0,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic log matching the shape this module targets.

    Each id accumulates behavioral observations until it either hits the
    terminal event ("purchase", at most once, closing the window) or its
    window closes without one. History length is 1-20 observations with a
    median near 3, and the true hazard depends on the current state -- an
    offer_click is roughly ten times likelier to be followed by a purchase
    than a browse -- so there is a real signal for the layer to recover.

    Returns the transaction log, the binary label, and the per-id as_of
    cutoff. Negatives' cutoffs come from the same waiting-time process as the
    positives' event dates rather than a common end-of-window, so the two
    classes are not separated by calendar position alone.
    """
    rng = np.random.default_rng(seed)
    states = ["browse", "offer_view", "offer_click", "service_call", "statement_view"]
    transition = {
        "browse":         [0.35, 0.30, 0.10, 0.10, 0.15],
        "offer_view":     [0.25, 0.25, 0.30, 0.10, 0.10],
        "offer_click":    [0.20, 0.25, 0.25, 0.15, 0.15],
        "service_call":   [0.30, 0.15, 0.10, 0.30, 0.15],
        "statement_view": [0.35, 0.20, 0.10, 0.10, 0.25],
    }
    true_hazard = {  # P(terminal event next | current state)
        "browse": 0.02, "offer_view": 0.07, "offer_click": 0.22,
        "service_call": 0.04, "statement_view": 0.03,
    }
    products = ["brokerage", "retirement", "insurance", "trust"]
    business_types = ["retail", "commercial", "private_bank"]
    lobs = ["wealth", "consumer"]

    rows: List[Dict[str, Any]] = []
    labels: Dict[int, int] = {}
    cutoffs: Dict[int, pd.Timestamp] = {}
    base_day = pd.Timestamp("2026-01-01")

    for cust_id in range(n_ids):
        max_obs = int(np.clip(rng.geometric(0.22), 1, 20))
        stamp = base_day + pd.Timedelta(days=int(rng.integers(0, 120)))
        state = str(rng.choice(states))
        balance = float(rng.uniform(5_000, 50_000))
        business_type = str(rng.choice(business_types))
        lob = str(rng.choice(lobs))
        label = 0

        def emit(event: str, when: pd.Timestamp) -> None:
            rows.append({
                "customer_id": cust_id,
                "txn_ts": when,
                "event": event,
                "product": str(rng.choice(products)),
                "txn_amount": round(float(rng.uniform(10, 2_000)), 2),
                "balance": round(balance, 2),
                "business_type": business_type,
                "lob": lob,
            })

        for _ in range(max_obs):
            balance *= float(rng.uniform(0.95, 1.08))
            emit(state, stamp)
            stamp = stamp + pd.Timedelta(days=float(rng.uniform(1, 40)))
            if rng.random() < true_hazard[state]:
                emit("purchase", stamp)
                label = 1
                break
            state = str(rng.choice(states, p=transition[state]))

        labels[cust_id] = label
        # positives: the event date. negatives: where the next observation
        # would have landed -- same waiting-time process, so the cutoff
        # distribution does not itself separate the classes.
        cutoffs[cust_id] = stamp

    label_series = pd.Series(labels, name="label")
    label_series.index.name = "customer_id"
    cutoff_series = pd.Series(cutoffs, name="as_of")
    cutoff_series.index.name = "customer_id"
    return pd.DataFrame(rows), label_series, cutoff_series


def _auc(scores: pd.Series, labels: pd.Series) -> float:
    """Rank AUC (Mann-Whitney), NaN-tolerant -- used only for the demo's leakage check."""
    frame = pd.DataFrame({"s": scores, "y": labels}).dropna()
    n_pos = int((frame["y"] == 1).sum())
    n_neg = int((frame["y"] == 0).sum())
    if n_pos == 0 or n_neg == 0 or frame["s"].nunique() < 2:
        return float("nan")
    ranks = frame["s"].rank()
    return float((ranks[frame["y"] == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


if __name__ == "__main__":
    transactions, labels, as_of = _make_dummy_transactions(n_ids=4_000, seed=0)
    observed = transactions[transactions["event"] != "purchase"]
    per_id = observed.groupby("customer_id").size()
    print(f"dummy log: {len(transactions)} rows, {transactions['customer_id'].nunique()} customers")
    print(f"pre-event observations per id: median {per_id.median():.0f}, "
          f"min {per_id.min()}, max {per_id.max()}, event rate {labels.mean():.1%}\n")

    cfg = FeatureEngineeringConfig(
        id_col="customer_id",
        time_col="txn_ts",
        event_col="event",
        product_col="product",
        flow_cols=["txn_amount"],
        level_cols=["balance"],
        categorical_passthrough_cols=["business_type", "lob"],
        numeric_passthrough_cols=["balance"],
        terminal_event_states=("purchase",),
        context_orders=(1, 2),
        sequence_min_support=25,
        rolling_windows_days=(30, 90),
    )

    artifacts = build_feature_table(transactions, cfg, as_of=as_of)
    table = artifacts.table

    print(f"feature table shape: {table.shape}")
    print("columns by layer:")
    for layer, cols in artifacts.feature_columns_by_layer.items():
        print(f"  {layer}: {len(cols)} columns")

    print("\nfitted hazard (base rate "
          f"{artifacts.hazard.base_rate:.3f}) vs the rates the data was generated from:")
    truth = {"browse": 0.02, "offer_view": 0.07, "offer_click": 0.22,
             "service_call": 0.04, "statement_view": 0.03}
    fitted = artifacts.hazard.to_frame()
    order1 = fitted[fitted["kind"] == "order1"].copy()
    order1["true_rate"] = order1["context"].map(truth)
    print(order1.to_string(index=False))

    y = labels.reindex(table["customer_id"]).to_numpy()
    print("\nsingle-feature rank AUC vs label (>0.99 would mean a leak):")
    aucs = {
        col: _auc(table[col], pd.Series(y))
        for col in table.columns
        if col != "customer_id" and pd.api.types.is_numeric_dtype(table[col])
    }
    ranked = pd.Series(aucs).dropna().sort_values(ascending=False)
    print(ranked.head(8).round(3).to_string())
    print(f"  ... max over all {len(ranked)} numeric features: {ranked.max():.3f}")

    print("\nsequence-layer coverage (pct non-null):")
    seq_cols = [c for c in artifacts.feature_columns_by_layer["sequence"] if not c.startswith("seq_last_state_")]
    print((table[seq_cols].notna().mean() * 100).round(1).sort_values().head(8).to_string())
