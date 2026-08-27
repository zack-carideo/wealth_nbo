"""General-purpose feature engineering for transactional data.

Turns any log shaped as one row per (entity, time, ...) into one row per
entity. The only requirement is an id column and something that orders events
within an id; every other column role is optional and inferred when not given.

    cfg = FeatureEngineeringConfig(id_col="customer_id", time_col="txn_ts")
    artifacts = build_feature_table(df, cfg)

Column roles
------------
  static       constant within an entity (segment, region) -> last known value
  flow         numeric and additive (amount, quantity)     -> sum, mean, max, last
  level        numeric and persistent (balance, price)     -> last, min, max, change
  categorical  low-cardinality labels (channel, product)   -> variety, concentration, current

Flow and level are told apart by how much a column carries over between an
entity's own rows -- a balance is close to what it was last time, a payment
amount is not. Inference is a starting point: it is reported in
`artifacts.column_roles` and can be overridden per role in the config.

Irregular timing
----------------
Transactional data is rarely on a cadence, and gaps between an entity's rows
range from minutes to years within one table. Two consequences shape the
design.

Fixed calendar windows do not survive it. A 30-day window over a log whose
median gap is 500 days is empty for essentially every entity, and because a
windowed sum fills to 0.0 the resulting columns look like data rather than
absence. So windows over the last k EVENTS are the default: always defined,
identical in meaning at any cadence. Calendar windows remain available via
`recent_windows` and are worth turning on for dense logs, but they are off
unless asked for.

Timing features are reported both raw and relative to the entity's own rhythm.
`recency` in days cannot be compared between an entity that transacts weekly
and one that transacts every other year; `recency_ratio`, which divides it by
that entity's median gap, can. Same for `gap_burstiness`.

Time
----
`time_col` may be datetime-like or numeric. Datetimes are measured in days; a
numeric column (a sequence number, a reading index) is used as-is and every
duration is in those units. The unit is recorded in `artifacts.time_unit`.

Point-in-time correctness
-------------------------
Features come only from rows at or before an `as_of` cutoff, defaulting to one
cutoff for the whole population at the latest timestamp present. A per-entity
cutoff of each entity's own last row looks natural and is not offered, because
it forces recency to zero for everyone.

Cutoffs must also be DRAWN the same way for both classes. Cutting positives at
their outcome date and negatives at today is the obvious design and it leaks:
recency then runs to a recent date for one class and to a date years back for
the other, separating them without saying anything about who converts. Use
`matched_cutoffs`. Nothing downstream detects the problem, because it spreads
thinly across every timing feature instead of concentrating in one.

Optional: terminal-event hazard
-------------------------------
Off unless the outcome is declared, via `terminal_flag_col` (a column set only
on outcome rows -- the usual shape when outcomes arrive from their own table
carrying an id, a date and a flag) or `terminal_event_states` (values within
`event_col`). Declaring them is what strips those rows from every layer; an
undeclared outcome row is counted as an ordinary event and hands the label to
`n_events` and `recency`, so a declaration matching nothing is an error rather
than a silent no-op.

Dependencies are numpy and pandas, nothing else in this repository included, so
the file is portable as-is. The core is vectorized; only the hazard iterates.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "FeatureEngineeringConfig",
    "FeatureEngineeringArtifacts",
    "HazardModel",
    "infer_column_roles",
    "zero_diagnostic",
    "matched_cutoffs",
    "build_feature_table",
]

Context = Tuple[Any, ...]
Counts = Tuple[int, int]  # (n_at_risk, n_events)

_T, _CUT, _STATE = "__t__", "__cutoff__", "__state__"
_INTERNAL = (_T, _CUT, _STATE)
_TERMINAL, _OBSERVED = "__terminal__", "__observed__"
_SECONDS_PER_DAY = 86_400.0


@dataclass
class FeatureEngineeringConfig:
    """Column roles and knobs. Only `id_col` and `time_col` are required.

    A role left as None is inferred; pass a list (even an empty one) to pin it.
    """

    id_col: str
    time_col: str

    flow_cols: Optional[Sequence[str]] = None
    level_cols: Optional[Sequence[str]] = None
    categorical_cols: Optional[Sequence[str]] = None
    static_cols: Optional[Sequence[str]] = None
    ignore_cols: Sequence[str] = field(default_factory=tuple)

    # Aggregate windows over the last k EVENTS -- defined at any cadence, which
    # calendar windows are not. `recent_windows` adds calendar windows in the
    # time column's units; leave it empty unless the log is dense enough that
    # entities reliably have rows inside them.
    recent_events: Sequence[int] = (3,)
    recent_windows: Sequence[float] = ()

    # Columns where a literal 0 means "not applicable to this row" rather than a
    # real zero -- common when one table carries several row types and each row
    # fills in only the columns its type uses. Zeros become nulls before
    # anything runs. Use `zero_diagnostic` to decide which columns need it.
    zero_is_missing: Sequence[str] = field(default_factory=tuple)

    max_categories: int = 20        # one-hot width cap; the rest fold into __other__
    level_persistence: float = 0.15  # persistence above which a numeric column is a level
    infer_sample_rows: int = 200_000

    category_universe: Optional[Dict[str, List[Any]]] = None  # persist and reuse when scoring

    # -- optional terminal-event hazard; inert unless the outcome is declared
    event_col: Optional[str] = None
    terminal_flag_col: Optional[str] = None
    terminal_event_states: Sequence[Any] = field(default_factory=tuple)
    context_orders: Sequence[int] = (1, 2)
    hazard_min_support: int = 25
    runlen_cap: int = 12

    @property
    def declares_outcome(self) -> bool:
        return bool(self.terminal_flag_col or self.terminal_event_states)


@dataclass
class FeatureEngineeringArtifacts:
    """The table plus everything needed to rebuild it identically.

    Pass `column_roles`, `category_universe` and `hazard` back in when scoring a
    later batch, so the schema and the population estimates stay fixed instead
    of being re-derived from whatever data arrives next.
    """

    table: pd.DataFrame
    column_roles: Dict[str, List[str]]
    category_universe: Dict[str, List[Any]]
    feature_columns_by_layer: Dict[str, List[str]]
    time_unit: str
    hazard: Optional["HazardModel"] = None


# --------------------------------------------------------------------------
# Time axis
# --------------------------------------------------------------------------

def _to_axis(values: Any, unit: str) -> pd.Series:
    """Project timestamps or numbers onto the float axis named by `unit`.

    Days come from `total_seconds`, not from casting to int64 and dividing by a
    nanosecond constant: since pandas 2.0 a datetime column may be backed by
    second, millisecond, microsecond or nanosecond resolution, and the cast
    returns the underlying integer in whichever it is -- right for one frame and
    wrong by a factor of a thousand for the next, silently.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if unit != "day":
        return pd.to_numeric(series, errors="coerce").astype(float)

    stamps = pd.to_datetime(series, errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(stamps):
        stamps = pd.to_datetime(series, errors="coerce", utc=True)  # mixed offsets
    tz = getattr(stamps.dt, "tz", None)
    epoch = pd.Timestamp("1970-01-01", tz="UTC") if tz is not None else pd.Timestamp("1970-01-01")
    return (stamps - epoch).dt.total_seconds().where(stamps.notna()) / _SECONDS_PER_DAY


def _detect_time_unit(series: pd.Series) -> str:
    """"day" for anything datetime-like, "unit" for a numeric ordering column.

    A numeric column is taken at face value: an integer that happens to look
    like 20260101 is the number it is, and windows are then in those units.
    """
    if pd.api.types.is_numeric_dtype(series):
        return "unit"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "day"
    if pd.to_datetime(series, errors="coerce").notna().any():
        return "day"
    raise TypeError(
        f"time column is neither numeric nor parseable as datetime (dtype {series.dtype}); "
        "convert it to timestamps or to a numeric sequence order first"
    )


# --------------------------------------------------------------------------
# Column roles
# --------------------------------------------------------------------------

def _persistence(df: pd.DataFrame, id_col: str, col: str) -> float:
    """How much a numeric column carries over between an entity's own rows.

    `1 - MSSD / (2 * within-entity variance)`: the mean squared successive
    difference against twice the variance. For a series with no carry-over the
    two are equal and this is 0; for one that barely moves between rows the
    difference term vanishes and it approaches 1. Both halves are unbiased at
    small n, so unlike a demeaned lag-1 autocorrelation the value does not drift
    with history length -- an additive column sits at ~0.00 whether entities
    have three rows or twenty.

    Entities with fewer than three observations of the column are excluded:
    with two rows the single difference and the single deviation are the same
    number, so there is no information about persistence to extract. A column
    seen only on such entities returns 0.0 and is treated as a flow.
    """
    valid = df.loc[df[col].notna(), [id_col, col]]
    valid = valid[valid.groupby(id_col)[col].transform("size") >= 3]
    if len(valid) < 60 or valid[id_col].nunique() < 20:
        return 0.0

    grouped = valid.groupby(id_col, sort=False)[col]
    deviations = valid[col].astype(float) - grouped.transform("mean").astype(float)
    variance = float((deviations ** 2).sum()) / (len(valid) - valid[id_col].nunique())
    if variance <= 0:
        return 0.0
    mssd = float((grouped.diff().dropna().astype(float) ** 2).mean())
    return 1.0 - mssd / (2.0 * variance)


def infer_column_roles(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    max_categories: int = 20,
    level_persistence: float = 0.15,
    sample_rows: int = 200_000,
    ignore_cols: Sequence[str] = (),
) -> Dict[str, List[str]]:
    """Classify each remaining column as static, flow, level, categorical or skipped.

    The only non-obvious case is flow vs level. Both are numeric, so dtype
    cannot separate them and names are not reliable across datasets. What does
    separate them is persistence: a balance is close to what it was last time,
    a payment amount is not.

    Persistence is measured as 1 - MSSD / 2*variance, both taken within entity
    (see `_persistence`), rather than as a lag-1 autocorrelation of
    entity-demeaned values. Demeaning inside a short history biases that
    correlation down by roughly 1/(n-1) -- at two rows per entity it is exactly
    -1 whatever the column, and it still misreads a balance as a flow at six
    rows. On transactional data most entities have a handful of rows, so that
    bias is the common case, not an edge case.

    Columns constant within an entity are static whatever their dtype, and
    non-numeric columns with too many distinct values are skipped rather than
    encoded, since free text or an identifier would otherwise yield a feature
    per value.
    """
    reserved = {id_col, time_col, *ignore_cols}
    roles: Dict[str, List[str]] = {k: [] for k in
                                   ("static", "flow", "level", "categorical", "skipped")}
    candidates = [c for c in df.columns if c not in reserved]
    if not candidates:
        return roles

    sample = df
    if len(df) > sample_rows:  # sample whole entities, never rows within one
        ids = df[id_col].drop_duplicates()
        keep = ids.sample(n=max(1, int(len(ids) * sample_rows / len(df))), random_state=0)
        sample = df[df[id_col].isin(set(keep))]
    sample = sample.sort_values([id_col, time_col], kind="mergesort")
    grouped = sample.groupby(id_col, sort=False)

    for col in candidates:
        series = sample[col]
        if series.notna().sum() == 0 or series.nunique(dropna=True) <= 1:
            roles["skipped"].append(col)
        elif float(grouped[col].nunique(dropna=True).le(1).mean()) >= 0.95:
            roles["static"].append(col)  # a property of the entity, not of its activity
        elif pd.api.types.is_numeric_dtype(series):
            score = _persistence(sample, id_col, col)
            roles["level" if score >= level_persistence else "flow"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            roles["skipped"].append(col)  # a second time column needs a role you choose
        elif series.nunique(dropna=True) <= max(max_categories * 5, 100):
            roles["categorical"].append(col)
        else:
            roles["skipped"].append(col)

    return roles


def zero_diagnostic(df: pd.DataFrame, numeric_col: str, by: str, top: int = 8) -> pd.DataFrame:
    """Is a zero in `numeric_col` a real zero, or "not applicable to this row"?

    Breaks the zero rate down by another column, normally the one naming the
    row's type. Zeros concentrated in the types that do not use the column are
    structural nulls and belong in `zero_is_missing`; zeros spread evenly across
    every type are real measurements and should stay.
    """
    valid = df.loc[df[numeric_col].notna(), [numeric_col, by]]
    if valid.empty:
        return pd.DataFrame(columns=[by, "n_rows", "n_zero", "zero_rate", "nonzero_median"])
    grouped = valid.groupby(by, dropna=False)[numeric_col]
    out = pd.DataFrame({
        "n_rows": grouped.size(),
        "n_zero": grouped.apply(lambda x: int((x == 0).sum())),
        "nonzero_median": grouped.apply(lambda x: x[x != 0].median()),
    })
    out["zero_rate"] = out["n_zero"] / out["n_rows"]
    return (out.reset_index()[[by, "n_rows", "n_zero", "zero_rate", "nonzero_median"]]
            .sort_values("zero_rate", ascending=False).head(top).reset_index(drop=True))


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _one_hot(values: pd.Series, prefix: str, universe: Sequence[Any]) -> pd.DataFrame:
    """One column per `universe` entry plus an always-present `__other__`.

    `__other__` is emitted even when empty so the schema depends only on the
    universe: a training table and a scoring table built from the same universe
    have identical columns whether or not the new batch holds an unseen value.
    NaN encodes as all zeros.
    """
    known = values.where(values.isin(universe) | values.isna(), other="__other__")
    return pd.DataFrame(
        {f"{prefix}_{cat}": (known == cat).astype("int8") for cat in list(universe) + ["__other__"]},
        index=values.index,
    )


def _universe_for(values: pd.Series, col: str, universe: Dict[str, List[Any]], cap: int) -> List[Any]:
    """Fixed category list for `col`, keeping the most frequent when capped."""
    if col not in universe:
        universe[col] = sorted(values.value_counts().index[:cap].tolist(), key=str)
    return universe[col]


def _positional(work: pd.DataFrame, id_col: str, cols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """First / previous / last non-null value per entity. Rows must be sorted.

    `prev` comes from tail(2) rather than nth(-2): as of pandas 2.0 nth returns
    rows on the ORIGINAL index instead of one value per group, which misaligns
    silently when reindexed by entity.
    """
    out: Dict[str, pd.DataFrame] = {}
    for col in cols:
        valid = work.loc[work[col].notna(), [id_col, col]]
        grouped = valid.groupby(id_col, sort=False)[col]
        tail = valid.groupby(id_col, sort=False).tail(2).groupby(id_col, sort=False)[col]
        out[col] = pd.DataFrame({
            "first": grouped.first(), "last": grouped.last(),
            "prev": tail.first().where(tail.size() == 2),
        })
    return out


def _subsets(work: pd.DataFrame, cfg: FeatureEngineeringConfig) -> Tuple[Dict[str, pd.Series], List[str]]:
    """Masks naming each "recent" slice, plus which of them are calendar windows.

    Rows are already filtered to at-or-before the cutoff, so a calendar window
    only needs its lower bound.
    """
    masks: Dict[str, pd.Series] = {}
    if len(cfg.recent_events):
        rank = work.groupby(cfg.id_col, sort=False).cumcount(ascending=False)
        for k in cfg.recent_events:
            masks[f"last{int(k)}"] = rank < int(k)
    window_tags = [f"w{float(w):g}" for w in cfg.recent_windows]
    for tag, w in zip(window_tags, cfg.recent_windows):
        masks[tag] = work[_T] > work[_CUT] - float(w)
    return masks, window_tags


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------

def _activity_layer(
    work: pd.DataFrame, cfg: FeatureEngineeringConfig, cutoff: pd.Series,
    masks: Dict[str, pd.Series], window_tags: Sequence[str],
) -> pd.DataFrame:
    """Recency, frequency, tenure and rhythm -- the one family every log supports."""
    id_col = cfg.id_col
    grouped = work.groupby(id_col, sort=False)[_T]
    first, last = grouped.min(), grouped.max()

    out = pd.DataFrame({"n_events": grouped.size()})
    out["tenure"] = last - first
    out["recency"] = cutoff.reindex(out.index) - last

    gaps = work[_T].groupby(work[id_col], sort=False).diff().groupby(work[id_col], sort=False)
    out["gap_median"] = gaps.median()
    out["gap_last"] = gaps.last()
    out["events_per_unit"] = out["n_events"] / out["tenure"].replace(0, np.nan)

    # Raw durations are not comparable between an entity transacting weekly and
    # one transacting every other year. Dividing by the entity's own median gap
    # gives the same quantity in units of its own rhythm: >1 means overdue.
    typical = out["gap_median"].replace(0, np.nan)
    out["recency_ratio"] = out["recency"] / typical
    out["gap_burstiness"] = out["gap_last"] / typical

    for tag in window_tags:  # a count over the last k events is just min(k, n_events)
        counts = work.loc[masks[tag]].groupby(id_col, sort=False).size()
        out[f"n_events_{tag}"] = counts.reindex(out.index).fillna(0).astype(int)
    return out


def _flow_layer(
    work: pd.DataFrame, cfg: FeatureEngineeringConfig, cols: Sequence[str],
    masks: Dict[str, pd.Series], index: pd.Index,
) -> pd.DataFrame:
    """Additive quantities: how much in total, how much typically, how much lately."""
    if not cols:
        return pd.DataFrame(index=index)
    cols = list(cols)

    def agg(frame: pd.DataFrame, stats: Sequence[str], suffix: str = "") -> pd.DataFrame:
        out = frame.groupby(cfg.id_col, sort=False)[cols].agg(list(stats))
        out.columns = [f"{c}_{s}{suffix}" for c, s in out.columns]
        return out

    frames = [agg(work, ("sum", "mean", "max"))]
    frames.append(pd.DataFrame({f"{c}_last": _positional(work, cfg.id_col, cols)[c]["last"]
                                for c in cols}))
    frames += [agg(work.loc[m], ("sum", "mean"), f"_{tag}") for tag, m in masks.items()]

    out = pd.concat([f.reindex(index) for f in frames], axis=1)
    sums = [c for c in out.columns if "_sum" in c]  # inactive but present means zero moved
    out[sums] = out[sums].fillna(0.0)
    return out


def _level_layer(
    work: pd.DataFrame, cfg: FeatureEngineeringConfig, cols: Sequence[str], index: pd.Index,
) -> pd.DataFrame:
    """Snapshots: where the entity stands now, and how it got there.

    Change is measured against the entity's own previous and first observations
    rather than a calendar window, which keeps it defined on short histories and
    free of any assumption about cadence.
    """
    if not cols:
        return pd.DataFrame(index=index)
    cols = list(cols)

    out = work.groupby(cfg.id_col, sort=False)[cols].agg(["min", "max"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]

    positions = _positional(work, cfg.id_col, cols)
    for col in cols:
        pos = positions[col].reindex(out.index)
        out[f"{col}_last"] = pos["last"]
        out[f"{col}_delta_last"] = pos["last"] - pos["prev"]
        out[f"{col}_delta_total"] = pos["last"] - pos["first"]
        out[f"{col}_pct_change_total"] = (
            out[f"{col}_delta_total"] / pos["first"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
    return out.reindex(index)


def _categorical_layer(
    work: pd.DataFrame, cfg: FeatureEngineeringConfig, cols: Sequence[str],
    universe: Dict[str, List[Any]], index: pd.Index,
) -> pd.DataFrame:
    """Variety, concentration and current value of each label column.

    Concentration is a top-value share rather than a raw distinct count, since
    distinct counts are bounded by the number of rows and so largely restate
    n_events on short histories.
    """
    if not cols:
        return pd.DataFrame(index=index)

    id_col, frames, encoded = cfg.id_col, [], []
    for col in cols:
        valid = work.loc[work[col].notna(), [id_col, col]]
        by_id = valid.groupby(id_col, sort=False)
        per_value = valid.groupby([id_col, col], sort=False).size().groupby(level=0)

        stats = pd.DataFrame({
            f"{col}_n_distinct": per_value.size(),
            f"{col}_top_share": per_value.max() / per_value.sum().replace(0, np.nan),
        })
        # consecutive rows the entity has now spent on its current value
        changed = (valid[col] != by_id[col].shift()).astype(int)
        run = changed.groupby(valid[id_col], sort=False).cumsum()
        stats[f"{col}_run_len"] = (
            (run == run.groupby(valid[id_col], sort=False).transform("max"))
            .groupby(valid[id_col], sort=False).sum()
        )

        last = by_id[col].last()
        one_hot = _one_hot(last, f"{col}_last", _universe_for(last, col, universe, cfg.max_categories))
        encoded += list(one_hot.columns)
        frames.append(pd.concat([stats, one_hot], axis=1))

    out = pd.concat([f.reindex(index) for f in frames], axis=1)
    out[encoded] = out[encoded].fillna(0).astype("int8")
    return out


def _static_layer(
    work: pd.DataFrame, cfg: FeatureEngineeringConfig, cols: Sequence[str],
    universe: Dict[str, List[Any]], index: pd.Index,
) -> pd.DataFrame:
    """Last known value of each entity-level attribute; one-hot if not numeric."""
    if not cols:
        return pd.DataFrame(index=index)

    positions, frames, encoded = _positional(work, cfg.id_col, cols), [], []
    for col in cols:
        last = positions[col]["last"]
        if pd.api.types.is_numeric_dtype(work[col]):
            frames.append(last.rename(col).to_frame())
        else:
            one_hot = _one_hot(last, col, _universe_for(last, col, universe, cfg.max_categories))
            encoded += list(one_hot.columns)
            frames.append(one_hot)

    out = pd.concat([f.reindex(index) for f in frames], axis=1)
    out[encoded] = out[encoded].fillna(0).astype("int8")
    return out


# --------------------------------------------------------------------------
# Optional layer: terminal-event hazard
# --------------------------------------------------------------------------

@dataclass
class HazardModel:
    """Population discrete-time hazard of a once-only terminal event.

    `by_order[k][context]` and `by_runlen[n]` hold (n_at_risk, n_events); the
    parallel `own_*` maps hold each entity's own contribution, which is what
    lets a per-entity feature be read leave-one-out. That subtraction is not
    cosmetic: the hazard is fit on data containing each entity's own outcome, so
    a positive in a thin context would otherwise read its own label back out of
    the population rate.

    Reuse this when scoring later. Entities absent from `own_*` subtract
    nothing, which is correct for ones the model has not seen.
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
        """Long-format view of the fitted rates. A flat `rate` across contexts
        means the chosen `event_col` does not discriminate."""
        rows = [{"kind": "runlen", "context": n_obs, "n_at_risk": n, "n_events": e,
                 "rate": e / n if n else np.nan}
                for n_obs, (n, e) in sorted(self.by_runlen.items())]
        rows += [{"kind": f"order{order}", "context": "_".join(map(str, ctx)),
                  "n_at_risk": n, "n_events": e, "rate": e / n if n else np.nan}
                 for order in sorted(self.by_order)
                 for ctx, (n, e) in self.by_order[order].items()]
        if not rows:
            return pd.DataFrame(columns=["kind", "context", "n_at_risk", "n_events", "rate"])
        return (pd.DataFrame(rows).sort_values(["kind", "n_at_risk"], ascending=[True, False])
                .reset_index(drop=True))


def _sequences(df: pd.DataFrame, id_col: str) -> Dict[Hashable, List[Any]]:
    """Each entity's states, oldest first, nulls dropped. Rows must be sorted."""
    valid = df.loc[df[_STATE].notna(), [id_col, _STATE]]
    return {i: list(states) for i, states in valid.groupby(id_col, sort=False)[_STATE]}


def _fit_hazard(
    sequences: Dict[Hashable, List[Any]], orders: Sequence[int], runlen_cap: int,
) -> HazardModel:
    """Person-period expansion: one at-risk row per observation before the event.

    Each row carries a 1 if the terminal event followed it and a 0 otherwise. An
    entity that never reached the event contributes all of its observations as
    zeros INCLUDING its last, which is a known non-event rather than a censored
    row, because the observation window runs past it -- that is what makes the
    entity a negative.

    Dropping that final observation would remove only zeros, and only from
    entities that never converted, biasing every rate upward by roughly one over
    the mean history length. On short histories that is not a rounding error: it
    inflated the strongest context by ~40% in testing.

    The exception is a history truncated by the data pull rather than by the
    label window, where the last row's outcome is genuinely unknown. Exclude
    such entities, or fit on an earlier window and pass the model in prefit.
    """
    base = [0, 0]
    own_base: Dict[Hashable, List[int]] = defaultdict(lambda: [0, 0])
    by_runlen: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    own_by_runlen: Dict[Tuple[Hashable, int], List[int]] = defaultdict(lambda: [0, 0])
    by_order = {k: defaultdict(lambda: [0, 0]) for k in orders}
    own_by_order = {k: defaultdict(lambda: [0, 0]) for k in orders}

    for id_, states in sequences.items():
        for i, state in enumerate(states):
            if state == _TERMINAL:
                break  # the window ended here; nothing after it is at risk
            event = int(i + 1 < len(states) and states[i + 1] == _TERMINAL)
            bucket = min(i + 1, runlen_cap)

            cells = [base, own_base[id_], by_runlen[bucket], own_by_runlen[(id_, bucket)]]
            for order in orders:
                if i + 1 >= order:
                    ctx = tuple(states[i - order + 1 : i + 1])
                    cells += [by_order[order][ctx], own_by_order[order][(id_, ctx)]]
            for cell in cells:
                cell[0] += 1
                cell[1] += event

    def freeze(counter: Dict[Any, List[int]]) -> Dict[Any, Counts]:
        return {k: (v[0], v[1]) for k, v in counter.items()}

    return HazardModel(
        base=(base[0], base[1]),
        by_order={k: freeze(v) for k, v in by_order.items()}, by_runlen=freeze(by_runlen),
        own_base=freeze(own_base),
        own_by_order={k: freeze(v) for k, v in own_by_order.items()},
        own_by_runlen=freeze(own_by_runlen), runlen_cap=runlen_cap,
    )


def _loo_rate(total: Optional[Counts], own: Counts, min_support: int) -> Optional[float]:
    """Population rate for one context with `own`'s contribution removed.

    None means the context is too thin once the entity's own rows are out, and
    the caller should back off to a shorter context.
    """
    if total is None:
        return None
    n, e = total[0] - own[0], total[1] - own[1]
    return e / n if n >= max(min_support, 1) else None


def _hazard_layer(
    fit_df: pd.DataFrame, work: pd.DataFrame, cfg: FeatureEngineeringConfig,
    index: pd.Index, prefit: Optional[HazardModel],
) -> Tuple[pd.DataFrame, Optional[HazardModel]]:
    """How likely the terminal event is from where each entity currently stands.

    `fit_df` still holds the outcome rows and is used only to fit the population
    hazard; `work` has them stripped and is the only frame a per-entity feature
    is read from.
    """
    id_col = cfg.id_col
    orders = sorted({int(k) for k in cfg.context_orders if int(k) >= 1}, reverse=True)

    # One distinct non-outcome state leaves nothing to condition on, so every
    # context feature would hold the same number for every entity.
    has_context = int(work[_STATE].nunique(dropna=True)) >= 2
    if not has_context:
        warnings.warn(
            "no event_col, or only one distinct state, so there is no context to condition on -- "
            "emitting hazard_runlen only. Set event_col to a per-row categorical for the rest."
        )

    hazard = prefit
    if hazard is None:
        fit_seqs = _sequences(fit_df, id_col)
        if not fit_seqs:
            warnings.warn("no usable history; hazard features skipped")
            return pd.DataFrame(index=index), None
        hazard = _fit_hazard(fit_seqs, orders, cfg.runlen_cap)

    rows: List[Dict[str, Any]] = []
    for id_, states in _sequences(work, id_col).items():
        n_obs = len(states)
        base = _loo_rate(hazard.base, hazard.own_base.get(id_, (0, 0)), 1)
        base = hazard.base_rate if base is None else base

        bucket = min(n_obs, hazard.runlen_cap)
        runlen = _loo_rate(hazard.by_runlen.get(bucket),
                           hazard.own_by_runlen.get((id_, bucket), (0, 0)), cfg.hazard_min_support)
        row = {id_col: id_, "hazard_runlen": base if runlen is None else runlen}

        if has_context:
            rate, used = None, 0
            for order in orders:  # longest context first, backing off when thin
                if n_obs < order:
                    continue
                ctx = tuple(states[-order:])
                rate = _loo_rate(hazard.by_order[order].get(ctx),
                                 hazard.own_by_order[order].get((id_, ctx), (0, 0)),
                                 cfg.hazard_min_support)
                if rate is not None:
                    used = order
                    break
            rate = base if rate is None else rate
            row["hazard_context"] = rate
            row["hazard_context_order"] = used  # 0 = fell back to the base rate
            row["hazard_lift"] = rate / base if base else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame(index=index), hazard
    return pd.DataFrame(rows).set_index(id_col).reindex(index), hazard


# --------------------------------------------------------------------------
# Cutoffs
# --------------------------------------------------------------------------

def _resolve_cutoff(work: pd.DataFrame, id_col: str, as_of: Any, unit: str) -> pd.Series:
    """Per-entity cutoff on the float time axis.

    None means one cutoff for the whole population at the latest timestamp in
    the data. Each entity's own last row looks like the natural default and is
    not offered, because it makes `recency` identically zero.
    """
    ids = pd.Index(work[id_col].unique(), name=id_col)
    if as_of is None:
        return pd.Series(float(work[_T].max()), index=ids)
    if isinstance(as_of, dict):
        as_of = pd.Series(as_of)
    if not isinstance(as_of, pd.Series):
        return pd.Series(float(_to_axis([as_of], unit).iloc[0]), index=ids)

    cutoff = _to_axis(as_of, unit)
    cutoff.index = as_of.index
    cutoff = cutoff.reindex(ids)
    if cutoff.isna().any():
        warnings.warn(
            f"{int(cutoff.isna().sum())} entity/entities have no as_of value; falling back to the "
            "population maximum. Pass a cutoff for every entity to avoid mixing cutoffs."
        )
        cutoff = cutoff.fillna(float(work[_T].max()))
    cutoff.index.name = id_col
    return cutoff


def matched_cutoffs(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    outcome_at: Union[pd.Series, Dict[Hashable, Any]],
    strata: Optional[str] = None,
    seed: int = 0,
    upper_bound: Optional[Any] = None,
) -> pd.Series:
    """Per-entity `as_of` cutoffs whose DISTRIBUTION matches across classes.

    Cutting every positive at its outcome date and every negative at today makes
    the cutoff a function of the label. Recency then runs to a recent date for
    one class and to a date years back for the other, separating them without
    carrying information about who converts. On simulated data with that shape
    it handed `recency` 0.32 of spurious AUC and pushed five other timing
    features off 0.5. A per-feature screen does not reveal it, because no single
    feature dominates.

    Each negative instead gets a pseudo-outcome date: a waiting time resampled
    from the positives, measured from its own first row, so both classes are
    observed at comparable points in their life cycle. Positives keep their real
    date.

    Parameters
    ----------
    outcome_at : outcome timestamp per positive; everything else is a negative.
    strata : optional column to resample within (e.g. line of business), for
        when waiting times differ by segment. Strata with too few positives fall
        back to the pooled distribution.
    upper_bound : cutoffs are clipped here (default: the latest timestamp in
        `df`), so none lands beyond the observable window.
    """
    rng = np.random.default_rng(seed)
    outcome_at = pd.to_datetime(pd.Series(outcome_at))
    stamps = pd.to_datetime(df[time_col])
    first_seen = stamps.groupby(df[id_col]).min()
    ids = first_seen.index
    cap = pd.Timestamp(upper_bound) if upper_bound is not None else stamps.max()

    positives = outcome_at.reindex(ids).dropna()
    if positives.empty:
        raise ValueError("outcome_at holds no entity present in df; nothing to resample from")

    def offsets(pool: pd.Index) -> np.ndarray:
        days = (positives.reindex(pool).dropna() - first_seen.reindex(pool)).dt.days
        return days[days.notna() & (days >= 0)].to_numpy()

    pooled = offsets(ids)
    if pooled.size == 0:
        raise ValueError("no positive has a non-negative waiting time; check outcome_at against df")

    entity_stratum = df.groupby(id_col)[strata].first() if strata else None
    by_stratum: Dict[Any, np.ndarray] = {}
    if entity_stratum is not None:
        for value, group in entity_stratum.groupby(entity_stratum):
            drawn = offsets(group.index)
            if drawn.size >= 20:  # too few to resample from meaningfully
                by_stratum[value] = drawn

    negatives = ids.difference(positives.index)
    draws = np.array([
        rng.choice(by_stratum.get(entity_stratum.get(e), pooled) if entity_stratum is not None
                   else pooled)
        for e in negatives
    ], dtype=float)

    cutoff = pd.concat([
        positives,
        pd.Series(first_seen.reindex(negatives) + pd.to_timedelta(draws, unit="D"), index=negatives),
    ]).reindex(ids)

    floor = first_seen + pd.Timedelta(days=1)  # never before the entity's own first row
    cutoff = cutoff.where(cutoff >= floor, floor).clip(upper=cap)
    if int((cutoff >= cap).sum()):
        warnings.warn(
            f"{int((cutoff >= cap).sum())} entity/entities drew a pseudo-cutoff at or past the end "
            "of the data and were clipped to it. Their cutoffs are no longer matched -- exclude "
            "them, or stratify so the resampled waiting times suit their tenure."
        )
    cutoff.index.name = id_col
    return cutoff


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _prepare(
    df: pd.DataFrame, cfg: FeatureEngineeringConfig, as_of: Any,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Coerce time, blank structural zeros, apply the cutoff, sort."""
    for col in (cfg.id_col, cfg.time_col):
        if col not in df.columns:
            raise KeyError(f"column '{col}' not found in input dataframe")

    unit = _detect_time_unit(df[cfg.time_col])
    work = df.copy()
    work[_T] = _to_axis(work[cfg.time_col], unit)

    for col in cfg.zero_is_missing:
        if col not in work.columns:
            raise KeyError(f"zero_is_missing names '{col}', which is not in the input dataframe")
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].replace(0, np.nan)
        else:
            warnings.warn(f"zero_is_missing names '{col}', which is not numeric; ignoring it")

    if work[_T].isna().any():
        warnings.warn(f"dropping {int(work[_T].isna().sum())} row(s) with an unusable {cfg.time_col}")
        work = work[work[_T].notna()]
    if work.empty:
        raise ValueError("no rows with a usable timestamp")

    cutoff = _resolve_cutoff(work, cfg.id_col, as_of, unit)
    work = work[work[_T] <= work[cfg.id_col].map(cutoff)]
    if work.empty:
        raise ValueError("no rows remain at or before the as_of cutoff(s)")
    return work.sort_values([cfg.id_col, _T], kind="mergesort"), cutoff, unit


def _split_outcome(
    work: pd.DataFrame, cfg: FeatureEngineeringConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tag each row's state and strip the outcome rows out of the feature frame."""
    id_col = cfg.id_col
    is_terminal = pd.Series(False, index=work.index)
    if cfg.terminal_flag_col:
        if cfg.terminal_flag_col not in work.columns:
            raise KeyError(f"terminal_flag_col '{cfg.terminal_flag_col}' not found")
        flag = work[cfg.terminal_flag_col]
        is_terminal |= flag.notna() & (flag != 0)  # 1 / True / "Y" all count as set
    if cfg.terminal_event_states and cfg.event_col:
        is_terminal |= work[cfg.event_col].isin(set(cfg.terminal_event_states))

    if not is_terminal.any():
        raise ValueError(
            "the outcome declaration matched no rows at/before the cutoff. Unfixed this would "
            "count outcome rows as ordinary events and hand the label to n_events and recency, so "
            "it is an error rather than a no-op. Check that terminal_flag_col is the column set on "
            f"outcome rows, or that terminal_event_states holds values appearing in {cfg.event_col!r}."
        )

    # The window should END at the outcome; rows after one mean the cutoff is
    # past it or the outcome repeats, and either way the label is ambiguous.
    after = int((is_terminal.groupby(work[id_col]).cumsum() - is_terminal.astype(int) > 0).sum())
    if after:
        warnings.warn(f"{after} row(s) fall after an outcome row inside the same entity's window.")

    base = (pd.Series(_OBSERVED, index=work.index, dtype=object) if cfg.event_col is None
            else work[cfg.event_col].astype(object).fillna(_OBSERVED))
    work[_STATE] = base.where(~is_terminal, _TERMINAL)

    fit_df, before = work, work[id_col].nunique()
    work = work[~is_terminal]
    if work.empty:
        raise ValueError("every row at/before the cutoff is an outcome row")
    if before - work[id_col].nunique():
        warnings.warn(
            f"{before - work[id_col].nunique()} entity/entities had no history beyond the outcome "
            "row and are absent from the table. Dropping them changes the population -- handle "
            "them explicitly rather than letting them vanish."
        )
    return fit_df, work


def _resolve_roles(
    work: pd.DataFrame, cfg: FeatureEngineeringConfig, given: Optional[Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    if given is None:
        given = infer_column_roles(
            work.drop(columns=[c for c in _INTERNAL if c in work.columns]),
            cfg.id_col, cfg.time_col, max_categories=cfg.max_categories,
            level_persistence=cfg.level_persistence, sample_rows=cfg.infer_sample_rows,
            ignore_cols=cfg.ignore_cols,
        )
    roles = {k: list(v) for k, v in given.items()}
    overrides = {"flow": cfg.flow_cols, "level": cfg.level_cols,
                 "categorical": cfg.categorical_cols, "static": cfg.static_cols}
    excluded = set(cfg.ignore_cols) | ({cfg.terminal_flag_col} if cfg.terminal_flag_col else set())
    for role, override in overrides.items():
        chosen = override if override is not None else roles.get(role, [])
        roles[role] = [c for c in chosen if c in work.columns and c not in excluded]
    roles.setdefault("skipped", [])
    return roles


def build_feature_table(
    df: pd.DataFrame,
    config: FeatureEngineeringConfig,
    as_of: Optional[Union[str, float, pd.Timestamp, pd.Series, Dict[Hashable, Any]]] = None,
    column_roles: Optional[Dict[str, List[str]]] = None,
    hazard: Optional[HazardModel] = None,
) -> FeatureEngineeringArtifacts:
    """Build one row of features per entity.

    Parameters
    ----------
    df : transactional log, one row per (entity, time, ...).
    config : column roles and knobs; only id_col and time_col are required.
    as_of : point-in-time cutoff. None uses one cutoff for the whole population
        at the latest timestamp present. Pass a per-entity Series/dict, or one
        timestamp, when joining to a historical outcome -- and build it with
        `matched_cutoffs` if the classes would otherwise be cut differently.
    column_roles : skip inference and use these (as returned in
        `artifacts.column_roles`), so a scoring batch cannot drift.
    hazard : a previously fitted `HazardModel`, used instead of refitting.
    """
    cfg = config
    if cfg.terminal_event_states and cfg.event_col is None:
        raise ValueError(
            "terminal_event_states names values inside event_col, which is not set. Use "
            "terminal_flag_col when the outcome is marked by its own flag column."
        )
    if cfg.declares_outcome and as_of is None:
        raise ValueError(
            "as_of is required when an outcome is declared: without it the cutoff falls after the "
            "event for positives and so encodes the label. Pass the per-entity decision-time "
            "cutoff the label was defined against."
        )

    work, cutoff, unit = _prepare(df, cfg, as_of)
    fit_df = work
    if cfg.declares_outcome:
        fit_df, work = _split_outcome(work, cfg)
        cutoff = cutoff.reindex(pd.Index(work[cfg.id_col].unique(), name=cfg.id_col))
    work[_CUT] = work[cfg.id_col].map(cutoff)

    roles = _resolve_roles(work, cfg, column_roles)
    masks, window_tags = _subsets(work, cfg)
    universe = {k: list(v) for k, v in (cfg.category_universe or {}).items()}

    layers = {"activity": _activity_layer(work, cfg, cutoff, masks, window_tags)}
    index = layers["activity"].index
    layers["flow"] = _flow_layer(work, cfg, roles["flow"], masks, index)
    layers["level"] = _level_layer(work, cfg, roles["level"], index)
    layers["categorical"] = _categorical_layer(work, cfg, roles["categorical"], universe, index)
    layers["static"] = _static_layer(work, cfg, roles["static"], universe, index)
    if cfg.declares_outcome:
        layers["hazard"], hazard = _hazard_layer(fit_df, work, cfg, index, hazard)
    else:
        hazard = None

    table = pd.concat([f for f in layers.values() if not f.empty], axis=1)
    table.index.name = cfg.id_col
    return FeatureEngineeringArtifacts(
        table=table.reset_index(),
        column_roles=roles,
        category_universe=universe,
        feature_columns_by_layer={k: list(v.columns) for k, v in layers.items() if not v.empty},
        time_unit=unit,
        hazard=hazard,
    )


# --------------------------------------------------------------------------
# Demo: four unrelated transactional shapes
# --------------------------------------------------------------------------

def _demo_datasets(seed: int = 0) -> Dict[str, Dict[str, Any]]:
    """Logs sharing nothing but the (entity, time, ...) shape."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, Any]] = {}

    def stamps(start: str, n: int, span: int) -> pd.Series:
        return pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, span, n), "D")

    n = 20_000
    out["card_payments"] = dict(df=pd.DataFrame({
        "account_id": rng.integers(0, 3_000, n), "posted_at": stamps("2026-01-01", n, 180),
        "amount": np.round(rng.gamma(2.0, 60.0, n), 2),
        "merchant_category": rng.choice(["grocery", "fuel", "travel", "dining", "online"], n),
        "channel": rng.choice(["chip", "online", "contactless"], n),
    }), cfg=FeatureEngineeringConfig(id_col="account_id", time_col="posted_at",
                                     recent_windows=(30, 90)))

    n = 30_000
    out["clickstream"] = dict(df=pd.DataFrame({
        "user_id": rng.integers(0, 4_000, n), "ts": stamps("2026-03-01", n, 30),
        "page": rng.choice(["home", "search", "product", "cart", "checkout"], n),
        "device": rng.choice(["ios", "android", "web"], n),
        "dwell_seconds": np.round(rng.exponential(45.0, n), 1),
    }), cfg=FeatureEngineeringConfig(id_col="user_id", time_col="ts", recent_windows=(7,)))

    devices, per = 500, 40
    out["sensor_readings"] = dict(df=pd.DataFrame({  # integer time axis
        "device_id": np.repeat(np.arange(devices), per), "reading_no": np.tile(np.arange(per), devices),
        "temperature_c": np.round((20 + rng.normal(0, .4, (devices, per)).cumsum(1)).ravel(), 2),
        "power_draw_wh": np.round(rng.gamma(3.0, 1.5, devices * per), 2),
        "firmware": np.repeat(rng.choice(["v1", "v2", "v3"], devices), per),
    }), cfg=FeatureEngineeringConfig(id_col="device_id", time_col="reading_no"))

    # wealth NBO: multi-year gaps, once-only outcome on its own marker row
    states = ["browse", "offer_view", "offer_click", "service_call"]
    trans = {"browse": [.35, .30, .20, .15], "offer_view": [.25, .30, .30, .15],
             "offer_click": [.20, .25, .35, .20], "service_call": [.35, .20, .15, .30]}
    true_haz = {"browse": .02, "offer_view": .07, "offer_click": .22, "service_call": .04}
    rows, labels, cuts = [], {}, {}
    for cid in range(4_000):
        state = str(rng.choice(states))
        when = pd.Timestamp("2007-01-01") + pd.Timedelta(days=int(rng.integers(0, 4_000)))
        balance, hit = float(rng.uniform(5_000, 90_000)), 0
        for _ in range(int(np.clip(rng.geometric(0.28), 1, 20))):
            balance *= float(rng.uniform(0.96, 1.04))
            rows.append(dict(customer_id=cid, txn_ts=when, event=state,
                             txn_amount=round(float(rng.uniform(10, 2_000)), 2),
                             balance=round(balance, 2), outcome_flag=np.nan))
            when += pd.Timedelta(days=float(rng.uniform(180, 1_400)))  # years, not days
            if rng.random() < true_haz[state]:
                # the outcome row: an id, a date and a flag. Nothing else.
                rows.append(dict(customer_id=cid, txn_ts=when, event=None, txn_amount=np.nan,
                                 balance=np.nan, outcome_flag=1))
                hit = 1
                break
            state = str(rng.choice(states, p=trans[state]))
        labels[cid], cuts[cid] = hit, when
    out["wealth_nbo"] = dict(
        df=pd.DataFrame(rows), labels=pd.Series(labels), as_of=pd.Series(cuts),
        cfg=FeatureEngineeringConfig(id_col="customer_id", time_col="txn_ts", event_col="event",
                                     terminal_flag_col="outcome_flag"),
    )
    return out


def _auc(scores: pd.Series, labels: pd.Series) -> float:
    """Rank AUC (Mann-Whitney), NaN-tolerant. Demo leakage check only."""
    frame = pd.DataFrame({"s": np.asarray(scores, float), "y": np.asarray(labels)}).dropna()
    n_pos, n_neg = int((frame.y == 1).sum()), int((frame.y == 0).sum())
    if not n_pos or not n_neg or frame.s.nunique() < 2:
        return float("nan")
    ranks = frame.s.rank()
    return float((ranks[frame.y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


if __name__ == "__main__":
    for name, case in _demo_datasets().items():
        df, cfg = case["df"], case["cfg"]
        art = build_feature_table(df, cfg, as_of=case.get("as_of"))
        table, per_id = art.table, df.groupby(cfg.id_col).size()
        gaps = df.sort_values([cfg.id_col, cfg.time_col]).groupby(cfg.id_col)[cfg.time_col].diff()
        gap = gaps.dt.days.median() if hasattr(gaps, "dt") else gaps.median()

        print(f"\n{'=' * 78}")
        print(f"{name}: {len(df):,} rows, {df[cfg.id_col].nunique():,} entities, "
              f"median {per_id.median():.0f} rows each, median gap {gap:,.0f} {art.time_unit}s")
        print("  roles: " + " | ".join(f"{r}={c}" for r, c in art.column_roles.items() if c))
        print(f"  table {table.shape[0]:,} x {table.shape[1]}  ->  " +
              ", ".join(f"{k} {len(v)}" for k, v in art.feature_columns_by_layer.items()))
        nulls = table.drop(columns=cfg.id_col).isna().mean()
        dead = [c for c in table.columns if c != cfg.id_col and table[c].nunique(dropna=False) <= 1]
        print(f"  columns >50% null: {int((nulls > 0.5).sum())} of {len(nulls)} | "
              f"constant columns: {len(dead)}")

        if "labels" in case:
            y = case["labels"].reindex(table[cfg.id_col]).reset_index(drop=True)
            num = [c for c in table.columns
                   if c != cfg.id_col and pd.api.types.is_numeric_dtype(table[c])]
            aucs = pd.Series({c: _auc(table[c], y) for c in num}).dropna()
            top = aucs.sort_values(ascending=False).head(3)
            print(f"  hazard base rate {art.hazard.base_rate:.3f} | top AUC: " +
                  ", ".join(f"{k} {v:.3f}" for k, v in top.items()))
            print(f"  max single-feature AUC {aucs.max():.3f} (>0.99 would mean a leak)")
    print(f"\n{'=' * 78}")
