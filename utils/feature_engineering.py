"""General-purpose feature engineering for transactional data.

Turns any log of the form "one row per (entity, time, ...)" into one row per
entity, suitable for variable selection and modeling. The only thing the
input must have is an id column and something that orders events within an
id. Every other column role is optional and, left unspecified, is inferred
from the data.

    cfg = FeatureEngineeringConfig(id_col="customer_id", time_col="txn_ts")
    artifacts = build_feature_table(df, cfg)

That is the whole required contract. Point it at card payments, clickstream,
sensor readings, claims, or trade tickets and it will produce a sensible
table without being told what the columns mean.

Column roles
------------
Roles are inferred by `infer_column_roles` unless you pass them explicitly:

  static       constant within an entity (segment, region, account type) ->
               carried through as the last known value.
  flow         numeric and additive; each row is an independent quantity
               (amount, quantity, duration) -> summed and averaged.
  level        numeric and persistent; each row is a snapshot that carries
               over (balance, price, temperature) -> last value and change.
  categorical  low-cardinality labels (channel, merchant type, page, status)
               -> distinct count, concentration, current value.

Flow and level are separated by within-entity lag-1 autocorrelation, which is
the property that actually distinguishes them: a balance is close to what it
was last time, a payment amount is not. Inference is a starting point, not an
oracle -- it is reported in `FeatureEngineeringArtifacts.column_roles`, and
anything it gets wrong can be pinned in the config.

Feature families
----------------
Every family is deliberately small. On short histories -- and transactional
data is dominated by entities with a handful of rows -- a wide aggregate
family is mostly redundant: with three rows the mean and median inter-event
gap are the same number, the standard deviation is a rescaled absolute
difference, and a 90-day window contains the entire history. What is kept is
the subset that stays distinct and defined when an entity has one or two rows.

  activity     n_events, tenure, recency, median/last gap, burstiness, rate,
               and per-window event counts. Always produced.
  flow         sum, mean, max, last, plus per-window sum and mean.
  level        last, min, max, change since previous, change since first.
  categorical  n_distinct, top-value share, current-value run length, and a
               schema-stable one-hot of the current value.
  static       last known value, one-hot encoded if not numeric.

Time
----
`time_col` may be datetime-like or numeric. Datetimes are measured in days; a
numeric column (a sequence number, an epoch, a reading index) is used as-is
and every window and duration is in those same units. The unit is recorded in
`artifacts.time_unit`.

Point-in-time correctness
-------------------------
Features come only from rows at or before an `as_of` cutoff. The default is a
single cutoff for the whole population at the latest timestamp in the data --
"score everyone as of now". Note that a PER-ENTITY cutoff of each entity's own
last row, which looks like the natural default, silently forces recency to
zero for everyone and so is not offered as one.

If the table will be joined to a historical outcome, pass a per-id `as_of`
strictly before the outcome is observed, or the features will include rows
from after the thing being predicted.

Optional: terminal-event hazard
-------------------------------
One label-aware extra, off unless the outcome is declared. It suits a specific
and common setup: an outcome that happens at most once and closes the
observation window, with history truncated at it.

Declare the outcome rows either way round:

  terminal_flag_col     a column that is set only on outcome rows. This is the
                        usual shape when outcomes arrive from a separate
                        table: the row carries an id, a date and a flag, and
                        every other column is null because its only job is to
                        say WHEN the outcome happened, not to describe it.
  terminal_event_states values within `event_col` that mark the outcome, for
                        logs where the outcome is one state among many.

Declaring them is what strips those rows out of every feature layer. An
undeclared outcome row is counted as an ordinary event, which hands the label
straight to `n_events`, `recency` and the windowed counts -- so a declaration
that matches nothing is an error here, never a silent no-op.

Marker rows being otherwise null is harmless: they are removed before roles
are inferred or any aggregate is computed, and the hazard reads only their
position in time. `event_col` is optional alongside a flag -- without one
there is no context to condition on, so only the run-length hazard is
produced. See `_hazard_layer` and `_fit_hazard` for the rest.

Everything above works without any of this, and the module has no other notion
of a label.

Scale
-----
The core is vectorized -- groupby aggregations, no per-entity Python loops --
so it is bounded by pandas rather than by row count. Only the optional hazard
layer iterates per entity.

Dependencies are numpy and pandas, and nothing else, including nothing else in
this repository, so the file can be dropped into another project as-is.
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
    "build_feature_table",
]

Context = Tuple[Any, ...]
Counts = Tuple[int, int]  # (n_at_risk, n_events)

_STATE = "__state__"       # internal per-row state used by the hazard layer
_OBSERVED = "__observed__"  # internal state for a row whose own event type is unknown
_TERMINAL = "__terminal__"  # internal state standing for an outcome row
_INTERNAL = ("__t__", "__cutoff__", _STATE)

_SECONDS_PER_DAY = 86_400.0


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class FeatureEngineeringConfig:
    """Column roles and knobs. Only `id_col` and `time_col` are required.

    Any role left as None is inferred from the data; pass a list (including an
    empty one) to pin it and skip inference for that role.
    """

    id_col: str
    time_col: str

    flow_cols: Optional[Sequence[str]] = None
    level_cols: Optional[Sequence[str]] = None
    categorical_cols: Optional[Sequence[str]] = None
    static_cols: Optional[Sequence[str]] = None
    ignore_cols: Sequence[str] = field(default_factory=tuple)

    # Windows are in the time column's own units: days for datetimes, raw
    # units for a numeric time column. Empty disables windowed features.
    recent_windows: Sequence[float] = (30, 90)

    max_categories: int = 20  # one-hot width cap per column; the rest fold into __other__
    level_autocorr: float = 0.5  # within-entity lag-1 autocorrelation above which numeric = level
    infer_sample_rows: int = 200_000  # rows sampled for role inference on large frames

    category_universe: Optional[Dict[str, List[Any]]] = None  # persist and reuse when scoring

    # -- optional terminal-event hazard; inert unless the outcome is declared
    event_col: Optional[str] = None
    # Outcome rows, declared either as a flag column that is set only on them
    # (the usual shape when outcomes come from a separate table) or as values
    # within event_col. Either one strips those rows from every feature layer.
    terminal_flag_col: Optional[str] = None
    terminal_event_states: Sequence[Any] = field(default_factory=tuple)
    context_orders: Sequence[int] = (1, 2)
    hazard_min_support: int = 25
    runlen_cap: int = 12


@dataclass
class FeatureEngineeringArtifacts:
    """The feature table plus everything needed to rebuild it identically.

    `column_roles`, `category_universe` and `hazard` are the reusable
    definitions: pass them back in when scoring a later batch so the schema and
    the population estimates stay fixed, rather than being re-derived from
    whatever data happens to arrive next.
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

    Days are derived through `total_seconds` rather than by casting to int64
    and dividing by a nanosecond constant. Since pandas 2.0 a datetime column
    may be backed by second, millisecond, microsecond or nanosecond
    resolution, and casting to int64 returns the underlying integer in
    whichever unit that happens to be -- so the constant is right for one
    frame and wrong by a factor of a thousand for the next, silently.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if unit != "day":
        return pd.to_numeric(series, errors="coerce").astype(float)

    stamps = pd.to_datetime(series, errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(stamps):
        stamps = pd.to_datetime(series, errors="coerce", utc=True)  # mixed offsets
    aware = getattr(stamps.dt, "tz", None) is not None
    epoch = pd.Timestamp("1970-01-01", tz="UTC") if aware else pd.Timestamp("1970-01-01")
    return (stamps - epoch).dt.total_seconds().where(stamps.notna()) / _SECONDS_PER_DAY


def _detect_time_unit(series: pd.Series) -> str:
    """"day" for anything datetime-like, "unit" for a numeric ordering column.

    A numeric column is taken at face value rather than guessed at: an integer
    that happens to look like 20260101 is treated as the number it is, and
    windows are then in those units. Convert it yourself if it means a date.
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
# Column role inference
# --------------------------------------------------------------------------

def infer_column_roles(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    max_categories: int = 20,
    level_autocorr: float = 0.5,
    sample_rows: int = 200_000,
    ignore_cols: Sequence[str] = (),
) -> Dict[str, List[str]]:
    """Classify every remaining column as static, flow, level, categorical or skipped.

    The one non-obvious test is flow vs. level. Both are numeric, so dtype
    cannot separate them, and column names cannot be relied on across datasets.
    What does separate them is persistence: a balance, a price or a temperature
    is close to its own previous value within the same entity, while an amount
    or a quantity is not. So the rule is the pooled within-entity lag-1
    autocorrelation, computed on entity-mean-centered values so that
    differences BETWEEN entities cannot masquerade as persistence.

    Columns constant within an entity are static regardless of dtype, and
    non-numeric columns with too many distinct values are skipped rather than
    encoded -- free text or an identifier would otherwise produce a feature per
    value.
    """
    reserved = {id_col, time_col, *ignore_cols}
    candidates = [c for c in df.columns if c not in reserved]
    roles: Dict[str, List[str]] = {
        "static": [], "flow": [], "level": [], "categorical": [], "skipped": [],
    }
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
            roles["skipped"].append(col)  # nothing to learn from a constant
            continue

        # constant within entity -> a property of the entity, not of its activity
        if float(grouped[col].nunique(dropna=True).le(1).mean()) >= 0.95:
            roles["static"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            centered = series.astype(float) - grouped[col].transform("mean").astype(float)
            lagged = centered.groupby(sample[id_col], sort=False).shift(1)
            usable = centered.notna() & lagged.notna()
            if usable.sum() >= 30 and centered[usable].std() > 0 and lagged[usable].std() > 0:
                persistence = float(np.corrcoef(centered[usable], lagged[usable])[0, 1])
            else:
                persistence = 0.0  # too little within-entity history to tell; treat as flow
            roles["level" if persistence >= level_autocorr else "flow"].append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            roles["skipped"].append(col)  # a second time column needs a role you choose
            continue

        if series.nunique(dropna=True) <= max(max_categories * 5, 100):
            roles["categorical"].append(col)
        else:
            roles["skipped"].append(col)

    return roles


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _one_hot(values: pd.Series, prefix: str, universe: Sequence[Any]) -> pd.DataFrame:
    """One column per `universe` entry, in order, plus an always-present `__other__`.

    `__other__` is emitted even when empty so the schema depends only on the
    universe: a training table and a scoring table built from the same universe
    have identical columns whether or not the new batch contains an unseen
    value. NaN encodes as all zeros.
    """
    columns = list(universe) + ["__other__"]
    known = values.where(values.isin(universe) | values.isna(), other="__other__")
    return pd.DataFrame(
        {f"{prefix}_{cat}": (known == cat).astype("int8") for cat in columns},
        index=values.index,
    )


def _universe_for(
    values: pd.Series, col: str, universe: Dict[str, List[Any]], max_categories: int,
) -> List[Any]:
    """Fixed category list for `col`, keeping the most frequent when capped."""
    if col in universe:
        return universe[col]
    counts = values.value_counts()
    chosen = sorted(counts.index[:max_categories].tolist(), key=str)
    universe[col] = chosen
    return chosen


def _positional(work: pd.DataFrame, id_col: str, cols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """First / previous / last non-null value per entity, for several columns.

    Rows must already be sorted by (id, time). Nulls are dropped per column,
    which is why this is not one groupby over all of them: the last KNOWN
    balance is rarely the balance on the entity's last row.
    """
    out: Dict[str, pd.DataFrame] = {}
    for col in cols:
        valid = work.loc[work[col].notna(), [id_col, col]]
        grouped = valid.groupby(id_col, sort=False)[col]

        # `prev` via tail(2) rather than nth(-2): as of pandas 2.0 nth returns
        # rows on the ORIGINAL index instead of one value per group, which
        # silently misaligns when reindexed by entity. tail(2).first() is the
        # second-to-last value, masked off where the entity has only one row.
        tail = valid.groupby(id_col, sort=False).tail(2).groupby(id_col, sort=False)[col]
        out[col] = pd.DataFrame({
            "first": grouped.first(),
            "last": grouped.last(),
            "prev": tail.first().where(tail.size() == 2),
        })
    return out


def _window_masks(
    work: pd.DataFrame, cutoff_col: str, time_col: str, windows: Sequence[float],
) -> Dict[float, pd.Series]:
    return {
        float(w): (work[time_col] > work[cutoff_col] - float(w)) & (work[time_col] <= work[cutoff_col])
        for w in windows
    }


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------

def _activity_layer(
    work: pd.DataFrame, cfg: FeatureEngineeringConfig, cutoff: pd.Series,
    windows: Dict[float, pd.Series],
) -> pd.DataFrame:
    """Recency, frequency, tenure and rhythm -- the one family every log supports."""
    id_col = cfg.id_col
    grouped = work.groupby(id_col, sort=False)["__t__"]

    out = pd.DataFrame({"n_events": grouped.size(), "__first": grouped.min(), "__last": grouped.max()})
    out["tenure"] = out["__last"] - out["__first"]
    out["recency"] = cutoff.reindex(out.index) - out["__last"]

    by_id = work["__t__"].groupby(work[id_col], sort=False).diff().groupby(work[id_col], sort=False)
    out["gap_median"] = by_id.median()
    out["gap_last"] = by_id.last()
    # >1 means the entity's latest gap is longer than its own norm: cooling off.
    # Scale-free, so it compares entities with very different natural rhythms.
    out["gap_burstiness"] = out["gap_last"] / out["gap_median"].replace(0, np.nan)
    out["events_per_unit"] = out["n_events"] / out["tenure"].replace(0, np.nan)

    for window, mask in windows.items():
        counts = work.loc[mask].groupby(id_col, sort=False).size()
        out[f"n_events_w{window:g}"] = counts.reindex(out.index).fillna(0).astype(int)

    return out.drop(columns=["__first", "__last"])


def _flow_layer(
    work: pd.DataFrame, cfg: FeatureEngineeringConfig, cols: Sequence[str],
    windows: Dict[float, pd.Series], index: pd.Index,
) -> pd.DataFrame:
    """Additive quantities: how much in total, how much typically, how much lately."""
    if not cols:
        return pd.DataFrame(index=index)

    cols = list(cols)
    overall = work.groupby(cfg.id_col, sort=False)[cols].agg(["sum", "mean", "max"])
    overall.columns = [f"{col}_{stat}" for col, stat in overall.columns]
    frames = [overall]

    positions = _positional(work, cfg.id_col, cols)
    frames.append(pd.DataFrame({f"{col}_last": positions[col]["last"] for col in cols}))

    for window, mask in windows.items():
        windowed = work.loc[mask].groupby(cfg.id_col, sort=False)[cols].agg(["sum", "mean"])
        windowed.columns = [f"{col}_{stat}_w{window:g}" for col, stat in windowed.columns]
        frames.append(windowed)

    out = pd.concat([f.reindex(index) for f in frames], axis=1)
    # a window the entity was present for but inactive in truly moved zero
    sums = [c for c in out.columns if "_sum" in c]
    out[sums] = out[sums].fillna(0.0)
    return out


def _level_layer(
    work: pd.DataFrame, cfg: FeatureEngineeringConfig, cols: Sequence[str], index: pd.Index,
) -> pd.DataFrame:
    """Snapshots: where the entity stands now, and how it got there.

    Change is measured against the entity's own previous and first observations
    rather than a calendar window. That keeps it defined on short histories and
    free of any assumption about cadence -- a window-based delta is null for
    every entity whose history is shorter than the window, which on
    transactional data is most of them.
    """
    if not cols:
        return pd.DataFrame(index=index)

    cols = list(cols)
    out = work.groupby(cfg.id_col, sort=False)[cols].agg(["min", "max"])
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]

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

    Concentration is a top-value share rather than a raw distinct count because
    distinct counts are bounded by the number of rows, which makes them largely
    a restatement of n_events on short histories.
    """
    if not cols:
        return pd.DataFrame(index=index)

    id_col = cfg.id_col
    frames: List[pd.DataFrame] = []
    encoded: List[str] = []

    for col in cols:
        valid = work.loc[work[col].notna(), [id_col, col]]
        by_entity = valid.groupby([id_col, col], sort=False).size().groupby(level=0)

        stats = pd.DataFrame({
            f"{col}_n_distinct": by_entity.size(),
            f"{col}_top_share": by_entity.max() / by_entity.sum().replace(0, np.nan),
        })

        # how many consecutive rows the entity has now spent on its current value
        current = valid[col]
        by_id = valid.groupby(id_col, sort=False)
        changed = (current != by_id[col].shift()).astype(int)
        run_id = changed.groupby(valid[id_col], sort=False).cumsum()
        in_last_run = run_id == run_id.groupby(valid[id_col], sort=False).transform("max")
        stats[f"{col}_run_len"] = in_last_run.groupby(valid[id_col], sort=False).sum()

        last_value = by_id[col].last()
        one_hot = _one_hot(last_value, f"{col}_last", _universe_for(
            last_value, col, universe, cfg.max_categories))
        encoded.extend(one_hot.columns)
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

    positions = _positional(work, cfg.id_col, cols)
    frames: List[pd.DataFrame] = []
    encoded: List[str] = []
    for col in cols:
        last_value = positions[col]["last"]
        if pd.api.types.is_numeric_dtype(work[col]):
            frames.append(last_value.rename(col).to_frame())
        else:
            one_hot = _one_hot(last_value, col, _universe_for(
                last_value, col, universe, cfg.max_categories))
            encoded.extend(one_hot.columns)
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

    `by_order[k][context]` and `by_runlen[n]` hold (n_at_risk, n_events). The
    parallel `own_*` maps hold each entity's own contribution to those cells,
    which is what lets a per-entity feature be read leave-one-out. That
    subtraction is not cosmetic: the hazard is fit on data containing each
    entity's own outcome, so a positive sitting in a thinly-populated context
    would otherwise read its own label back out of the population rate.

    Reuse this when scoring a later batch. Entities absent from `own_*`
    subtract nothing, which is correct for ones the model has not seen.
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
        """Long-format view of the fitted rates, for inspection."""
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
        return (pd.DataFrame(rows)
                .sort_values(["kind", "n_at_risk"], ascending=[True, False])
                .reset_index(drop=True))


def _terminal_mask(work: pd.DataFrame, cfg: FeatureEngineeringConfig) -> pd.Series:
    """Rows that mark the outcome, from a flag column or from event_col values.

    A flag counts as set when it is non-null and not zero, so 1/True/"Y" all
    work and an all-null column on non-outcome rows is simply absent.
    """
    mask = pd.Series(False, index=work.index)
    if cfg.terminal_flag_col:
        if cfg.terminal_flag_col not in work.columns:
            raise KeyError(f"terminal_flag_col '{cfg.terminal_flag_col}' not found in input dataframe")
        flag = work[cfg.terminal_flag_col]
        mask |= flag.notna() & (flag != 0)
    if cfg.terminal_event_states and cfg.event_col:
        mask |= work[cfg.event_col].isin(set(cfg.terminal_event_states))
    return mask


def _sequences(df: pd.DataFrame, id_col: str, value_col: str) -> Dict[Hashable, List[Any]]:
    """Each entity's states, oldest first, nulls dropped. Rows must be sorted."""
    valid = df.loc[df[value_col].notna(), [id_col, value_col]]
    return {id_: list(states) for id_, states in valid.groupby(id_col, sort=False)[value_col]}


def _fit_hazard(
    sequences: Dict[Hashable, List[Any]], terminal: set, orders: Sequence[int], runlen_cap: int,
) -> HazardModel:
    """Person-period expansion: one at-risk row per observation before the event.

    Each row carries a 1 if the terminal event followed it and a 0 otherwise. An
    entity that never reached the event contributes all of its observations as
    zeros, INCLUDING its last one -- that is a known non-event rather than a
    censored row, because the observation window runs past it, which is what
    makes the entity a negative.

    Dropping that final observation would remove only zeros, and only from the
    entities that never converted, biasing every rate upward by roughly one over
    the mean history length. On short histories that is not a rounding error: it
    inflated the strongest context by ~40% in testing.

    The exception is a history truncated by the data pull rather than by the
    label window, where the last row's outcome is genuinely unknown and counting
    it as a zero is optimistic. Exclude such entities from the fitting
    population, or fit on an earlier window and pass the model in prefit.
    """
    base = [0, 0]
    own_base: Dict[Hashable, List[int]] = defaultdict(lambda: [0, 0])
    by_runlen: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    own_by_runlen: Dict[Tuple[Hashable, int], List[int]] = defaultdict(lambda: [0, 0])
    by_order = {k: defaultdict(lambda: [0, 0]) for k in orders}
    own_by_order = {k: defaultdict(lambda: [0, 0]) for k in orders}

    for id_, states in sequences.items():
        for i, state in enumerate(states):
            if state in terminal:
                break  # the window ended here; nothing after it is at risk
            event = 1 if i + 1 < len(states) and states[i + 1] in terminal else 0

            for cell in (base, own_base[id_]):
                cell[0] += 1
                cell[1] += event

            bucket = min(i + 1, runlen_cap)
            for cell in (by_runlen[bucket], own_by_runlen[(id_, bucket)]):
                cell[0] += 1
                cell[1] += event

            for order in orders:
                if i + 1 < order:
                    continue
                ctx = tuple(states[i - order + 1 : i + 1])
                for cell in (by_order[order][ctx], own_by_order[order][(id_, ctx)]):
                    cell[0] += 1
                    cell[1] += event

    def freeze(counter: Dict[Any, List[int]]) -> Dict[Any, Counts]:
        return {key: (val[0], val[1]) for key, val in counter.items()}

    return HazardModel(
        base=(base[0], base[1]),
        by_order={k: freeze(v) for k, v in by_order.items()},
        by_runlen=freeze(by_runlen),
        own_base=freeze(own_base),
        own_by_order={k: freeze(v) for k, v in own_by_order.items()},
        own_by_runlen=freeze(own_by_runlen),
        runlen_cap=runlen_cap,
    )


def _loo_rate(total: Optional[Counts], own: Counts, min_support: int) -> Optional[float]:
    """Population rate for one context with `own`'s contribution removed.

    None means the context is too thin to trust once the entity's own rows are
    taken out, and the caller should back off to a shorter context.
    """
    if total is None:
        return None
    n_at_risk = total[0] - own[0]
    n_events = total[1] - own[1]
    if n_at_risk < min_support or n_at_risk <= 0:
        return None
    return n_events / n_at_risk


def _hazard_layer(
    fit_df: pd.DataFrame, work: pd.DataFrame, cfg: FeatureEngineeringConfig,
    index: pd.Index, prefit: Optional[HazardModel],
) -> Tuple[pd.DataFrame, Optional[HazardModel]]:
    """How likely the terminal event is from where each entity currently stands.

    `fit_df` still contains the terminal-event rows and is used only to fit the
    population hazard. `work` has them stripped and is the only frame any
    per-entity feature is read from.
    """
    id_col = cfg.id_col
    terminal = {_TERMINAL}
    orders = sorted({int(k) for k in cfg.context_orders if int(k) >= 1}, reverse=True)

    # With one distinct non-outcome state there is nothing to condition on, so
    # every context feature would be the same number for every entity. Emit the
    # run-length hazard alone rather than three constant columns.
    has_context = int(work[_STATE].nunique(dropna=True)) >= 2
    if not has_context:
        warnings.warn(
            "no event_col, or only one distinct state, so there is no context to condition on -- "
            "emitting hazard_runlen only. Set event_col to a behavioural state column to get the "
            "context hazard as well."
        )

    hazard = prefit
    if hazard is None:
        fit_seqs = _sequences(fit_df.sort_values([id_col, "__t__"], kind="mergesort"),
                              id_col, _STATE)
        if not fit_seqs:
            warnings.warn("no usable history; hazard features skipped")
            return pd.DataFrame(index=index), None
        hazard = _fit_hazard(fit_seqs, terminal, orders, cfg.runlen_cap)

    rows: List[Dict[str, Any]] = []
    for id_, states in _sequences(work, id_col, _STATE).items():
        n_obs = len(states)
        base_rate = _loo_rate(hazard.base, hazard.own_base.get(id_, (0, 0)), 1)
        if base_rate is None:
            base_rate = hazard.base_rate

        bucket = min(n_obs, hazard.runlen_cap)
        by_runlen = _loo_rate(
            hazard.by_runlen.get(bucket), hazard.own_by_runlen.get((id_, bucket), (0, 0)),
            cfg.hazard_min_support,
        )

        context_rate, context_order = None, 0
        for order in orders:  # longest context first, backing off when too thin
            if n_obs < order:
                continue
            ctx = tuple(states[-order:])
            rate = _loo_rate(
                hazard.by_order[order].get(ctx),
                hazard.own_by_order[order].get((id_, ctx), (0, 0)),
                cfg.hazard_min_support,
            )
            if rate is not None:
                context_rate, context_order = rate, order
                break
        if context_rate is None:
            context_rate = base_rate

        row = {id_col: id_, "hazard_runlen": base_rate if by_runlen is None else by_runlen}
        if has_context:
            row["hazard_context"] = context_rate
            row["hazard_context_order"] = context_order  # 0 = fell back to the base rate
            row["hazard_lift"] = context_rate / base_rate if base_rate else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame(index=index), hazard
    return pd.DataFrame(rows).set_index(id_col).reindex(index), hazard


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _resolve_cutoff(work: pd.DataFrame, id_col: str, as_of: Any, unit: str) -> pd.Series:
    """Per-entity cutoff on the float time axis.

    None means one cutoff for the whole population at the latest timestamp in
    the data. The alternative that looks natural -- each entity's own last row
    -- makes `recency` identically zero for everyone, so it is not a default.
    """
    ids = pd.Index(work[id_col].unique(), name=id_col)
    if as_of is None:
        return pd.Series(float(work["__t__"].max()), index=ids)

    if isinstance(as_of, dict):
        as_of = pd.Series(as_of)
    if isinstance(as_of, pd.Series):
        cutoff = _to_axis(as_of, unit)
        cutoff.index = as_of.index
        cutoff = cutoff.reindex(ids)
        missing = cutoff.isna()
        if missing.any():
            warnings.warn(
                f"{int(missing.sum())} entity/entities have no as_of value; falling back to the "
                "population maximum. Pass a cutoff for every entity to avoid mixing cutoffs."
            )
            cutoff = cutoff.fillna(float(work["__t__"].max()))
        cutoff.index.name = id_col
        return cutoff

    return pd.Series(float(_to_axis([as_of], unit).iloc[0]), index=ids)


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
    as_of : point-in-time cutoff. None (default) uses one cutoff for the whole
        population at the latest timestamp present. Pass a per-entity
        Series/dict, or a single timestamp, when the table will be joined to a
        historical outcome -- the cutoff must sit strictly before the outcome is
        observed.
    column_roles : skip inference and use these roles (as returned in
        `artifacts.column_roles`). Pass this when scoring a later batch so the
        schema cannot drift with the data.
    hazard : a previously fitted `HazardModel`, used instead of refitting. Only
        relevant when `terminal_event_states` is configured.
    """
    id_col, time_col = config.id_col, config.time_col
    for col in (id_col, time_col):
        if col not in df.columns:
            raise KeyError(f"column '{col}' not found in input dataframe")

    unit = _detect_time_unit(df[time_col])
    work = df.copy()
    work["__t__"] = _to_axis(work[time_col], unit)

    unusable = int(work["__t__"].isna().sum())
    if unusable:
        warnings.warn(f"dropping {unusable} row(s) with an unparseable or missing {time_col}")
        work = work[work["__t__"].notna()]
    if work.empty:
        raise ValueError("no rows with a usable timestamp")

    declares_outcome = bool(config.terminal_event_states) or bool(config.terminal_flag_col)
    if config.terminal_event_states and config.event_col is None:
        raise ValueError(
            "terminal_event_states names values inside event_col, which is not set. Use "
            "terminal_flag_col instead when the outcome is marked by its own flag column."
        )
    if declares_outcome and as_of is None:
        raise ValueError(
            "as_of is required when terminal_event_states is configured: without it the cutoff "
            "falls after the event for positives and so encodes the label. Pass the per-entity "
            "decision-time cutoff the label was defined against."
        )

    cutoff = _resolve_cutoff(work, id_col, as_of, unit)
    work = work[work["__t__"] <= work[id_col].map(cutoff)]
    if work.empty:
        raise ValueError("no rows remain at or before the as_of cutoff(s)")
    work = work.sort_values([id_col, "__t__"], kind="mergesort")

    # The hazard is a population object and has to see outcomes, so it is fit
    # from the pre-strip frame; every feature is read from the stripped one.
    fit_df = work
    if declares_outcome:
        is_terminal = _terminal_mask(work, config)
        if not is_terminal.any():
            raise ValueError(
                "the outcome declaration matched no rows at/before the cutoff. Left unfixed this "
                "would count outcome rows as ordinary events and hand the label to n_events and "
                "recency, so it is an error rather than a no-op. Check that terminal_flag_col is "
                "the column set on outcome rows, or that terminal_event_states holds values that "
                f"actually appear in {config.event_col!r}."
            )

        # The window is supposed to END at the outcome; anything after one means
        # either the cutoff is past it or the outcome is not really once-only.
        after = is_terminal.groupby(work[id_col]).cumsum() - is_terminal.astype(int)
        if int((after > 0).sum()):
            warnings.warn(
                f"{int((after > 0).sum())} row(s) fall after an outcome row inside the same "
                "entity's window. The window should end at the outcome, so this means the cutoff "
                "is past it or the outcome repeats -- either way the label is ambiguous."
            )

        # One state per row for the hazard: the entity's behaviour where known,
        # a placeholder where not, and the terminal marker on outcome rows.
        if config.event_col is None:
            state = pd.Series(_OBSERVED, index=work.index, dtype=object)
        else:
            state = work[config.event_col].astype(object).fillna(_OBSERVED)
        work[_STATE] = state.where(~is_terminal, _TERMINAL)

        fit_df = work
        before = work[id_col].nunique()
        work = work[~is_terminal]
        if work.empty:
            raise ValueError("every row at/before the cutoff is an outcome row")
        lost = before - work[id_col].nunique()
        if lost:
            warnings.warn(
                f"{lost} entity/entities had no history other than the outcome row and are "
                "absent from the table. They are not scoreable, but dropping them changes the "
                "population -- handle them explicitly rather than letting them vanish."
            )

    cutoff = cutoff.reindex(pd.Index(work[id_col].unique(), name=id_col))
    work["__cutoff__"] = work[id_col].map(cutoff)

    if column_roles is None:
        column_roles = infer_column_roles(
            work.drop(columns=[c for c in _INTERNAL if c in work.columns]), id_col, time_col,
            max_categories=config.max_categories, level_autocorr=config.level_autocorr,
            sample_rows=config.infer_sample_rows, ignore_cols=config.ignore_cols,
        )
    roles = {k: list(v) for k, v in column_roles.items()}
    for role, override in (
        ("flow", config.flow_cols), ("level", config.level_cols),
        ("categorical", config.categorical_cols), ("static", config.static_cols),
    ):
        if override is not None:
            roles[role] = [c for c in override if c in work.columns]
    for role in ("flow", "level", "categorical", "static"):
        roles[role] = [c for c in roles.get(role, []) if c not in set(config.ignore_cols)]
    roles.setdefault("skipped", [])
    if config.terminal_flag_col:  # describes the outcome, never the entity
        for role in ("flow", "level", "categorical", "static"):
            roles[role] = [c for c in roles[role] if c != config.terminal_flag_col]

    windows = _window_masks(work, "__cutoff__", "__t__", config.recent_windows)
    universe: Dict[str, List[Any]] = {k: list(v) for k, v in (config.category_universe or {}).items()}

    layers: Dict[str, pd.DataFrame] = {"activity": _activity_layer(work, config, cutoff, windows)}
    index = layers["activity"].index
    layers["flow"] = _flow_layer(work, config, roles["flow"], windows, index)
    layers["level"] = _level_layer(work, config, roles["level"], index)
    layers["categorical"] = _categorical_layer(work, config, roles["categorical"], universe, index)
    layers["static"] = _static_layer(work, config, roles["static"], universe, index)

    if declares_outcome:
        layers["hazard"], hazard = _hazard_layer(fit_df, work, config, index, hazard)
    else:
        hazard = None

    table = pd.concat([f for f in layers.values() if not f.empty], axis=1)
    table.index.name = id_col

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
    """Four logs that share nothing but the (entity, time, ...) shape."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, Any]] = {}

    # 1. card payments: amounts and labels, no state column, no outcome
    n = 20_000
    out["card_payments"] = {
        "df": pd.DataFrame({
            "account_id": rng.integers(0, 3_000, n),
            "posted_at": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 180, n), "D"),
            "amount": np.round(rng.gamma(2.0, 60.0, n), 2),
            "merchant_category": rng.choice(["grocery", "fuel", "travel", "dining", "online"], n),
            "channel": rng.choice(["chip", "online", "contactless"], n),
        }),
        "cfg": FeatureEngineeringConfig(id_col="account_id", time_col="posted_at"),
    }

    # 2. clickstream: mostly categorical, minute-level timestamps
    n = 30_000
    out["clickstream"] = {
        "df": pd.DataFrame({
            "user_id": rng.integers(0, 4_000, n),
            "ts": pd.Timestamp("2026-03-01") + pd.to_timedelta(rng.integers(0, 43_200, n), "m"),
            "page": rng.choice(["home", "search", "product", "cart", "checkout", "help"], n),
            "device": rng.choice(["ios", "android", "web"], n),
            "dwell_seconds": np.round(rng.exponential(45.0, n), 1),
        }),
        "cfg": FeatureEngineeringConfig(id_col="user_id", time_col="ts", recent_windows=(7, 30)),
    }

    # 3. sensor readings: INTEGER time axis, one true level and one true flow column
    devices, per_device = 500, 40
    walk = rng.normal(0, 0.4, (devices, per_device)).cumsum(axis=1)
    out["sensor_readings"] = {
        "df": pd.DataFrame({
            "device_id": np.repeat(np.arange(devices), per_device),
            "reading_no": np.tile(np.arange(per_device), devices),
            "temperature_c": np.round((20 + walk).ravel(), 2),                        # persistent
            "power_draw_wh": np.round(rng.gamma(3.0, 1.5, devices * per_device), 2),  # additive
            "firmware": np.repeat(rng.choice(["v1", "v2", "v3"], devices), per_device),
        }),
        "cfg": FeatureEngineeringConfig(id_col="device_id", time_col="reading_no",
                                        recent_windows=(5, 20)),
    }

    # 4. the NBO case: short histories truncated at a once-only outcome
    states = ["browse", "offer_view", "offer_click", "service_call", "statement_view"]
    transition = {
        "browse": [0.35, 0.30, 0.10, 0.10, 0.15], "offer_view": [0.25, 0.25, 0.30, 0.10, 0.10],
        "offer_click": [0.20, 0.25, 0.25, 0.15, 0.15], "service_call": [0.30, 0.15, 0.10, 0.30, 0.15],
        "statement_view": [0.35, 0.20, 0.10, 0.10, 0.25],
    }
    true_hazard = {"browse": 0.02, "offer_view": 0.07, "offer_click": 0.22,
                   "service_call": 0.04, "statement_view": 0.03}
    rows: List[Dict[str, Any]] = []
    labels: Dict[int, int] = {}
    cutoffs: Dict[int, pd.Timestamp] = {}
    for cust_id in range(4_000):
        state = str(rng.choice(states))
        stamp = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(0, 120)))
        balance, label = float(rng.uniform(5_000, 50_000)), 0
        for _ in range(int(np.clip(rng.geometric(0.22), 1, 20))):
            balance *= float(rng.uniform(0.95, 1.08))
            segment = "private" if balance > 30_000 else "retail"
            rows.append({"customer_id": cust_id, "txn_ts": stamp, "event": state,
                         "txn_amount": round(float(rng.uniform(10, 2_000)), 2),
                         "balance": round(balance, 2), "segment": segment,
                         "outcome_flag": np.nan})
            stamp += pd.Timedelta(days=float(rng.uniform(1, 40)))
            if rng.random() < true_hazard[state]:
                # the outcome row as it actually arrives: an id, a date and a
                # flag saying WHEN it happened. Every other column is null
                # because the row does not describe the event, only marks it.
                rows.append({"customer_id": cust_id, "txn_ts": stamp, "event": None,
                             "txn_amount": np.nan, "balance": np.nan, "segment": None,
                             "outcome_flag": 1})
                label = 1
                break
            state = str(rng.choice(states, p=transition[state]))
        labels[cust_id], cutoffs[cust_id] = label, stamp

    out["nbo_events"] = {
        "df": pd.DataFrame(rows),
        "cfg": FeatureEngineeringConfig(
            id_col="customer_id", time_col="txn_ts", event_col="event",
            terminal_flag_col="outcome_flag", context_orders=(1, 2),
        ),
        "as_of": pd.Series(cutoffs),
        "labels": pd.Series(labels),
    }
    return out


def _auc(scores: pd.Series, labels: pd.Series) -> float:
    """Rank AUC (Mann-Whitney), NaN-tolerant. Used only for the demo leakage check."""
    frame = pd.DataFrame({"s": np.asarray(scores, dtype=float), "y": np.asarray(labels)}).dropna()
    n_pos, n_neg = int((frame.y == 1).sum()), int((frame.y == 0).sum())
    if n_pos == 0 or n_neg == 0 or frame.s.nunique() < 2:
        return float("nan")
    ranks = frame.s.rank()
    return float((ranks[frame.y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


if __name__ == "__main__":
    for name, case in _demo_datasets().items():
        df, cfg = case["df"], case["cfg"]
        artifacts = build_feature_table(df, cfg, as_of=case.get("as_of"))
        table = artifacts.table

        per_id = df.groupby(cfg.id_col).size()
        print(f"\n{'=' * 78}")
        print(f"{name}: {len(df):,} rows, {df[cfg.id_col].nunique():,} entities, "
              f"median {per_id.median():.0f} rows each, time measured in {artifacts.time_unit}s")
        print("  roles: " + " | ".join(
            f"{role}={cols}" for role, cols in artifacts.column_roles.items() if cols))
        print(f"  table {table.shape[0]:,} x {table.shape[1]}  ->  " + ", ".join(
            f"{layer} {len(cols)}" for layer, cols in artifacts.feature_columns_by_layer.items()))
        null_rate = table.drop(columns=cfg.id_col).isna().mean()
        worst = null_rate.sort_values(ascending=False)
        print(f"  columns over 50% null: {int((null_rate > 0.5).sum())} of {len(null_rate)}"
              + (f" (worst: {worst.index[0]} {worst.iloc[0]:.0%})" if len(worst) else ""))

        if "labels" in case:
            y = case["labels"].reindex(table[cfg.id_col]).reset_index(drop=True)
            numeric = [c for c in table.columns
                       if c != cfg.id_col and pd.api.types.is_numeric_dtype(table[c])]
            aucs = pd.Series({c: _auc(table[c], y) for c in numeric}).dropna()
            top = aucs.sort_values(ascending=False).head(3)
            print(f"  hazard base rate {artifacts.hazard.base_rate:.3f}, "
                  f"top AUC: {', '.join(f'{k} {v:.3f}' for k, v in top.items())}")
            print(f"  max single-feature AUC {aucs.max():.3f} (>0.99 would mean a leak)")
    print(f"\n{'=' * 78}")
