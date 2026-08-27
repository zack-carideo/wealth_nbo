"""Generalized Markov chain estimation, quantification, and population querying.

Input contract
--------------
A tidy DataFrame with three named columns: an id column, a timestamp column
(anything sortable), and a state/value column (the thing whose sequence you
want to model — a category, an event type, a bucketed score, etc.). Rows do
not need to be pre-sorted; sequencing per id is handled internally.

Pipeline
--------
1. `_build_sequences`   — sorts each id's events by timestamp, optionally
                           breaks a stale gap into separate sessions, and
                           optionally collapses consecutive repeated states.
2. `fit_markov_chains`  — builds order-k transition counts / probabilities /
                           support / entropy / stationary distribution for
                           each requested order (k = context length).
3. `score_sequences`    — scores each id's own sequence against a fitted
                           population model (log-likelihood, perplexity,
                           z-scored surprisal) to flag unusual sequences.
4. `find_ids_in_context`— given an order-k LHS context, returns the ids
                           whose MOST RECENT k states currently match it
                           (i.e. who is sitting in that context right now),
                           plus the model's predicted next-state distribution
                           for that context if a fitted model is supplied.

Caveats (read before treating scores as ground truth)
------------------------------------------------------
- `score_sequences` scores each id against a model fit on the WHOLE
  population, including that id's own transitions (in-sample, not
  held-out). This understates anomalousness for high-volume ids who
  dominate the contexts they're being scored against. Fine for a first
  ranking pass; not a substitute for leave-one-out validation if this
  feeds a compliance/fraud decision.
- Stationary distributions are computed over order-k context tuples via
  power iteration and can be non-degenerate even for contexts that were
  never observed transitioning further (treated as absorbing).
- Ties in the timestamp column are broken by original row order
  (stable sort) — if simultaneous events have a true underlying order,
  encode it in the timestamp column before calling this.
- Not vectorized for very large populations (pure-Python loops over
  per-id sequences); fine for typical NBO/customer-level populations,
  revisit if scoring tens of millions of ids.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

State = Hashable
Context = Tuple[State, ...]

__all__ = [
    "OrderModel",
    "fit_markov_chains",
    "score_sequences",
    "find_ids_in_context",
    "analyze_markov_chains",
]


@dataclass
class OrderModel:
    """Fitted order-k Markov chain: transition counts/probs plus summary stats."""

    order: int
    counts: Dict[Context, Counter]
    probs: Dict[Context, Dict[State, float]]
    support: Dict[Context, int]
    stationary: Dict[Context, float]
    entropy: Dict[Context, float]

    def to_frame(self) -> pd.DataFrame:
        """Long-format table: one row per (lhs context, rhs state)."""
        rows = []
        for lhs, dist in self.probs.items():
            for rhs, p in dist.items():
                rows.append(
                    {
                        "order": self.order,
                        "lhs": lhs,
                        "rhs": rhs,
                        "prob": p,
                        "count": self.counts[lhs][rhs],
                        "lhs_support": self.support[lhs],
                        "lhs_entropy_bits": self.entropy[lhs],
                        "lhs_stationary_prob": self.stationary.get(lhs, np.nan),
                    }
                )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "order", "lhs", "rhs", "prob", "count",
                    "lhs_support", "lhs_entropy_bits", "lhs_stationary_prob",
                ]
            )
        return (
            pd.DataFrame(rows)
            .sort_values(["lhs_support", "lhs", "prob"], ascending=[False, True, False])
            .reset_index(drop=True)
        )

    def to_matrix(self) -> pd.DataFrame:
        """Wide-format transition probability matrix: rows=lhs context, cols=rhs state."""
        frame = self.to_frame()
        if frame.empty:
            return pd.DataFrame()
        return frame.pivot_table(index="lhs", columns="rhs", values="prob", fill_value=0.0)


@dataclass
class _Sequence:
    id_: Hashable
    session: int
    states: List[State]
    timestamps: List[Any]


def _build_sequences(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
    max_gap: Optional[pd.Timedelta] = None,
    collapse_repeats: bool = False,
    min_length: int = 2,
) -> List[_Sequence]:
    """Per-id (optionally per-session) state sequences, oldest first.

    `min_length` is the shortest sequence kept. The default of 2 is what
    transition counting needs; pass 1 when the caller wants a row for every
    id that has any history at all (e.g. describing an id's current state
    rather than its transitions) -- otherwise every single-observation id is
    silently dropped from the result.
    """
    for col in (id_col, time_col, value_col):
        if col not in df.columns:
            raise KeyError(f"column '{col}' not found in input dataframe")

    work = df[[id_col, time_col, value_col]].dropna()
    work = work.sort_values([id_col, time_col], kind="mergesort")

    sequences: List[_Sequence] = []
    for id_, grp in work.groupby(id_col, sort=False):
        times = grp[time_col].to_list()
        states = grp[value_col].to_list()

        if max_gap is not None:
            breaks = [0]
            for i in range(1, len(times)):
                if times[i] - times[i - 1] > max_gap:
                    breaks.append(i)
            breaks.append(len(times))
        else:
            breaks = [0, len(times)]

        for session, (start, end) in enumerate(zip(breaks[:-1], breaks[1:])):
            seg_states = states[start:end]
            seg_times = times[start:end]

            if collapse_repeats:
                collapsed_states: List[State] = []
                collapsed_times: List[Any] = []
                for s, t in zip(seg_states, seg_times):
                    if not collapsed_states or collapsed_states[-1] != s:
                        collapsed_states.append(s)
                        collapsed_times.append(t)
                seg_states, seg_times = collapsed_states, collapsed_times

            if len(seg_states) >= min_length:
                sequences.append(_Sequence(id_, session, seg_states, seg_times))

    return sequences


def _stationary_distribution(
    probs: Dict[Context, Dict[State, float]],
    order: int,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> Dict[Context, float]:
    """Power iteration over the order-k context graph: (s1..sk) --rhs--> (s2..sk,rhs)."""
    contexts = list(probs.keys())
    if not contexts:
        return {}

    dist: Dict[Context, float] = {c: 1.0 / len(contexts) for c in contexts}

    for _ in range(max_iter):
        nxt: Dict[Context, float] = defaultdict(float)
        for c, p_c in dist.items():
            transitions = probs.get(c)
            if not transitions:
                nxt[c] += p_c  # dead-end context observed in training: treat as absorbing
                continue
            for rhs, p in transitions.items():
                c_next = (c + (rhs,))[-order:]
                nxt[c_next] += p_c * p

        total = sum(nxt.values())
        if total == 0:
            break
        nxt = {c: v / total for c, v in nxt.items()}

        keys = set(nxt) | set(dist)
        diff = sum(abs(nxt.get(k, 0.0) - dist.get(k, 0.0)) for k in keys)
        dist = dict(nxt)
        if diff < tol:
            break

    return dist


def fit_markov_chains(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
    orders: Sequence[int] = (1, 2, 3),
    max_gap: Optional[pd.Timedelta] = None,
    collapse_repeats: bool = False,
    min_support: int = 1,
) -> Dict[int, OrderModel]:
    """Fit order-k Markov chains for each k in `orders` from raw event-level data.

    Parameters
    ----------
    df : DataFrame with at least id_col, time_col, value_col.
    max_gap : if given, a gap between consecutive events for the same id larger
        than this starts a new session (a fresh sequence) rather than bridging
        an unrelated episode.
    collapse_repeats : if True, consecutive identical states are collapsed
        before building transitions (self-transitions are not counted as moves).
    min_support : drop lhs contexts observed fewer than this many times total.

    Returns
    -------
    dict mapping order -> OrderModel. An order is omitted if no context met
    `min_support` (e.g. order=3 requested on too little data).
    """
    sequences = _build_sequences(df, id_col, time_col, value_col, max_gap, collapse_repeats)
    if not sequences:
        raise ValueError("no id had at least 2 events after preprocessing — nothing to fit")

    models: Dict[int, OrderModel] = {}
    for order in orders:
        counts: Dict[Context, Counter] = defaultdict(Counter)
        for seq in sequences:
            states = seq.states
            if len(states) < order + 1:
                continue
            for i in range(len(states) - order):
                lhs = tuple(states[i : i + order])
                rhs = states[i + order]
                counts[lhs][rhs] += 1

        counts = {lhs: c for lhs, c in counts.items() if sum(c.values()) >= min_support}
        if not counts:
            continue

        support = {lhs: sum(c.values()) for lhs, c in counts.items()}
        probs = {
            lhs: {rhs: n / support[lhs] for rhs, n in c.items()} for lhs, c in counts.items()
        }
        entropy = {
            lhs: -sum(p * math.log2(p) for p in dist.values() if p > 0)
            for lhs, dist in probs.items()
        }
        stationary = _stationary_distribution(probs, order)

        models[order] = OrderModel(
            order=order,
            counts=dict(counts),
            probs=probs,
            support=support,
            stationary=stationary,
            entropy=entropy,
        )

    if not models:
        raise ValueError(
            f"no order in {list(orders)} met min_support={min_support} — lower min_support "
            "or check that ids have enough events"
        )
    return models


def score_sequences(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
    model: OrderModel,
    max_gap: Optional[pd.Timedelta] = None,
    collapse_repeats: bool = False,
    laplace_alpha: float = 1e-6,
) -> pd.DataFrame:
    """Score each id's own sequence against a fitted population `model`.

    Returns one row per id with log-likelihood, perplexity, and a z-scored
    average surprisal (higher = the id's transitions were less predictable
    under the population model — a candidate anomaly signal, see module
    docstring for the in-sample-scoring caveat).
    """
    order = model.order
    sequences = _build_sequences(df, id_col, time_col, value_col, max_gap, collapse_repeats)

    vocab = sorted({rhs for dist in model.probs.values() for rhs in dist}, key=str)
    vocab_size = max(len(vocab), 1)
    fallback_p = laplace_alpha / vocab_size

    agg: Dict[Hashable, Dict[str, float]] = defaultdict(
        lambda: {"log_likelihood": 0.0, "n_transitions": 0, "n_sessions": 0}
    )
    for seq in sequences:
        states = seq.states
        if len(states) < order + 1:
            continue

        log_lik = 0.0
        n = 0
        for i in range(len(states) - order):
            lhs = tuple(states[i : i + order])
            rhs = states[i + order]
            dist = model.probs.get(lhs)
            p = fallback_p if dist is None else dist.get(rhs, fallback_p)
            log_lik += math.log(max(p, 1e-300))
            n += 1

        if n == 0:
            continue
        a = agg[seq.id_]
        a["log_likelihood"] += log_lik
        a["n_transitions"] += n
        a["n_sessions"] += 1

    rows = []
    for id_, a in agg.items():
        avg_surprisal = -a["log_likelihood"] / a["n_transitions"]
        rows.append(
            {
                id_col: id_,
                "order": order,
                "n_sessions": a["n_sessions"],
                "n_transitions": a["n_transitions"],
                "log_likelihood": a["log_likelihood"],
                "avg_surprisal_nats": avg_surprisal,
                "perplexity": math.exp(avg_surprisal),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        mu = out["avg_surprisal_nats"].mean()
        sigma = out["avg_surprisal_nats"].std(ddof=0)
        out["surprisal_zscore"] = (out["avg_surprisal_nats"] - mu) / sigma if sigma > 0 else 0.0
        out = out.sort_values("surprisal_zscore", ascending=False).reset_index(drop=True)
    return out


def find_ids_in_context(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
    order: int,
    lhs: Sequence[State],
    max_gap: Optional[pd.Timedelta] = None,
    collapse_repeats: bool = False,
    model: Optional[OrderModel] = None,
) -> pd.DataFrame:
    """Return ids whose MOST RECENT `order` states currently equal `lhs`.

    "Currently" = within each id's latest session (most recent event). If a
    `model` is supplied, the predicted next-state distribution for that
    context is attached to each row for convenience.
    """
    lhs = tuple(lhs)
    if len(lhs) != order:
        raise ValueError(f"lhs must have exactly {order} element(s) for order={order}, got {len(lhs)}")

    sequences = _build_sequences(df, id_col, time_col, value_col, max_gap, collapse_repeats)

    latest: Dict[Hashable, _Sequence] = {}
    for seq in sequences:
        current = latest.get(seq.id_)
        if current is None or seq.timestamps[-1] > current.timestamps[-1]:
            latest[seq.id_] = seq

    rows = []
    for id_, seq in latest.items():
        if len(seq.states) < order:
            continue
        current_context = tuple(seq.states[-order:])
        if current_context != lhs:
            continue

        row: Dict[str, Any] = {
            id_col: id_,
            "current_context": current_context,
            "as_of": seq.timestamps[-1],
            "n_events_in_session": len(seq.states),
        }
        if model is not None:
            dist = model.probs.get(lhs, {})
            if dist:
                likely_next, likely_p = max(dist.items(), key=lambda kv: kv[1])
                row["most_likely_next"] = likely_next
                row["most_likely_next_prob"] = likely_p
                row["next_state_distribution"] = dist
        rows.append(row)

    return pd.DataFrame(rows)


def analyze_markov_chains(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_col: str,
    orders: Sequence[int] = (1, 2, 3),
    max_gap: Optional[pd.Timedelta] = None,
    collapse_repeats: bool = False,
    min_support: int = 1,
    score: bool = True,
) -> Dict[str, Any]:
    """One-call entry point: fit chains for each order, optionally score ids against each.

    Returns {"models": {order: OrderModel}, "scores": {order: DataFrame}}.
    `find_ids_in_context` is called separately once you know which order/lhs
    you want to query — it isn't run here since the lhs to query is arbitrary.
    """
    models = fit_markov_chains(
        df, id_col, time_col, value_col,
        orders=orders, max_gap=max_gap, collapse_repeats=collapse_repeats, min_support=min_support,
    )
    result: Dict[str, Any] = {"models": models}
    if score:
        result["scores"] = {
            order: score_sequences(
                df, id_col, time_col, value_col, m,
                max_gap=max_gap, collapse_repeats=collapse_repeats,
            )
            for order, m in models.items()
        }
    return result


def _make_dummy_data(n_ids: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic per-customer event sequences for the __main__ demo below."""
    rng = np.random.default_rng(seed)
    states = ["browse", "offer_view", "offer_click", "purchase", "churn"]
    transition_bias = {
        "browse": [0.30, 0.35, 0.15, 0.05, 0.15],
        "offer_view": [0.25, 0.15, 0.35, 0.10, 0.15],
        "offer_click": [0.10, 0.10, 0.15, 0.45, 0.20],
        "purchase": [0.40, 0.20, 0.10, 0.10, 0.20],
        "churn": [0.80, 0.05, 0.05, 0.05, 0.05],  # rarely "reawakens"
    }

    rows = []
    base_day = pd.Timestamp("2026-01-01")
    for cust_id in range(n_ids):
        n_events = rng.integers(6, 20)
        start = base_day + pd.Timedelta(days=int(rng.integers(0, 45)))
        cur = "browse"
        for i in range(n_events):
            rows.append({"customer_id": cust_id, "event_ts": start + pd.Timedelta(hours=int(i)), "event": cur})
            cur = rng.choice(states, p=transition_bias[cur])

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # --- 1. dummy data: raw, unsorted, one row per (customer, timestamp, event) ---
    events = _make_dummy_data(n_ids=200, seed=0)
    print(f"dummy data: {len(events)} events across {events['customer_id'].nunique()} customers")
    print(events.head(), "\n")

    # --- 2. fit + score orders 1-3 in one call ---
    result = analyze_markov_chains(
        events,
        id_col="customer_id",
        time_col="event_ts",
        value_col="event",
        orders=(1, 2, 3),
        max_gap=pd.Timedelta(days=3),  # break a sequence if a customer goes quiet >3 days
        collapse_repeats=True,          # don't count staying in the same state as a "move"
        min_support=5,                  # ignore contexts seen fewer than 5 times
    )

    for order, model in result["models"].items():
        print(f"--- order {order} transition matrix ---")
        print(model.to_matrix().round(3), "\n")

    # --- 3. per-customer anomaly scoring (order-2 model) ---
    scores = result["scores"][2]
    print("--- 5 most surprising customers under the order-2 model ---")
    print(scores.head(5), "\n")

    # --- 4. population query: who is sitting in ('offer_view', 'offer_click') right now? ---
    hits = find_ids_in_context(
        events,
        id_col="customer_id",
        time_col="event_ts",
        value_col="event",
        order=2,
        lhs=("offer_view", "offer_click"),
        max_gap=pd.Timedelta(days=3),
        collapse_repeats=True,
        model=result["models"][2],
    )
    print(f"--- {len(hits)} customers currently in context ('offer_view', 'offer_click') ---")
    print(hits.head(5))
