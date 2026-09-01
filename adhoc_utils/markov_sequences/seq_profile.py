"""
Markov sequence profiling.

A standalone utility. Give it a dataframe, the column holding the state
you want to profile, and the column that defines what a sequence is.
It tells you how those states follow one another.

    import seq_profile as sp

    model = sp.fit(df, id_col="customer_id", state_col="product_family")
    print(sp.summarize(model))

    model.save("markov_product_family.json")
    scores = sp.score(df, model)          # one row per id

Assumptions
    * The frame is ALREADY SORTED, by id then by time. Nothing here
      re-sorts it. If the order is wrong the answers are wrong and
      nothing will complain.
    * Rows within an id are contiguous.
    * States are treated as strings. Nulls become the literal state
      "MISSING" so they show up in the profile rather than vanishing.

No dependencies beyond pandas and numpy. No config file: every knob is
an argument with a sane default.

--------------------------------------------------------------------
WHAT COMES OUT
--------------------------------------------------------------------
    states        frequency of each state, and how many ids ever see it
    lengths       distribution of sequence lengths
    transitions   from -> to, with counts, probabilities and gap stats
    from_states   per-origin summary: n, entropy, most likely next
    paths         most common contiguous runs of `path_length` states
    pairs         one row per transition WITH the id (in-memory only)
    path_rows     one row per run WITH the id (in-memory only)

    model.ids_for_transition("cd_standard", "heloc")   -> who did it
    model.ids_for_path("checking_basic>savings_standard>auto_loan")

--------------------------------------------------------------------
THINGS THAT ARE EASY TO GET WRONG HERE
--------------------------------------------------------------------
  * A transition table fit on all your data and then used to build
    features is a mild leak into cross-validation. It is unsupervised,
    so it is a weak one, but if you care: fit on training ids only and
    persist the model, exactly as you would a percentile grid.
  * `order=2` needs far more data than `order=1`. The number of cells
    grows as states^(order+1). Check the `thin_share` line in the
    summary before believing anything.
  * Boundaries are off by default. With them on, "what do people start
    with" and "what do they stop after" become readable, but the state
    frequencies now include the sentinels.
"""

import argparse
import json

import numpy as np
import pandas as pd

START = "<START>"
END = "<END>"
MISSING = "MISSING"


# =====================================================================
# the fitted object
# =====================================================================

class SequenceModel:
    """
    Holds the fitted transition table plus everything needed to reapply
    it later. Save it, load it at scoring time, never refit on the batch
    you are scoring.
    """

    def __init__(self, transitions, states, lengths, from_states, paths, params,
                 pairs=None, path_rows=None):
        self.transitions = transitions      # from_state, to_state, n, prob, ...
        self.states = states                # state, n, n_ids, share
        self.lengths = lengths              # per-id sequence lengths
        self.from_states = from_states      # per-origin summary
        self.paths = paths                  # common runs
        self.params = params                # order, alpha, min_count, ...
        # Row-level frames, kept only on the object returned by fit().
        # They are NOT written by save(), so a loaded model has None here.
        self.pairs = pairs                  # id, from_state, to_state[, gap_days]
        self.path_rows = path_rows          # id, path

    # -- who did what ------------------------------------------------

    def ids_for_transition(self, from_state, to_state):
        """
        Ids that made this transition, with how many times each did.
        Only available on a model straight from fit(); a loaded model
        raises because the row-level frame is not persisted.
        """
        self._need_rows(self.pairs, "pairs")
        id_col = self.params["id_col"]
        hit = self.pairs[(self.pairs["from_state"] == str(from_state))
                         & (self.pairs["to_state"] == str(to_state))]
        return (hit.groupby(id_col).size().rename("n").reset_index()
                   .sort_values(["n", id_col], ascending=[False, True])
                   .reset_index(drop=True))

    def ids_for_path(self, path):
        """
        Ids whose sequence contains this contiguous run, e.g.
        "checking_basic>savings_standard>auto_loan". Length must match
        the `path_length` the model was fit with.
        """
        self._need_rows(self.path_rows, "path_rows")
        id_col = self.params["id_col"]
        hit = self.path_rows[self.path_rows["path"] == str(path)]
        return (hit.groupby(id_col).size().rename("n").reset_index()
                   .sort_values(["n", id_col], ascending=[False, True])
                   .reset_index(drop=True))

    @staticmethod
    def _need_rows(frame, name):
        if frame is None:
            raise ValueError("model.%s is not available: row-level frames are "
                             "kept only on the object returned by fit(), not "
                             "on one reloaded with load()" % name)

    # -- lookup ------------------------------------------------------

    def prob(self, from_state, to_state):
        """Smoothed P(to | from). Never returns 0, so logs are safe."""
        key = (str(from_state), str(to_state))
        return self._prob_map().get(key, self._floor(str(from_state)))

    def _prob_map(self):
        if not hasattr(self, "_pmap"):
            self._pmap = {(r.from_state, r.to_state): r.prob_smoothed
                          for r in self.transitions.itertuples()}
        return self._pmap

    def _floor(self, from_state):
        """Mass reserved for a pair never seen from this origin."""
        if not hasattr(self, "_fmap"):
            alpha = self.params["alpha"]
            k = self.params["n_to_states"]
            self._fmap = {r.from_state: alpha / (r.n + alpha * k)
                          for r in self.from_states.itertuples()}
        # An origin never seen at all: uniform over the alphabet.
        return self._fmap.get(from_state, 1.0 / max(1, self.params["n_to_states"]))

    def matrix(self, value="prob"):
        """Wide from x to table. Handy for eyeballing or a heatmap."""
        return (self.transitions
                .pivot(index="from_state", columns="to_state", values=value)
                .fillna(0.0))

    # -- persistence -------------------------------------------------

    def save(self, path):
        payload = {"params": self.params,
                   "transitions": self.transitions.to_dict(orient="list"),
                   "states": self.states.to_dict(orient="list"),
                   "lengths": self.lengths.to_dict(orient="list"),
                   "from_states": self.from_states.to_dict(orient="list"),
                   "paths": self.paths.to_dict(orient="list")}
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            payload = json.load(fh)
        return cls(pd.DataFrame(payload["transitions"]),
                   pd.DataFrame(payload["states"]),
                   pd.DataFrame(payload["lengths"]),
                   pd.DataFrame(payload["from_states"]),
                   pd.DataFrame(payload["paths"]),
                   payload["params"])


# =====================================================================
# 1. prepare
# =====================================================================

def _prepare(df, id_col, state_col, time_col, add_boundaries):
    """Normalise states to strings and optionally pad each id."""
    keep = [id_col, state_col] + ([time_col] if time_col else [])
    out = df[keep].copy()
    out[state_col] = out[state_col].astype("object").where(
        out[state_col].notna(), MISSING).astype(str)
    if time_col:
        out[time_col] = pd.to_datetime(out[time_col])

    if not add_boundaries:
        return out.reset_index(drop=True)

    # Sentinel rows sort into place on a positional key rather than by
    # re-sorting the frame, so the caller's ordering survives intact.
    # Positions are spaced out so that the <END> of one id and the
    # <START> of the next cannot land on the same key and get ordered
    # by luck. That collision silently interleaves the sequences.
    out["_pos"] = np.arange(len(out)) * 3
    head = out.groupby(id_col, sort=False).head(1).copy()
    head[state_col] = START
    head["_pos"] -= 1
    tail = out.groupby(id_col, sort=False).tail(1).copy()
    tail[state_col] = END
    tail["_pos"] += 1
    if time_col:
        head[time_col] = pd.NaT
        tail[time_col] = pd.NaT

    padded = pd.concat([head, out, tail], ignore_index=True)
    padded = padded.sort_values("_pos", kind="mergesort").reset_index(drop=True)
    return padded.drop(columns=["_pos"])


def _pairs(prep, id_col, state_col, time_col, order):
    """
    One row per observed transition.

    For order k the origin is the k states ending at the current row,
    joined with ">". The destination is the next row. A window is valid
    only if the id is unchanged from its first element to its last,
    which is enough because rows within an id are contiguous.
    """
    ids = prep[id_col]
    states = prep[state_col]

    parts = [states.shift(k) for k in range(order - 1, -1, -1)]
    from_state = parts[0]
    for p in parts[1:]:
        from_state = from_state.str.cat(p, sep=">")

    valid = (ids.shift(order - 1) == ids.shift(-1)) & from_state.notna()

    pairs = pd.DataFrame({
        id_col: ids,
        "from_state": from_state,
        "to_state": states.shift(-1),
    })
    if time_col:
        pairs["gap"] = (prep[time_col].shift(-1) - prep[time_col])
        pairs["gap_days"] = pairs["gap"].dt.total_seconds() / 86400.0
        pairs = pairs.drop(columns=["gap"])
    return pairs[valid].reset_index(drop=True)


# =====================================================================
# 2. fit
# =====================================================================

def fit(df, id_col, state_col, time_col=None, order=1, alpha=0.5,
        min_count=30, add_boundaries=False, path_length=3, top_paths=25):
    """
    Profile the sequences of `state_col` within `id_col`.

    order            1 = plain Markov chain. 2 = the previous two states
                     predict the next. Cells explode fast; watch
                     thin_share in the summary.
    alpha            additive smoothing. Keeps log-probabilities finite
                     for pairs that were never observed.
    min_count        an origin seen fewer times than this is flagged
                     `thin`. Nothing is dropped; the flag is there so
                     you can decide.
    add_boundaries   insert <START> and <END> per id, so first and last
                     states become part of the chain.
    """
    prep = _prepare(df, id_col, state_col, time_col, add_boundaries)
    pairs = _pairs(prep, id_col, state_col, time_col, order)

    if pairs.empty:
        raise ValueError("no transitions found - every id has fewer than "
                         "%d rows, or the frame is not grouped by id" % (order + 1))

    # ---- state frequencies -----------------------------------------
    states = (prep.groupby(state_col)
                  .agg(n=(id_col, "size"), n_ids=(id_col, "nunique"))
                  .reset_index()
                  .rename(columns={state_col: "state"}))
    states["share"] = states["n"] / states["n"].sum()
    states = states.sort_values("n", ascending=False).reset_index(drop=True)

    to_states = sorted(pairs["to_state"].unique())
    n_to = len(to_states)

    # ---- sequence lengths ------------------------------------------
    lengths = (prep.groupby(id_col).size()
                   .rename("length").reset_index()
                   .rename(columns={id_col: "id"}))

    # ---- transitions ------------------------------------------------
    agg = {"n": ("to_state", "size")}
    if time_col:
        agg.update(gap_days_median=("gap_days", "median"),
                   gap_days_mean=("gap_days", "mean"))
    trans = (pairs.groupby(["from_state", "to_state"]).agg(**agg).reset_index())

    row_total = trans.groupby("from_state")["n"].transform("sum")
    trans["from_n"] = row_total
    trans["prob"] = trans["n"] / row_total
    trans["prob_smoothed"] = (trans["n"] + alpha) / (row_total + alpha * n_to)
    trans["thin"] = row_total < min_count
    # Lift over the destination's unconditional share: >1 means this
    # origin makes that destination more likely than chance.
    base = trans.groupby("to_state")["n"].transform("sum") / trans["n"].sum()
    trans["lift"] = trans["prob"] / base
    trans = trans.sort_values(["from_state", "n"],
                              ascending=[True, False]).reset_index(drop=True)

    # ---- per-origin summary ----------------------------------------
    from_states = _origin_summary(trans, n_to, min_count)

    # ---- common runs ------------------------------------------------
    paths, path_rows = _paths(prep, id_col, state_col, path_length, top_paths)

    params = {"id_col": id_col, "state_col": state_col, "time_col": time_col,
              "order": order, "alpha": alpha, "min_count": min_count,
              "add_boundaries": add_boundaries, "path_length": path_length,
              "n_to_states": n_to, "to_states": to_states,
              "n_ids": int(prep[id_col].nunique()),
              "n_rows": int(len(prep)), "n_transitions": int(len(pairs))}

    return SequenceModel(trans, states, lengths, from_states, paths, params,
                         pairs=pairs, path_rows=path_rows)


def _origin_summary(trans, n_to, min_count):
    """Entropy and the single most likely next state, per origin."""
    rows = []
    for origin, grp in trans.groupby("from_state"):
        p = grp["prob"].values
        h = float(-(p * np.log2(p)).sum()) + 0.0
        best = grp.iloc[0]
        rows.append({
            "from_state": origin,
            "n": int(grp["from_n"].iloc[0]),
            "n_distinct_next": int(len(grp)),
            "entropy_bits": h,
            # 0 = fully predictable, 1 = uniform over the whole alphabet.
            "entropy_norm": h / np.log2(n_to) if n_to > 1 else 0.0,
            "top_next": best["to_state"],
            "top_next_prob": float(best["prob"]),
            "thin": bool(grp["from_n"].iloc[0] < min_count),
        })
    return (pd.DataFrame(rows)
            .sort_values("n", ascending=False).reset_index(drop=True))


def _paths(prep, id_col, state_col, path_length, top_n):
    """
    Most common contiguous runs of `path_length` states.

    Returns two frames: the aggregate (path, n, share) that goes into the
    summary, and the row-level (id, path) frame it was counted from, so a
    path can be traced back to the ids that walked it.
    """
    if path_length < 2:
        return (pd.DataFrame(columns=["path", "n", "share"]),
                pd.DataFrame(columns=[id_col, "path"]))
    ids, states = prep[id_col], prep[state_col]

    joined = states.shift(path_length - 1)
    for k in range(path_length - 2, -1, -1):
        joined = joined.str.cat(states.shift(k), sep=">")
    valid = (ids.shift(path_length - 1) == ids) & joined.notna()

    path_rows = pd.DataFrame({id_col: ids[valid], "path": joined[valid]}
                             ).reset_index(drop=True)
    counts = path_rows["path"].value_counts()
    out = counts.head(top_n).rename_axis("path").reset_index(name="n")
    out["share"] = out["n"] / counts.sum()
    return out, path_rows


# =====================================================================
# 3. score  ->  per-id features
# =====================================================================

def score(df, model, id_col=None, state_col=None):
    """
    One row per id, describing how that id's sequence sits against the
    fitted chain. Four columns you can hand straight to a model:

        n_transitions     length of the chain actually scored
        mean_logprob      how typical this path is. Low = unusual.
        min_logprob       the single most surprising step
        unseen_rate       share of steps never observed at fit time
        mean_entropy      how predictable this id's positions were
        self_rate         share of steps that repeated the same state

    Prefer these over one-hot encoding the transition pairs. A handful
    of continuous columns costs far fewer degrees of freedom than a
    block of dummies, which matters when events are scarce.
    """
    id_col = id_col or model.params["id_col"]
    state_col = state_col or model.params["state_col"]
    p = model.params

    prep = _prepare(df, id_col, state_col, None, p["add_boundaries"])
    pairs = _pairs(prep, id_col, state_col, None, p["order"])
    if pairs.empty:
        return pd.DataFrame(columns=[id_col, "n_transitions", "mean_logprob",
                                     "min_logprob", "unseen_rate",
                                     "mean_entropy", "self_rate"])

    seen = set(model._prob_map())
    ent = dict(zip(model.from_states["from_state"],
                   model.from_states["entropy_norm"]))

    pairs["p"] = [model.prob(f, t) for f, t in
                  zip(pairs["from_state"], pairs["to_state"])]
    pairs["logp"] = np.log(pairs["p"])
    pairs["unseen"] = [(f, t) not in seen for f, t in
                       zip(pairs["from_state"], pairs["to_state"])]
    pairs["entropy"] = pairs["from_state"].map(ent)
    # The last element of the origin is the immediately preceding state.
    pairs["self"] = (pairs["from_state"].str.rsplit(">", n=1).str[-1]
                     == pairs["to_state"])

    return (pairs.groupby(id_col)
                 .agg(n_transitions=("logp", "size"),
                      mean_logprob=("logp", "mean"),
                      min_logprob=("logp", "min"),
                      unseen_rate=("unseen", "mean"),
                      mean_entropy=("entropy", "mean"),
                      self_rate=("self", "mean"))
                 .reset_index())


# =====================================================================
# 4. report
# =====================================================================

def summarize(model, top=12):
    """A readable text block. Print it, paste it into a ticket."""
    p = model.params
    L = model.lengths["length"]
    thin_share = float(model.transitions.loc[model.transitions["thin"], "n"].sum()
                       / model.transitions["n"].sum())

    lines = []
    add = lines.append
    add("sequence profile: %s within %s" % (p["state_col"], p["id_col"]))
    add("-" * 64)
    add("ids                %d" % p["n_ids"])
    add("rows               %d" % p["n_rows"])
    add("transitions        %d  (order %d)" % (p["n_transitions"], p["order"]))
    add("distinct states    %d" % len(model.states))
    add("length  min/med/p95/max   %d / %.0f / %.0f / %d"
        % (L.min(), L.median(), L.quantile(0.95), L.max()))
    add("thin share         %.1f%%  (transitions from an origin seen < %d times)"
        % (100 * thin_share, p["min_count"]))
    add("")

    add("STATES")
    for r in model.states.head(top).itertuples():
        add("  %-24s %8d  %5.1f%%  %6d ids" % (r.state, r.n, 100 * r.share, r.n_ids))
    add("")

    add("ORIGINS  (entropy_norm: 0 = one certain next state, 1 = anyone's guess)")
    for r in model.from_states.head(top).itertuples():
        flag = " *thin" if r.thin else ""
        add("  %-24s n=%-7d H=%.2f  ->%-20s %.2f%s"
            % (r.from_state, r.n, r.entropy_norm, r.top_next, r.top_next_prob, flag))
    add("")

    add("TOP TRANSITIONS  (lift vs the destination's overall share)")
    top_t = model.transitions.sort_values("n", ascending=False).head(top)
    for r in top_t.itertuples():
        gap = ""
        if "gap_days_median" in model.transitions.columns:
            gap = "  med %.0fd" % r.gap_days_median
        add("  %-22s -> %-22s n=%-7d p=%.2f  lift=%.2f%s"
            % (r.from_state, r.to_state, r.n, r.prob, r.lift, gap))
    add("")

    if len(model.paths):
        add("COMMON RUNS  (length %d)" % p["path_length"])
        for r in model.paths.head(top).itertuples():
            add("  %-46s %7d  %4.1f%%" % (r.path, r.n, 100 * r.share))
    return "\n".join(lines)


# =====================================================================
# CLI
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", required=True, help="csv, already sorted")
    ap.add_argument("--id", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--time", default=None, help="optional, adds gap stats")
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--boundaries", action="store_true")
    ap.add_argument("--path-length", type=int, default=3)
    ap.add_argument("--save", default=None, help="write the fitted model here")
    ap.add_argument("--scores", default=None, help="write per-id features here")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    model = fit(df, id_col=args.id, state_col=args.state, time_col=args.time,
                order=args.order, alpha=args.alpha, min_count=args.min_count,
                add_boundaries=args.boundaries, path_length=args.path_length)
    print(summarize(model))

    if args.save:
        print("\nmodel -> %s" % model.save(args.save))
    if args.scores:
        score(df, model).to_csv(args.scores, index=False)
        print("scores -> %s" % args.scores)


if __name__ == "__main__":
    main()
