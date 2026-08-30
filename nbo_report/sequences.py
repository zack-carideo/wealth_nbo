"""
Generic n-order Markov profiling of event sequences.

WHAT THIS IS FOR
----------------
Given a per-customer ordered list of events and a 0/1 outcome per
customer, answer: which short runs of events immediately precede the
outcome more often than chance, and how does the event-to-event
transition structure differ between customers who converted and
customers who did not.

NOTHING HERE KNOWS ABOUT CROSS-SELL. The alphabet is whatever values
`sequences.event_column` takes in the raw file, and the outcome is
whatever `preprocess.compute_anchors` marked as the event. Point the
config at a servicing-complaint column and an attrition flag and the
same functions profile the paths into attrition.

THE LEAKAGE RULE APPLIES HERE TOO
---------------------------------
Sequences are cut at the customer's anchor, using the anchor that
`preprocess.compute_anchors` computed -- not a date derived locally.
For a converter the anchor is their last acquisition BEFORE the
outcome, so the outcome itself and anything after it never enters a
sequence. Getting this wrong would produce a spectacular and completely
fake "the wealth product predicts the wealth product" finding.

The distance from the anchor to the outcome is also never used: an
n-gram is a run of event labels, with no timing attached.

READING THE OUTPUT
------------------
`lift` is P(convert | history ends with this run) / P(convert overall).
Lift of 2.0 means twice the base rate. Because the tail of the ranking
is where small cells collect, every row also carries a Wilson lower
bound on the rate; when `lift_lo95` drops under 1.0 the pattern is not
distinguishable from chance at that support and is flagged as such.
"""

import numpy as np
import pandas as pd

ARROW = " → "
START = "(start)"


def _rate(value):
    """
    NaN-safe base rate. NaN is truthy, so `value or 0` does NOT rescue
    an undefined rate -- it renders as "nan%" in the report.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


# =====================================================================
# building sequences
# =====================================================================

def build_sequences(acq, anchors, cfg_pre, event_column, anchor_schema):
    """
    Cut each kept customer's history at their anchor and return the
    ordered list of event labels.

    acq            acquisition rows (outcome rows already split away)
    anchors        output of preprocess.compute_anchors
    event_column   the column whose values form the event alphabet
    anchor_schema  names of the columns on the `anchors` frame
    returns        (sequences: {customer: [label, ...]},
                    labels:    {customer: 0/1})
    """
    id_col = cfg_pre["columns"]["id"]
    date_col = cfg_pre["columns"]["date"]
    if event_column not in acq.columns:
        raise ValueError(
            "sequences.event_column '%s' is not a column on the raw "
            "acquisition rows. Available: %s"
            % (event_column, ", ".join(map(str, acq.columns))))

    kept = anchors[anchors[anchor_schema["keep"]]]
    anchor_by_id = kept.set_index(anchor_schema["customer"])[
        anchor_schema["anchor_date"]].to_dict()
    event_by_id = kept.set_index(anchor_schema["customer"])[
        anchor_schema["event"]].to_dict()

    sequences, labels = {}, {}
    for cid, grp in acq.groupby(id_col, sort=True):
        anchor = anchor_by_id.get(cid)
        if anchor is None or pd.isna(anchor):
            continue
        # Same filter build_features uses: at or before the anchor.
        hist = grp[grp[date_col] <= anchor].sort_values(date_col,
                                                        kind="mergesort")
        if hist.empty:
            continue
        values = [str(v) for v in hist[event_column].tolist()]
        sequences[cid] = values
        labels[cid] = int(event_by_id.get(cid, 0))

    return sequences, labels


def alphabet(sequences):
    """Distinct event labels, most frequent first."""
    counts = {}
    for seq in sequences.values():
        for token in seq:
            counts[token] = counts.get(token, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])]


# =====================================================================
# terminal n-grams -- the run immediately preceding the anchor
# =====================================================================

def terminal_ngram(seq, order):
    """
    The last `order` events, left-padded so a customer with a shorter
    history is still represented rather than silently dropped.
    """
    tail = seq[-order:]
    if len(tail) < order:
        tail = [START] * (order - len(tail)) + tail
    return tuple(tail)


def ngram_lift_table(sequences, labels, order, min_support=1, top_n=None):
    """
    One row per distinct terminal n-gram:

        sequence        the run, oldest -> newest
        customers       how many customers end on it (the support)
        converters      how many of them converted
        rate            converters / customers
        lift            rate / base rate
        rate_lo95       Wilson lower bound on rate
        lift_lo95       that bound expressed as lift
        distinguishable whether lift_lo95 clears 1.0

    Rows below `min_support` are dropped: at low support lift is noise,
    and a table sorted by lift will otherwise be nothing but noise at
    the top.
    """
    rows = {}
    for cid, seq in sequences.items():
        key = terminal_ngram(seq, order)
        bucket = rows.setdefault(key, [0, 0])
        bucket[0] += 1
        bucket[1] += int(labels.get(cid, 0))

    total = sum(v[0] for v in rows.values())
    events = sum(v[1] for v in rows.values())
    base = (events / float(total)) if total else np.nan

    records = []
    for key, (n, k) in rows.items():
        if n < min_support:
            continue
        rate = k / float(n)
        lo = _wilson_lower(k, n)
        records.append({
            "sequence": ARROW.join(key),
            "customers": n,
            "share_of_population": n / float(total) if total else np.nan,
            "converters": k,
            "rate": rate,
            "lift": rate / base if base else np.nan,
            "rate_lo95": lo,
            "lift_lo95": lo / base if base else np.nan,
        })

    table = pd.DataFrame(records)
    if table.empty:
        return table, base
    table["distinguishable"] = table["lift_lo95"] > 1.0
    table = table.sort_values(["lift", "customers"],
                              ascending=[False, False]).reset_index(drop=True)
    return (table if top_n is None else table.head(top_n)), base


def _wilson_lower(successes, n, z=1.96):
    """
    Lower bound of the Wilson score interval. Used instead of a raw rate
    so a 3-of-4 cell cannot outrank a 300-of-1000 one on a report that
    somebody is about to spend a marketing budget against.
    """
    if n == 0:
        return np.nan
    p = successes / float(n)
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


# =====================================================================
# transition structure
# =====================================================================

def transition_counts(sequences, order=1, customers=None):
    """
    Count from-state -> next-event pairs. For order k the from-state is
    a tuple of k consecutive events, which is what makes this an n-order
    Markov profile rather than a first-order one.
    """
    counts = {}
    for cid, seq in sequences.items():
        if customers is not None and cid not in customers:
            continue
        for i in range(len(seq) - order):
            source = ARROW.join(seq[i:i + order])
            target = seq[i + order]
            counts.setdefault(source, {})
            counts[source][target] = counts[source].get(target, 0) + 1
    return counts


def transition_matrix(counts, row_states=None, col_states=None):
    """
    Row-normalised P(next | from). Returns (matrix, rows, cols) with
    rows ordered by how much traffic they carry.
    """
    if not counts:
        return np.zeros((0, 0)), [], []

    volume = {src: sum(dst.values()) for src, dst in counts.items()}
    rows = row_states or [s for s, _ in sorted(volume.items(),
                                               key=lambda kv: -kv[1])]
    if col_states is None:
        col_volume = {}
        for dst in counts.values():
            for target, n in dst.items():
                col_volume[target] = col_volume.get(target, 0) + n
        col_states = [s for s, _ in sorted(col_volume.items(),
                                           key=lambda kv: -kv[1])]

    matrix = np.full((len(rows), len(col_states)), np.nan)
    for i, source in enumerate(rows):
        dst = counts.get(source, {})
        total = float(sum(dst.values()))
        if total == 0:
            continue
        for j, target in enumerate(col_states):
            matrix[i, j] = dst.get(target, 0) / total
    return matrix, list(rows), list(col_states)


def comparison_matrices(sequences, labels, order=1, top_states=8):
    """
    Converter and non-converter transition matrices on a shared grid,
    plus their difference.

    The difference is the one worth looking at. Two sequential heatmaps
    side by side mostly show that both groups buy checking accounts; the
    difference shows where the two populations actually diverge.
    """
    converters = {c for c, v in labels.items() if v == 1}
    others = {c for c, v in labels.items() if v == 0}

    all_counts = transition_counts(sequences, order)
    if not all_counts:
        return None

    volume = {src: sum(dst.values()) for src, dst in all_counts.items()}
    rows = [s for s, _ in sorted(volume.items(), key=lambda kv: -kv[1])][:top_states]
    col_volume = {}
    for dst in all_counts.values():
        for target, n in dst.items():
            col_volume[target] = col_volume.get(target, 0) + n
    cols = [s for s, _ in sorted(col_volume.items(),
                                 key=lambda kv: -kv[1])][:top_states]

    conv, _, _ = transition_matrix(
        transition_counts(sequences, order, converters), rows, cols)
    non, _, _ = transition_matrix(
        transition_counts(sequences, order, others), rows, cols)

    with np.errstate(invalid="ignore"):
        difference = conv - non

    return {
        "rows": rows,
        "cols": cols,
        "converter": conv,
        "non_converter": non,
        "difference": difference,
        "n_converters": len(converters),
        "n_others": len(others),
    }


# =====================================================================
# one call that does the whole profile
#
# eda.py runs this over the whole eligible population; main.py runs the
# SAME function over the high-propensity subset. Sharing the function
# rather than the output is the point -- the two reports are then
# guaranteed to be measuring the same thing.
# =====================================================================

def profile(sequences, labels, cfg_seq, customers=None):
    """
    Returns a dict with an n-gram table per configured order plus the
    transition comparison. `customers` restricts the population without
    changing any of the logic.
    """
    if customers is not None:
        customers = set(customers)
        sequences = {c: s for c, s in sequences.items() if c in customers}
        labels = {c: v for c, v in labels.items() if c in customers}

    result = {
        "n_customers": len(sequences),
        "n_converters": int(sum(labels.values())) if labels else 0,
        "alphabet": alphabet(sequences),
        "ngrams": {},
        "base_rate": np.nan,
    }
    for order in cfg_seq["orders"]:
        table, base = ngram_lift_table(sequences, labels, order,
                                       min_support=cfg_seq["min_support"],
                                       top_n=cfg_seq["top_n"])
        result["ngrams"][order] = table
        result["base_rate"] = base

    result["transitions"] = comparison_matrices(
        sequences, labels,
        order=cfg_seq["transition_order"],
        top_states=cfg_seq["transition_top_states"])

    result["length"] = pd.Series([len(s) for s in sequences.values()],
                                 dtype="float64")
    return result


# =====================================================================
# rendering
# =====================================================================

def render(result, cfg_seq, charts, html, vocab, heading_note=""):
    """
    Turn a profile() result into report HTML. Shared by both entry
    points so the section reads identically in each.
    """
    blocks = []
    event_column = cfg_seq["event_column"]

    if result["n_customers"] == 0:
        return html.card(html.callout(
            "warning", "No sequences",
            "No customer in this population had an eligible history."))

    blocks.append(html.callout(
        "neutral", "How to read this",
        "Each row is the run of <b>%s</b> values a customer's history "
        "ends on, immediately before the anchor. Lift is that run's "
        "%s rate divided by the %.2f%% base rate for this "
        "population. A run needs at least %d customers to appear, and "
        "the 95%% column is a Wilson lower bound &mdash; when it sits "
        "below 1.0 the pattern is not separable from chance at this "
        "support."
        % (html.esc(event_column), html.esc(vocab["event_name"]),
           100.0 * _rate(result["base_rate"]), cfg_seq["min_support"])))
    if heading_note:
        blocks.append(html.note(heading_note))

    for order in cfg_seq["orders"]:
        table = result["ngrams"].get(order)
        blocks.append(html.h3("Order %d — the last %d event%s before the anchor"
                              % (order, order, "" if order == 1 else "s")))
        if table is None or table.empty:
            blocks.append(charts.empty(
                "No run of %d events reached the %d-customer support "
                "threshold, so nothing here is rankable."
                % (order, cfg_seq["min_support"])))
            continue

        blocks.append(charts.diverging_barh(
            table["sequence"], table["lift"], center=1.0,
            xlabel="lift vs base rate", value_fmt="%.2fx",
            title="Order-%d runs ranked by lift" % order,
            alt="Terminal %d-event runs ranked by %s lift"
                % (order, vocab["event_name"])))
        # The column is named for the vocabulary so the header reads
        # correctly for whatever event this pipeline is pointed at.
        actor = vocab["actor_name"] + "s"
        display = table[["sequence", "customers", "share_of_population",
                         "converters", "rate", "lift", "lift_lo95",
                         "distinguishable"]].rename(
                             columns={"converters": actor})
        blocks.append(html.table(
            display,
            formats={"share_of_population": "pct1", "rate": "pct",
                     "lift": "%.2fx", "lift_lo95": "%.2fx",
                     "customers": "int", actor: "int"},
            wrap_columns=("sequence",),
            badges={"distinguishable": _support_badge},
            caption="Terminal %d-event runs, ranked by lift." % order))

    transitions = result.get("transitions")
    if transitions is not None and len(transitions["rows"]):
        blocks.append(html.h3("Where %ss and %ss diverge"
                              % (vocab["actor_name"], vocab["non_actor_name"])))
        blocks.append(charts.heatmap(
            transitions["difference"], transitions["rows"], transitions["cols"],
            mode="diverging", center=0.0, value_fmt="%+.2f",
            cbar_label="P(next) difference",
            title="%s minus %s transition probability"
                  % (vocab["actor_name"].capitalize(), vocab["non_actor_name"]),
            alt="Difference in next-event probability between %ss and %ss"
                % (vocab["actor_name"], vocab["non_actor_name"])))
        blocks.append(html.note(
            "Red means %ss make that move more often than "
            "%ss; blue means less often. Rows are the %d "
            "busiest from-states, columns the %d busiest next events. "
            "Built from %s of the former and %s of the latter."
            % (vocab["actor_name"], vocab["non_actor_name"],
               len(transitions["rows"]), len(transitions["cols"]),
               "{:,}".format(transitions["n_converters"]),
               "{:,}".format(transitions["n_others"]))))
        blocks.append(charts.heatmap(
            transitions["converter"], transitions["rows"], transitions["cols"],
            mode="sequential", value_fmt="%.2f",
            cbar_label="P(next | from)",
            title="%s transition matrix" % vocab["actor_name"].capitalize(),
            alt="Next-event probabilities among customers who %sed"
                % vocab["event_verb"]))

    return html.card(*blocks)


def _support_badge(value, _row):
    return ("good", "separable") if bool(value) else ("warning", "low support")
