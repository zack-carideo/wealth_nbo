"""
eda.py -- what is actually in the data, before any model exists.

    python eda.py                        # uses report_config.yaml
    python eda.py --config other.yaml
    python eda.py --no-run               # re-render from artifacts on disk

Writes outputs/reports/eda_report.html, a single self-contained file
aimed at a marketing audience: no jargon without a definition, every
chart backed by the table of numbers behind it.

WHAT IT COVERS
    1. optionally generate synthetic raw data and run preprocessing
    2. the raw acquisition file: volume, balances, data quality, dynamics
    3. n-order Markov profiling of the event runs that precede the
       target event
    4. the engineered dataset: raw -> engineered lineage, distributions,
       relationship to the target, exclusions

Steps 1-4 are all driven from report_config.yaml plus the two stage
configs. Nothing here hardcodes a column name, so pointing the pipeline
at a different dataset needs no edit to this file.

The Markov functions in nbo_report/sequences.py are the same ones
main.py calls on the high-propensity subset, which is what makes the
two reports comparable.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nbo_report import charts, config, html, profiling, sequences   # noqa: E402


# =====================================================================
# section 1: headline + raw data
# =====================================================================

def headline(df, acq, out, cfg_pre, vocab):
    id_col = cfg_pre["columns"]["id"]
    date_col = cfg_pre["columns"]["date"]
    n_customers = df[id_col].nunique()
    n_converters = out[id_col].nunique()
    rate = n_converters / float(n_customers) if n_customers else 0.0

    # An empty acquisition frame gives NaT here, and .date() on NaT
    # raises. Report the gap instead of dying on it.
    first, last = acq[date_col].min(), acq[date_col].max()
    span = ("%s to %s" % (first.date(), last.date())
            if pd.notna(first) and pd.notna(last)
            else "no acquisition rows")

    return html.kpi_row([
        ("Customers", "{:,}".format(n_customers), "in the raw extract"),
        ("Acquisitions", "{:,}".format(len(acq)),
         "%.1f per customer" % (len(acq) / float(n_customers) if n_customers else 0)),
        (vocab["actor_name"].capitalize() + "s", "{:,}".format(n_converters),
         "customers with an outcome row"),
        (vocab["event_name"].capitalize() + " rate", "%.2f%%" % (100.0 * rate),
         "of all customers"),
        ("History span",
         (span.split(" to ")[0][:4] + "–" + span.split(" to ")[1][:4])
         if " to " in span else "n/a", span),
    ]), rate, n_customers


def raw_overview(acq, cfg_pre, cfg_rep, schema):
    col = cfg_pre["columns"]
    family_col = schema["family"]
    blocks = [html.h3("Volume by %s" % family_col.replace("_", " "))]

    if family_col not in acq.columns:
        blocks.append(charts.empty(
            "schema.family names '%s', which is not a column on the "
            "acquisition rows, so the volume breakdown is skipped."
            % family_col))
    else:
        counts = acq[family_col].value_counts()
        top = counts.head(cfg_rep["profiling"]["max_categories_charted"])
        blocks.append(charts.barv(
            top.index, top.values, ylabel="acquisitions", value_fmt="%.0f",
            title="Acquisitions by %s" % family_col.replace("_", " "),
            alt="Count of acquisitions per %s" % family_col))

        family = (acq.groupby(family_col)
                     .agg(acquisitions=(col["date"], "size"),
                          customers=(col["id"], "nunique"))
                     .reset_index())
        family["share"] = family["acquisitions"] / float(len(acq))
        for name in cfg_pre["numeric_columns"]:
            family["mean_" + name] = (acq.groupby(family_col)[name]
                                         .mean().reindex(family[family_col]).values)
        blocks.append(html.table(
            family.sort_values("acquisitions", ascending=False),
            formats={"acquisitions": "int", "customers": "int",
                     "share": "pct1"},
            caption="Every %s in the file, with the mean of each numeric "
                    "column named in config.yaml. Blanks are structural: a "
                    "row has no value for a measure that does not apply to "
                    "its product." % family_col.replace("_", " ")))

    # ---- balance distributions --------------------------------------
    blocks.append(html.h3("Balance distributions"))
    blocks.append(html.note(
        "Long right tails are why preprocessing ranks these within "
        "product type and vintage rather than using the raw amount. "
        "Plotted on a log scale for that reason."))
    charted = 0
    for name in cfg_pre["numeric_columns"]:
        if charted >= cfg_rep["profiling"]["balance_columns_charted"]:
            break
        values = acq[name].dropna()
        if values.empty:
            continue
        blocks.append(charts.histogram(
            values, xlabel=name, log_x=True,
            title="%s, where present (n=%s)" % (name, "{:,}".format(len(values))),
            alt="Distribution of %s across acquisition rows" % name))
        charted += 1

    # ---- data quality -----------------------------------------------
    blocks.append(html.h3("Data quality"))
    quality = []
    for name in acq.columns:
        series = acq[name]
        quality.append({
            "column": name,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing_pct": float(series.isna().mean()),
            "distinct": int(series.nunique(dropna=True)),
        })
    blocks.append(html.table(
        pd.DataFrame(quality).sort_values("missing_pct", ascending=False),
        formats={"non_null": "int", "distinct": "int", "missing_pct": "pct1"},
        caption="Acquisition rows only; the outcome rows were split away "
                "first and are null in every column but id, date and target."))
    return html.card(*blocks)


def acquisition_dynamics(acq, cfg_pre):
    col = cfg_pre["columns"]
    if acq.empty:
        return html.card(charts.empty(
            "There are no acquisition rows, so there is nothing to "
            "describe here."))
    per_customer = acq.groupby(col["id"]).size()

    ordered = acq.sort_values([col["id"], col["date"]], kind="mergesort")
    delta = ordered.groupby(col["id"])[col["date"]].diff().dt.days / 30.44
    gaps = delta.dropna()

    blocks = [
        html.h3("How many products, and how far apart"),
        charts.histogram(per_customer.values,
                         bins=int(per_customer.max())
                         if pd.notna(per_customer.max()) else 10,
                         xlabel="acquisitions per customer",
                         title="Acquisitions per customer",
                         alt="Distribution of acquisition counts per customer"),
        charts.histogram(gaps.values, bins=48,
                         xlabel="months between consecutive acquisitions",
                         title="Gap between consecutive acquisitions",
                         alt="Distribution of months between one acquisition "
                             "and the next"),
    ]

    summary = pd.DataFrame([
        {"measure": "acquisitions per customer",
         "p10": per_customer.quantile(0.10), "median": per_customer.median(),
         "mean": per_customer.mean(), "p90": per_customer.quantile(0.90),
         "max": per_customer.max()},
        {"measure": "months between acquisitions",
         "p10": gaps.quantile(0.10), "median": gaps.median(),
         "mean": gaps.mean(), "p90": gaps.quantile(0.90), "max": gaps.max()},
    ])
    blocks.append(html.table(summary, formats={c: "%.1f" for c in
                                               ["p10", "median", "mean", "p90", "max"]},
                             caption="Gaps are measured between acquisitions "
                                     "only. The distance from the last "
                                     "acquisition to the outcome is the label "
                                     "and never appears as a feature."))
    zero = float((gaps == 0).mean()) if len(gaps) else 0.0
    blocks.append(html.note(
        "%.1f%% of consecutive pairs are same-month, i.e. bundled openings. "
        "preprocess.py counts these as bundle_count." % (100.0 * zero)))
    return html.card(*blocks)


# =====================================================================
# section 2: markov
# =====================================================================

def markov_section(ctx, acq, out, pre):
    cfg_seq = ctx.cfg["sequences"]
    anchors = pre.compute_anchors(acq, out, ctx.pre, mode=cfg_seq["anchor_mode"])
    seqs, labels = sequences.build_sequences(acq, anchors, ctx.pre,
                                             cfg_seq["event_column"],
                                             ctx.schema["anchors"])
    result = sequences.profile(seqs, labels, cfg_seq)
    body = sequences.render(
        result, cfg_seq, charts, html, ctx.labels,
        heading_note="Population: every customer eligible under the '%s' "
                     "anchor rule (%s customers, %s %ss). Histories are cut "
                     "at the anchor, so nothing at or after the outcome is "
                     "visible here."
                     % (cfg_seq["anchor_mode"],
                        "{:,}".format(result["n_customers"]),
                        "{:,}".format(result["n_converters"]),
                        ctx.labels["actor_name"]))
    return body, result, anchors, seqs, labels


# =====================================================================
# section 3: engineered features
# =====================================================================

def engineered_section(ctx, dataset, design, label_col):
    cfg_prof = ctx.cfg["profiling"]
    blocks = []

    # ---- lineage ----------------------------------------------------
    lineage = profiling.lineage_map(ctx.pre, list(dataset.columns), design,
                                    ctx.schema)
    present = lineage[lineage["in_dataset"]]
    missing = lineage[~lineage["in_dataset"]]
    unmapped = lineage[lineage["block"] == "unmapped"]

    blocks.append(html.h3("Raw → engineered lineage"))
    blocks.append(html.note(
        "Rebuilt from config.yaml, not hand-maintained: preprocess.py "
        "names every column from the config, so this table is the config "
        "read back. %d of %d expected columns are present."
        % (len(present), len(lineage) - len(unmapped))))
    if len(unmapped):
        blocks.append(html.callout(
            "warning", "%d column(s) not explained by the config" % len(unmapped),
            "These are in the dataset but nothing in config.yaml implies "
            "them: <span class='mono'>%s</span>. That usually means "
            "preprocess.py grew a feature the config does not describe."
            % html.esc(", ".join(unmapped["feature"].head(12)))))
    if len(missing):
        blocks.append(html.callout(
            "warning", "%d configured column(s) absent from the file" % len(missing),
            "<span class='mono'>%s</span>"
            % html.esc(", ".join(missing["feature"].head(12)))))
    blocks.append(html.table(
        lineage[["feature", "block", "source", "transform"]],
        wrap_columns=("transform",),
        max_rows=ctx.cfg["report"]["max_table_rows"],
        caption="Every engineered column, the raw column it came from, "
                "and what was done to it."))

    # ---- distributions ----------------------------------------------
    # Keys, labels and bookkeeping come from schema, so nothing here is
    # a literal column name.
    numeric, categorical = profiling.split_feature_types(
        dataset, ctx.bookkeeping_columns(design))

    blocks.append(html.h3("Distributions and missingness"))
    blocks.append(html.note(profiling.structural_missing_note(ctx.pre)))
    blocks.append(html.table(
        profiling.numeric_summary(dataset, numeric),
        formats={"non_null": "int", "distinct": "int", "missing_pct": "pct1",
                 "mean": "%.2f", "std": "%.2f", "p10": "%.2f",
                 "median": "%.2f", "p90": "%.2f", "max": "%.2f"},
        max_rows=ctx.cfg["report"]["max_table_rows"],
        caption="Numeric engineered features, most-missing first."))
    if categorical:
        blocks.append(html.table(
            profiling.categorical_summary(dataset, categorical),
            formats={"non_null": "int", "levels": "int", "missing_pct": "pct1"},
            wrap_columns=("most_common",),
            max_rows=ctx.cfg["report"]["max_table_rows"],
            caption="Categorical engineered features."))

    # ---- relationship to the target ---------------------------------
    blocks.append(html.h3("Relationship to the target"))
    ranking = profiling.iv_ranking(dataset, numeric + categorical, label_col,
                                   bins=cfg_prof["iv_bins"],
                                   min_bin_count=cfg_prof["min_bin_count"])
    blocks.append(html.note(
        "Information Value summarises how far a feature separates %ss "
        "from everyone else, and reads the same for numeric and "
        "categorical columns. Below 0.02 is no signal; above 0.5 is "
        "usually a leak rather than a discovery, and is flagged."
        % ctx.labels["actor_name"]))
    top = ranking.head(cfg_prof["top_iv_features"])
    blocks.append(charts.barh(
        top["feature"], top["iv"].fillna(0.0), xlabel="Information Value",
        value_fmt="%.3f", title="Features ranked by Information Value",
        alt="Engineered features ranked by information value"))
    blocks.append(html.table(
        top, formats={"iv": "%.4f", "missing_pct": "pct1"},
        badges={"strength": _iv_badge},
        caption="Top %d of %d features by IV." % (len(top), len(ranking))))

    # ---- is that separation the customer, or the anchor rule? --------
    asymmetry = _anchor_asymmetry(ctx, dataset, design, numeric)
    if asymmetry is not None and len(asymmetry):
        blocks.append(html.h3("How much of that separation is the anchor rule?"))
        blocks.append(html.note(
            "The fixed design anchors everyone who did not %s at their "
            "last acquisition at least %d months before the extract date, "
            "so both classes get equal time at risk. That also discards "
            "any acquisition they made inside the horizon, while a %s "
            "keeps their history right up to the outcome. History-length "
            "features therefore look stronger under the fixed rule than "
            "under hazard, which applies no cutoff. Separation is Cohen's "
            "d; shrinkage is how much of it disappears once the cutoff is "
            "removed."
            % (ctx.labels["event_verb"],
               ctx.pre["labeling"]["horizon_months"],
               ctx.labels["actor_name"])))
        actor = ctx.labels["actor_name"]
        display = asymmetry.rename(columns={
            "mean_converter_fixed": "mean %s (fixed)" % actor,
            "mean_other_fixed": "mean other (fixed)",
            "mean_other_hazard": "mean other (hazard)"})
        blocks.append(html.table(
            display,
            formats={"mean %s (fixed)" % actor: "%.2f",
                     "mean other (fixed)": "%.2f",
                     "mean other (hazard)": "%.2f",
                     "separation_fixed": "%.2f",
                     "separation_hazard": "%.2f", "shrinkage": "%.2f"},
            caption="Features whose class separation shrinks most when the "
                    "fixed-horizon cutoff is removed. This is a property of "
                    "the documented anchor rule, not a defect — but it is "
                    "why a very high Information Value on a count or tenure "
                    "feature should not be read as pure customer behaviour."))

    # Event rate by decile for the strongest few.
    blocks.append(html.h3("Event rate by decile, strongest features"))
    charted = 0
    for name in top["feature"]:
        if charted >= cfg_prof["max_features_charted"]:
            break
        table, base = profiling.event_rate_by_bin(
            dataset, name, label_col, bins=cfg_prof["decile_bins"],
            min_bin_count=cfg_prof["min_bin_count"])
        if len(table) < 2:
            continue
        blocks.append(charts.diverging_barh(
            table["bin"], table["lift"], center=1.0,
            xlabel="lift vs the %.2f%% overall rate" % (100.0 * base),
            value_fmt="%.2fx", title=name,
            alt="Event rate lift by bin for %s" % name))
        blocks.append(html.table(
            table[["bin", "customers", "share", "events", "event_rate", "lift"]],
            formats={"customers": "int", "events": "int", "share": "pct1",
                     "event_rate": "pct", "lift": "%.2fx"},
            wrap_columns=("bin",)))
        charted += 1

    return html.card(*blocks), ranking, numeric, categorical, asymmetry


def _anchor_asymmetry(ctx, dataset, design, numeric):
    """
    Only computable when both designs are on disk, and only meaningful
    when we are profiling the fixed/flat one.
    """
    if design != "flat":
        return None
    hazard_path = ctx.dataset_path("hazard")
    if not os.path.exists(hazard_path):
        return None
    try:
        hazard = pd.read_csv(hazard_path)
    except Exception:                                  # noqa: BLE001
        return None
    return profiling.anchor_asymmetry(dataset, hazard, numeric, ctx.schema)


def _iv_badge(value, _row):
    text = str(value)
    if text in ("no signal", "not computable"):
        return "neutral", text
    if text == "weak":
        return "neutral", text
    if text == "medium":
        return "good", text
    if text == "strong":
        return "good", text
    if text.startswith("failed"):
        return "warning", text
    return "critical", text


# =====================================================================
# section 4: exclusions
# =====================================================================

def exclusions_section(ctx, n_customers):
    path = os.path.join(ctx.pre["io"]["output_dir"], "exclusions.csv")
    if not os.path.exists(path):
        return html.card(charts.empty(
            "exclusions.csv is not on disk yet. Run preprocessing first.")), None

    excl = pd.read_csv(path)
    ex = ctx.schema["exclusions"]
    design_col, reason_col = ex["design"], ex["reason"]
    if excl.empty:
        return html.card(html.callout(
            "good", "Nobody was dropped",
            "Every customer in the raw file was eligible under both "
            "designs.")), excl

    summary = (excl.groupby([design_col, reason_col]).size()
                   .reset_index(name="customers"))
    summary["share_of_population"] = summary["customers"] / float(n_customers)

    blocks = [
        html.note("Exclusions are recorded per design because the two "
                  "anchor rules disagree about who is eligible. The fixed "
                  "design drops %ss whose outcome falls beyond the "
                  "%d-month horizon rather than relabelling them 0."
                  % (ctx.labels["actor_name"],
                     ctx.pre["labeling"]["horizon_months"])),
        charts.grouped_barh(
            sorted(summary[reason_col].unique()),
            {design: [float(summary[(summary[design_col] == design) &
                                    (summary[reason_col] == reason)]
                            ["customers"].sum())
                      for reason in sorted(summary[reason_col].unique())]
             for design in sorted(summary[design_col].unique())},
            xlabel="customers dropped", value_fmt="%.0f",
            title="Customers dropped, by reason and design",
            alt="Excluded customers by reason for each modelling design"),
        html.table(summary.sort_values([design_col, "customers"],
                                       ascending=[True, False]),
                   formats={"customers": "int", "share_of_population": "pct1"},
                   wrap_columns=(reason_col,),
                   caption="Share is of all customers in the raw extract."),
    ]
    return html.card(*blocks), excl


# =====================================================================
# insights, derived from the numbers rather than written by hand
# =====================================================================

def synthetic_warning(ctx):
    """
    make_synthetic.py injects its signal deliberately, so metrics from
    it say nothing about achievable lift. If the config is pointed at
    generated data, the report says so before anything else.
    """
    if not ctx.cfg["steps"]["generate_synthetic"]:
        return None
    return ("critical", "This run is on synthetic data",
            "<code>steps.generate_synthetic</code> is on, so the input is "
            "<code>make_synthetic.py</code> output. Signal is injected "
            "deliberately in that generator — larger deposits and "
            "savings/time products are wired to raise conversion odds — "
            "so every rate, lift and Information Value below is a "
            "property of the generator, not of any real population. Use "
            "this run to confirm the pipeline works. Do not quote a "
            "number from it.")


def build_insights(ctx, rate, n_customers, seq_result, ranking, excl, dataset,
                   label_col, asymmetry):
    cfg_rep = ctx.cfg["report"]
    vocab = ctx.labels
    items = []
    banner = synthetic_warning(ctx)
    if banner:
        items.append(banner)

    # -- prevalence
    if rate < cfg_rep["low_prevalence_threshold"]:
        items.append((
            "warning", "Rare event: %.2f%% %s"
            % (100.0 * rate, vocab["event_name"]),
            "Below %.1f%% prevalence, ROC-AUC carries almost no "
            "information. Judge the model on PR-AUC and lift at the top "
            "of the ranking instead. The event count, not the row count, "
            "is what limits how many features this data supports."
            % (100.0 * cfg_rep["low_prevalence_threshold"])))
    else:
        items.append((
            "neutral", "%s rate is %.2f%%"
            % (vocab["event_name"].capitalize(), 100.0 * rate),
            "High enough that PR-AUC is stable, but lift at the top of "
            "the ranking is still the number a campaign is planned "
            "against."))

    # -- strongest sequence
    best = None
    for order in sorted(seq_result["ngrams"], reverse=True):
        table = seq_result["ngrams"][order]
        if table is None or table.empty:
            continue
        solid = table[table["distinguishable"]]
        if len(solid):
            best = (order, solid.iloc[0])
            break
    if best is not None:
        order, row = best
        items.append((
            "good", "Strongest sequence: %s" % row["sequence"],
            "Customers whose last %d acquisitions run <b>%s</b> %s at "
            "%.2f%%, <b>%.2fx</b> the base rate, across %s customers "
            "(%.1f%% of the population). The 95%% lower bound is still "
            "%.2fx, so this is separable from chance."
            % (order, html.esc(row["sequence"]), vocab["event_verb"],
               100.0 * row["rate"], row["lift"],
               "{:,}".format(int(row["customers"])),
               100.0 * row["share_of_population"], row["lift_lo95"])))
    else:
        items.append((
            "warning", "No sequence clears the support threshold",
            "No run of events reaches %d customers with a lift whose 95%% "
            "lower bound clears 1.0. Either the alphabet is too fine "
            "(try product_family instead of product_type) or the event is "
            "too rare to profile by path."
            % ctx.cfg["sequences"]["min_support"]))

    # -- feature signal
    usable = ranking[ranking["iv"].notna()]
    if len(usable):
        strongest = usable.iloc[0]
        leaky = usable[usable["strength"] == "suspiciously strong"]
        if len(leaky):
            detail = ""
            if asymmetry is not None and len(asymmetry):
                worst = asymmetry.iloc[0]
                detail = (" The anchor-rule check found the largest effect on "
                          "<span class='mono'>%s</span>: %ss average "
                          "%.2f against %.2f for everyone else under the "
                          "fixed rule, but that second group recovers to "
                          "%.2f once the horizon cutoff is removed, so "
                          "%.0f%% of the gap is the cutoff rather than "
                          "behaviour."
                          % (html.esc(worst["feature"]), vocab["actor_name"],
                             worst["mean_converter_fixed"],
                             worst["mean_other_fixed"],
                             worst["mean_other_hazard"],
                             100.0 * max(0.0, worst["shrinkage"]) /
                             (abs(worst["separation_fixed"]) or 1.0)))
            items.append((
                "critical", "%d feature(s) carry implausible signal" % len(leaky),
                "<span class='mono'>%s</span> exceed IV 0.5 on a %.2f%% "
                "event. Three things produce that, and they need telling "
                "apart: genuine leakage, a generator that injected the "
                "signal, or the fixed-horizon anchor truncating the "
                "histories of everyone else.%s"
                % (html.esc(", ".join(leaky["feature"].head(6))),
                   100.0 * rate, detail)))
        n_signal = int((usable["iv"] >= 0.02).sum())
        items.append((
            "neutral", "%d of %d features carry any signal" % (n_signal, len(usable)),
            "Strongest is <span class='mono'>%s</span> at IV %.3f (%s). "
            "Features below IV 0.02 are not separating the classes at "
            "all; they cost width in the model without earning it."
            % (html.esc(strongest["feature"]), strongest["iv"],
               strongest["strength"])))

    # -- events available
    events = int(dataset[label_col].sum())
    items.append((
        "neutral" if events >= 200 else "warning",
        "%s events available to fit against" % "{:,}".format(events),
        "Rule of thumb is roughly ten events per feature considered. At "
        "%s events, that supports on the order of %d features; the "
        "dataset currently offers %d columns."
        % ("{:,}".format(events), max(1, events // 10),
           dataset.shape[1] - 1)))

    # -- exclusions
    if excl is not None and len(excl):
        ex = ctx.schema["exclusions"]
        worst = (excl.groupby(ex["design"]).size() / float(n_customers)).max()
        top_reason = excl[ex["reason"]].value_counts().index[0]
        items.append((
            "warning" if worst > 0.25 else "neutral",
            "Up to %.1f%% of customers are excluded" % (100.0 * worst),
            "Most common reason: <b>%s</b>. Exclusions are a modelling "
            "decision, not a bug &mdash; but the scored population is "
            "the eligible one, so a campaign built on this model reaches "
            "the survivors only." % html.esc(str(top_reason))))

    return items


# =====================================================================
# main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="report_config.yaml")
    parser.add_argument("--no-run", action="store_true",
                        help="skip every pipeline step and re-render from "
                             "whatever is already on disk")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")     # sequences contain arrows
    except (AttributeError, ValueError):
        pass

    ctx = config.ReportContext(args.config)
    steps = ctx.cfg["steps"]
    if not args.no_run:
        if steps["generate_synthetic"]:
            config.run_synthetic(ctx)
        if steps["run_preprocess"]:
            config.run_preprocess(ctx)

    pre = config.preprocess_module(ctx)

    print("\nreading %s" % ctx.raw_input_path())
    raw = pre.load_data(ctx.pre)
    acq, out = pre.split_outcome_rows(raw, ctx.pre)

    design = ctx.cfg["profiling"]["dataset"]
    label_col = ctx.label_column(design)
    dataset_path = ctx.dataset_path(design)
    if not os.path.exists(dataset_path):
        raise SystemExit(
            "%s does not exist. Run with steps.run_preprocess: true, or "
            "drop --no-run." % dataset_path)
    dataset = pd.read_csv(dataset_path)

    print("building sections...")
    kpis, rate, n_customers = headline(raw, acq, out, ctx.pre, ctx.labels)
    raw_block = raw_overview(acq, ctx.pre, ctx.cfg, ctx.schema)
    dynamics = acquisition_dynamics(acq, ctx.pre)
    markov, seq_result, _anchors, _seqs, _labels = markov_section(
        ctx, acq, out, pre)
    engineered, ranking, _num, _cat, asymmetry = engineered_section(
        ctx, dataset, design, label_col)
    exclusions, excl = exclusions_section(ctx, n_customers)
    insight_items = build_insights(ctx, rate, n_customers, seq_result,
                                   ranking, excl, dataset, label_col,
                                   asymmetry)

    toc = [("insights", "Insights"), ("raw", "Raw data"),
           ("dynamics", "Acquisition dynamics"), ("markov", "Event sequences"),
           ("engineered", "Engineered features"), ("exclusions", "Exclusions")]

    body = [
        kpis,
        html.section("insights", "What the data says",
                     "Generated from this run's numbers, not written by "
                     "hand — every statement below re-derives when the "
                     "pipeline is pointed at a different dataset.",
                     html.insights(insight_items)),
        html.section("raw", "The raw acquisition file",
                     "One row per product a customer opened, plus one "
                     "outcome row per %s. Outcome rows are split away "
                     "before anything is aggregated."
                     % ctx.labels["actor_name"],
                     raw_block),
        html.section("dynamics", "Acquisition dynamics",
                     "How many products customers hold and how far apart "
                     "they buy them — the raw material the sequence "
                     "features are built from.",
                     dynamics),
        html.section("markov", "Event sequences leading to the target",
                     "Which runs of events precede %s more often than "
                     "chance. Generic over any event alphabet: the column "
                     "being profiled is set in report_config.yaml."
                     % ctx.labels["event_name"],
                     markov),
        html.section("engineered", "Engineered features",
                     "The %s dataset preprocess.py wrote: where each "
                     "column came from, how it is distributed, and how it "
                     "relates to the target." % design,
                     engineered),
        html.section("exclusions", "Who was excluded, and why",
                     "Eligibility is decided by the anchor rule, and the "
                     "two designs disagree. This is who did not make it "
                     "into each dataset.",
                     exclusions),
    ]

    meta = ("%s · %s customers · %s design profiled · event column '%s'"
            % (os.path.basename(ctx.raw_input_path()),
               "{:,}".format(n_customers), design,
               ctx.cfg["sequences"]["event_column"]))
    document = html.page(ctx.labels["eda_title"], ctx.labels["eda_subtitle"],
                         meta, body, toc)

    path = html.write(ctx.report_path("eda_report"), document)
    print("\nwrote %s (%.1f KB)" % (path, os.path.getsize(path) / 1024.0))


if __name__ == "__main__":
    main()
