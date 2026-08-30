"""
main.py -- the full chain, ending in a call list.

    python main.py                       # uses report_config.yaml
    python main.py --no-run              # re-render from artifacts on disk

    1. generate synthetic raw data        (config-gated)
    2. preprocess into modelling datasets (config-gated)
    3. fit and evaluate                   (config-gated)
    4. subset to the high-propensity customers and profile the event
       sequences leading to conversion, using the SAME functions eda.py
       uses on the whole population

Writes:
    outputs/reports/model_report.html     model insights + target list
    outputs/reports/next_best_customers.csv   for campaign upload
    outputs/reports/segment_profiles.csv
    outputs/model_output/model.joblib     the refit deliverable model

TWO SCORES, KEPT APART
    The campaign list is RANKED on out-of-fold scores -- each customer
    scored by a model that never trained on them. The persisted
    model.joblib is refit on everything and is for scoring customers
    who were not in this extract. Ranking today's population with the
    refit model would look better and mean less; see nbo_report/
    targeting.py for the longer version.
"""

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nbo_report import charts, config, html, sequences, targeting   # noqa: E402
import eda                                                          # noqa: E402


# =====================================================================
# model.py's OUTPUT CONTRACT
#
# Dataset column names never appear in this file -- they come from
# report_config.yaml's `schema:` block or from the two stage configs.
# The names below are different: they are the columns model.py writes
# into its own artifact files, and model.py hardcodes them itself.
#
# They are listed here so the coupling is discoverable in one place
# rather than only as scattered string literals. They are deliberately
# NOT routed through report_config.yaml: a config entry could silently
# disagree with what model.py actually writes, which is worse than a
# literal that fails loudly on a KeyError.
#
#   predictions.csv   score, actual, repeat, <columns.id>
#   metrics.csv       metric, mean, std, min, max
#   metrics_by_repeat.csv
#                     repeat, pr_auc, roc_auc, brier, lift_top_<pct>%
#   calibration.csv   bin, n, mean_predicted, observed_rate
#   features.csv      feature, and either
#                       mean_coef, std_coef, sign_consistency, nonzero_rate
#                     (logistic) or
#                       mean_importance, std_importance                (gbm)
#   in-memory frame   _sample_weight, added by model.load_data
#
# If model.py is ever forked to rename these, grep for them here.
# =====================================================================


# =====================================================================
# run summary
# =====================================================================

def run_summary(ctx, df, cfg_mdl, scores, bundle):
    col = cfg_mdl["columns"]
    fraction = cfg_mdl.get("sampling", {}).get("negative_customer_fraction", 1.0)
    prevalence = scores["actual"].mean() if len(scores) else np.nan

    kpis = html.kpi_row([
        ("Design", cfg_mdl["design"], "anchor and label shape"),
        ("Model", cfg_mdl["model"]["type"], "%s CV folds x %s repeats"
         % (cfg_mdl["cv"]["n_splits"], cfg_mdl["cv"]["n_repeats"])),
        ("Scored customers", "{:,}".format(len(scores)),
         "evaluated out-of-fold"),
        ("Events", "{:,}".format(int(scores["actual"].sum())),
         "%.2f%% of the scored population" % (100.0 * prevalence)),
        ("Rows fitted", "{:,}".format(len(df)),
         "%s customers after sampling" % "{:,}".format(df[col["group"]].nunique())),
    ])

    settings = pd.DataFrame([
        ("input dataset", os.path.basename(cfg_mdl["io"]["input_path"])),
        ("design", cfg_mdl["design"]),
        ("model type", cfg_mdl["model"]["type"]),
        ("label column", col["label"]),
        ("row weight", col["weight"] or "(none)"),
        ("negative_customer_fraction", "%.2f" % fraction),
        ("CV", "StratifiedGroupKFold on '%s', %d splits x %d repeats"
         % (col["group"], cfg_mdl["cv"]["n_splits"], cfg_mdl["cv"]["n_repeats"])),
        ("features", "%d numeric, %d categorical"
         % (len(bundle["numeric"]), len(bundle["categorical"]))),
        ("hazard horizon", "%d months in %d-month bins"
         % (cfg_mdl["evaluation"]["hazard_horizon_months"],
            cfg_mdl["evaluation"]["hazard_bin_months"])
         if cfg_mdl["design"] == "hazard" else "n/a (flat design)"),
        ("scikit-learn", bundle["sklearn_version"]),
    ], columns=["setting", "value"])

    return kpis, html.card(
        html.h3("The configuration this run actually used"),
        html.table(settings, wrap_columns=("value",),
                   caption="Read back from model_config.yaml at run time, "
                           "so the report cannot drift from the run."))


# =====================================================================
# performance
# =====================================================================

def performance_section(ctx, metrics, per_repeat, calibration, scores):
    cfg_rep = ctx.cfg["report"]
    lookup = dict(zip(metrics["metric"], metrics["mean"]))
    prevalence = lookup.get("prevalence", np.nan)
    lift_columns = [c for c in per_repeat.columns if c.startswith("lift_top_")]

    blocks = []

    # ---- the numbers that matter, first ----------------------------
    cards = [
        ("PR-AUC", "%.4f" % lookup.get("pr_auc", np.nan),
         "vs %.4f at random" % prevalence),
    ]
    for name in lift_columns:
        cards.append((name.replace("lift_top_", "Lift @ top "),
                      "%.2fx" % lookup.get(name, np.nan),
                      "of the ranked population"))
    cards.append(("Brier", "%.4f" % lookup.get("brier", np.nan),
                  "lower is better calibrated"))
    cards.append(("ROC-AUC", "%.4f" % lookup.get("roc_auc", np.nan),
                  "reference only" if prevalence < cfg_rep["low_prevalence_threshold"]
                  else "supporting metric"))
    blocks.append(html.kpi_row(cards))

    if prevalence < cfg_rep["low_prevalence_threshold"]:
        blocks.append(html.callout(
            "warning", "Read PR-AUC and lift, not ROC-AUC",
            "At %.2f%% prevalence ROC-AUC is close to uninformative — it "
            "is dominated by the enormous negative class and will look "
            "respectable even for a model with no useful ranking. It is "
            "printed for reference only. PR-AUC against a %.4f random "
            "baseline, and lift at the top of the ranking, are the "
            "numbers to judge this on."
            % (100.0 * prevalence, prevalence)))

    blocks.append(html.h3("Full metric table"))
    blocks.append(html.table(
        metrics, formats={"mean": "%.4f", "std": "%.4f",
                          "min": "%.4f", "max": "%.4f"},
        caption="Mean and spread across %d repeats of %d-fold CV."
                % (ctx.mdl["cv"]["n_repeats"], ctx.mdl["cv"]["n_splits"])))

    # ---- stability across repeats -----------------------------------
    interesting = ["pr_auc", "roc_auc", "brier"] + lift_columns
    interesting = [c for c in interesting if c in per_repeat.columns]
    if len(per_repeat) > 1 and interesting:
        blocks.append(html.h3("Spread across repeats"))
        blocks.append(charts.spread(
            interesting,
            [per_repeat[c].mean() for c in interesting],
            [per_repeat[c].min() for c in interesting],
            [per_repeat[c].max() for c in interesting],
            ylabel="value", title="Metric stability across repeats",
            alt="Mean and range of each metric across CV repeats"))
        blocks.append(html.note(
            "A wide range here means the metric is being driven by which "
            "customers landed in which fold, and a single point estimate "
            "from one run would be misleading."))

    # ---- calibration -------------------------------------------------
    if calibration is not None and len(calibration) > 1:
        blocks.append(html.h3("Calibration"))
        blocks.append(charts.line(
            calibration["mean_predicted"].tolist(),
            {"observed": calibration["observed_rate"].tolist()},
            xlabel="mean predicted probability",
            ylabel="observed rate",
            reference=([0, float(calibration["mean_predicted"].max())],
                       [0, float(calibration["mean_predicted"].max())],
                       "perfect calibration"),
            title="Predicted vs observed, by score bin",
            alt="Calibration curve: predicted probability against "
                "observed %s rate" % ctx.labels["event_name"]))
        blocks.append(html.table(
            calibration, formats={"n": "int", "mean_predicted": "%.4f",
                                  "observed_rate": "%.4f"},
            caption="Points above the reference line mean the model is "
                    "under-predicting in that band; below means it is "
                    "over-predicting."))
    return html.card(*blocks), lift_columns


# =====================================================================
# features
# =====================================================================

def _readable(raw_name):
    """
    Strip the ColumnTransformer prefixes off a feature name for display.

    features.csv carries model.py's raw `get_feature_names_out()` names
    (num__x, cat__x_LEVEL, num__missingindicator_x). The table keeps
    them verbatim so it still matches the file on disk; the chart, which
    has far less room, shows the readable form.
    """
    name = str(raw_name)
    for prefix in ("num__", "cat__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.startswith("missingindicator_"):
        return name[len("missingindicator_"):] + " (missing)"
    return name


def feature_section(ctx, features, converged_flag, n_iterations):
    floor = ctx.cfg["report"]["sign_consistency_floor"]
    is_logistic = "mean_coef" in features.columns
    blocks = []

    if converged_flag is False:
        blocks.append(html.callout(
            "critical", "The final fit did not converge",
            "saga stopped at the %s iteration cap. Coefficients from a "
            "non-converged fit must not be reported as findings. Lower "
            "<code>logistic.max_iter</code>'s workload by reducing "
            "<code>sampling.negative_customer_fraction</code>, or raise "
            "the cap, and refit before quoting anything below."
            % "{:,}".format(n_iterations or 0)))

    if is_logistic:
        unproven = features[features["sign_consistency"] < floor]
        proven = features[features["sign_consistency"] >= floor]
        blocks.append(html.note(
            "Sign consistency is how often a coefficient kept the same "
            "sign across folds. Below %.0f%% the feature is noise, not a "
            "finding, and is marked unproven — %d of %d features fall "
            "there." % (100.0 * floor, len(unproven), len(features))))

        top = proven.head(ctx.cfg["profiling"]["top_iv_features"])
        if len(top):
            blocks.append(charts.diverging_barh(
                [_readable(f) for f in top["feature"]],
                top["mean_coef"], center=0.0,
                xlabel="mean coefficient (log-odds, standardised inputs)",
                value_fmt="%+.3f",
                title="Strongest stable coefficients",
                alt="Model coefficients that kept a consistent sign across "
                    "folds"))
            blocks.append(html.note(
                "Red raises the odds of %s, blue lowers them. Inputs are "
                "standardised, so a coefficient is the log-odds change "
                "per standard deviation of that feature."
                % ctx.labels["event_name"]))
        blocks.append(html.table(
            features, formats={"mean_coef": "%+.4f", "std_coef": "%.4f",
                               "sign_consistency": "pct1",
                               "nonzero_rate": "pct1"},
            badges={"sign_consistency": lambda v, r: (
                ("good", "%.0f%% stable" % (100.0 * v)) if v >= floor
                else ("warning", "%.0f%% unproven" % (100.0 * v)))},
            max_rows=ctx.cfg["report"]["max_table_rows"],
            caption="All model features, strongest coefficient first."))
    else:
        top = features.head(ctx.cfg["profiling"]["top_iv_features"])
        blocks.append(charts.barh(
            top["feature"], top["mean_importance"],
            xlabel="permutation importance (PR-AUC drop)", value_fmt="%.4f",
            title="Permutation importance",
            alt="Features ranked by permutation importance"))
        blocks.append(html.note(
            "A tree model has no signed coefficient, so there is no "
            "sign-consistency check available here. Importance says a "
            "feature matters, not which direction it pushes."))
        blocks.append(html.table(
            features, formats={"mean_importance": "%.5f",
                               "std_importance": "%.5f"},
            max_rows=ctx.cfg["report"]["max_table_rows"]))

    return html.card(*blocks)


# =====================================================================
# score distribution and deciles
# =====================================================================

def score_section(ctx, scores, deciles):
    blocks = [
        html.h3("Where the scores land"),
        charts.histogram_by_class(
            scores["score"].values, scores["actual"].values,
            [(0, "did not %s" % ctx.labels["event_verb"]),
             (1, "%sed" % ctx.labels["event_verb"])], bins=40,
            xlabel="out-of-fold predicted probability",
            title="Score distribution by outcome",
            alt="Predicted probability distribution, %ss against %ss"
                % (ctx.labels["actor_name"], ctx.labels["non_actor_name"])),
        html.note("Separation between the two curves is the model's whole "
                  "value. Overlap is unavoidable; what matters is that the "
                  "%s mass sits to the right." % ctx.labels["actor_name"]),
    ]

    if len(deciles):
        blocks.append(html.h3("Decile table"))
        blocks.append(charts.diverging_barh(
            ["decile %d" % d for d in deciles["decile"]],
            deciles["lift"], center=1.0, xlabel="lift vs base rate",
            value_fmt="%.2fx", title="Lift by score decile",
            alt="Lift in each decile of the score ranking"))
        blocks.append(html.table(
            deciles, formats={"customers": "int", "events": "int",
                              "mean_score": "%.4f", "observed_rate": "pct",
                              "lift": "%.2fx", "cumulative_capture": "pct1"},
            caption="Decile 1 is the highest-scoring tenth. Cumulative "
                    "capture is the share of all %ss reached by calling "
                    "down to that decile." % ctx.labels["actor_name"]))

        share = deciles["customers"].cumsum() / deciles["customers"].sum()
        blocks.append(html.h3("Cumulative capture"))
        blocks.append(charts.line(
            share.tolist(),
            {"model": deciles["cumulative_capture"].tolist()},
            xlabel="share of the population called",
            ylabel="share of %ss reached" % ctx.labels["actor_name"],
            reference=([0, 1], [0, 1], "calling at random"),
            percent=True, title="Gains curve",
            alt="Cumulative share of %ss reached against share of "
                "population contacted" % ctx.labels["actor_name"]))
        top_decile = deciles.iloc[0]
        blocks.append(html.note(
            "Calling the top decile reaches %.1f%% of all %ss for "
            "%.0f%% of the contact cost."
            % (100.0 * top_decile["cumulative_capture"],
               ctx.labels["actor_name"],
               100.0 * top_decile["customers"] / deciles["customers"].sum())))
    return html.card(*blocks)


# =====================================================================
# segments
# =====================================================================

def segment_section(ctx, result, high, scores_by_customer):
    if result is None:
        return html.card(charts.empty(
            "Not enough high-propensity customers, or no usable numeric "
            "feature with variation, to fit segments.")), None

    table = result["table"].copy()
    observed = []
    for segment_name in table["segment"]:
        members = high.index[result["assignment"].map(
            targeting.segment_names(result)) == segment_name]
        actual = scores_by_customer.loc[
            scores_by_customer.index.intersection(members), "actual"]
        observed.append(float(actual.mean()) if len(actual) else np.nan)
    table["observed_rate"] = observed

    blocks = [
        html.note(
            "K-means with k=%d over the %s highest-scoring customers, on "
            "the standardised engineered features. A segment is described "
            "by how far it sits from the high-propensity average, so "
            "\"high\" means high relative to other good prospects, not to "
            "the whole book. Features moving less than %.2f standard "
            "deviations are left out of the description."
            % (ctx.cfg["targeting"]["segments"]["k"],
               "{:,}".format(len(high)),
               ctx.cfg["targeting"]["segments"]["min_zscore"])),
        charts.barh(table["segment"], table["customers"],
                    xlabel="customers", value_fmt="%.0f",
                    title="Segment sizes",
                    alt="Number of high-propensity customers in each segment"),
        html.table(
            table[["segment", "customers", "share", "mean_score",
                   "observed_rate", "distinguishing_features"]],
            formats={"customers": "int", "share": "pct1",
                     "mean_score": "%.4f", "observed_rate": "pct"},
            wrap_columns=("segment", "distinguishing_features"),
            caption="Observed rate is the actual %s rate among that "
                    "segment's members, which is the check on whether the "
                    "segmentation found anything real."
                    % ctx.labels["event_name"]),
    ]
    return html.card(*blocks), table


# =====================================================================
# the call list
# =====================================================================

def target_section(ctx, target, reason_kind, csv_path, customer_col,
                   n_scored, n_eligible):
    blocks = [
        html.callout(
            "good", "%s customers written to %s"
            % ("{:,}".format(len(target)), os.path.basename(csv_path)),
            "Ranked by out-of-fold score, so no customer here was ranked "
            "by a model that trained on them. The file carries the score, "
            "decile, reason codes and segment for each customer."),
    ]
    if n_eligible > n_scored:
        blocks.append(html.callout(
            "warning", "The list can only reach %s of %s eligible customers"
            % ("{:,}".format(n_scored), "{:,}".format(n_eligible)),
            "A customer can only be ranked if they have an out-of-fold "
            "score, and only the customers cross-validation actually ran "
            "on have one. %s customers (%.0f%%) are therefore absent from "
            "this list entirely — not low-scoring, simply never scored. "
            "Negative downsampling is the usual cause; in the hazard "
            "design, customers censored before the evaluation horizon are "
            "dropped too. Re-run with "
            "<code>sampling.negative_customer_fraction: 1.0</code> before "
            "treating this as full coverage of the book."
            % ("{:,}".format(n_eligible - n_scored),
               100.0 * (n_eligible - n_scored) / float(n_eligible)))) 
    if reason_kind == "deviation":
        blocks.append(html.callout(
            "warning", "Reason codes are approximate for this model type",
            "A tree model has no exact per-customer decomposition, so the "
            "reasons below describe where the customer sits on the "
            "features the model leans on globally. Switch "
            "<code>model.type</code> to logistic for exact reason codes."))
    else:
        blocks.append(html.note(
            "Reason codes are an exact decomposition of the PERSISTED "
            "model's log-odds for this customer, measured against the "
            "average customer: each number is that feature's contribution, "
            "and with the intercept they sum to that model's log-odds. "
            "They do not sum to the score column, which is the "
            "out-of-fold score averaged across repeats — and in the "
            "hazard design is a survival product across intervals rather "
            "than a single log-odds. Use the reasons to explain the "
            "customer, and the score to rank them."))

    display = target.head(ctx.cfg["report"]["max_table_rows"])
    blocks.append(html.table(
        display[["rank", customer_col, "score", "decile", "segment",
                 "reason_codes"]],
        formats={"score": "%.4f", "decile": "int", "rank": "int"},
        wrap_columns=("reason_codes", "segment"),
        caption="Top %d of %d. The full list is in the CSV."
                % (len(display), len(target))))
    return html.card(*blocks)


# =====================================================================
# model risk notes
# =====================================================================

def risk_section(ctx, cfg_mdl, features, scores, converged_flag,
                 n_iterations, deciles, n_eligible, bundle):
    cfg_rep = ctx.cfg["report"]
    fraction = cfg_mdl.get("sampling", {}).get("negative_customer_fraction", 1.0)
    items = []

    banner = eda.synthetic_warning(ctx)
    if banner:
        items.append(banner)

    if bundle.get("downsampled_training_frame"):
        items.append((
            "warning", "The persisted model was fit on sampled negatives",
            "<code>model.joblib</code> was refit on the downsampled frame, "
            "so its predicted probabilities sit above the true base rate. "
            "Set <code>targeting.refit_on_full_population: true</code> to "
            "fit the deliverable on every eligible customer."))
    elif fraction < 1.0:
        items.append((
            "good", "The persisted model was refit on the full population",
            "Cross-validation ran on the %.0f%% negative sample, but "
            "<code>model.joblib</code> was refit on all %s eligible "
            "customers, so its predicted probabilities are calibrated to "
            "the real base rate rather than the sampled one."
            % (100.0 * fraction, "{:,}".format(n_eligible))))

    if fraction < 1.0:
        items.append((
            "warning", "Negatives are downsampled to %.0f%%" % (100.0 * fraction),
            "Lift is measured against the prevalence of the SAMPLED "
            "population, so the numbers in this report are not comparable "
            "to a run at a different fraction. Set "
            "<code>sampling.negative_customer_fraction: 1.0</code> in "
            "model_config.yaml before reporting final numbers. Whole "
            "customers are sampled, never rows, and retained negatives "
            "carry weight %.1f so predicted probabilities stay near the "
            "true base rate." % (1.0 / fraction)))
    else:
        items.append((
            "good", "All negatives retained",
            "sampling.negative_customer_fraction is 1.0, so lift is "
            "measured against the true population prevalence."))

    if converged_flag is False:
        items.append((
            "critical", "Non-converged fit",
            "The persisted model hit its iteration cap at %s. Do not "
            "report its coefficients."
            % "{:,}".format(n_iterations or 0)))
    elif converged_flag is True:
        items.append((
            "good", "The persisted fit converged",
            "saga stopped at %s iterations, inside the "
            "<code>logistic.max_iter</code> cap of %s, so the "
            "coefficients are safe to read."
            % ("{:,}".format(n_iterations or 0),
               "{:,}".format(cfg_mdl["model"]["logistic"]["max_iter"]))))

    if "sign_consistency" in features.columns:
        floor = cfg_rep["sign_consistency_floor"]
        unproven = int((features["sign_consistency"] < floor).sum())
        share = unproven / float(len(features)) if len(features) else 0.0
        items.append((
            "warning" if share > 0.5 else "neutral",
            "%d of %d coefficients are unproven" % (unproven, len(features)),
            "Their sign flipped across folds more than %.0f%% of the time. "
            "That is noise, not a finding — do not build a narrative on "
            "them." % (100.0 * (1 - floor))))

    events = int(scores["actual"].sum()) if len(scores) else 0
    items.append((
        "neutral" if events >= 200 else "warning",
        "%s events support this fit" % "{:,}".format(events),
        "Row count is not sample size. In the hazard design one customer "
        "owns many rows, but it is the event count that limits how many "
        "features the data can carry — here roughly %d."
        % max(1, events // 10)))

    if len(deciles):
        top_lift = float(deciles.iloc[0]["lift"])
        items.append((
            "good" if top_lift >= cfg_rep["weak_lift_threshold"] else "warning",
            "Top-decile lift is %.2fx" % top_lift,
            "Calling the top tenth reaches %.2fx the %s rate of "
            "calling at random.%s"
            % (top_lift, ctx.labels["event_name"],
               "" if top_lift >= cfg_rep["weak_lift_threshold"] else
               " Below %.1fx the ranking is barely separating the "
               "population, and a campaign built on it will not beat a "
               "simple rule by much."
               % cfg_rep["weak_lift_threshold"])))

    # Saturation: in the hazard design the survival product can pin a
    # customer at 1.0, and a ranking of ties is not a ranking.
    if len(scores):
        saturated = float((scores["score"] > 0.999).mean())
        if saturated > 0.01:
            items.append((
                "warning", "%.1f%% of scores are pinned above 0.999"
                % (100.0 * saturated),
                "P(convert by H) = 1 - prod(1 - h_k) saturates once the "
                "per-interval hazards are large, so the top of the "
                "ranking is a block of near-ties and the order within it "
                "is arbitrary. Downsampling inflates this: at "
                "<code>negative_customer_fraction: %.2f</code> the "
                "prevalence the model sees is %.1f%%, far above the true "
                "base rate. Restore the fraction to 1.0, or shorten "
                "<code>evaluation.hazard_horizon_months</code>, before "
                "using the order within the top decile to prioritise "
                "calls." % (fraction, 100.0 * scores["actual"].mean())))

    items.append((
        "neutral", "Leakage controls in force",
        "No feature uses a row dated after its anchor, and none encodes "
        "the distance from the anchor to the outcome or the extract date. "
        "Gaps are measured between acquisitions only. Percentile grids "
        "were fitted once on training rows and reloaded, never refit at "
        "scoring time. CV splits on the customer, never the row."))

    return html.card(html.insights(items))


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
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    ctx = config.ReportContext(args.config)
    steps = ctx.cfg["steps"]
    if not args.no_run:
        if steps["generate_synthetic"]:
            config.run_synthetic(ctx)
        if steps["run_preprocess"]:
            config.run_preprocess(ctx)
        if steps["run_model"]:
            config.run_model(ctx)

    model_module = config.model_module(ctx)
    pre = config.preprocess_module(ctx)
    cfg_mdl = ctx.mdl
    col = cfg_mdl["columns"]

    # ---- artifacts from the modelling stage --------------------------
    for name in ["metrics.csv", "metrics_by_repeat.csv", "predictions.csv",
                 "features.csv"]:
        if not os.path.exists(ctx.model_output(name)):
            raise SystemExit(
                "%s is missing. Run with steps.run_model: true, or drop "
                "--no-run." % ctx.model_output(name))

    metrics = pd.read_csv(ctx.model_output("metrics.csv"))
    per_repeat = pd.read_csv(ctx.model_output("metrics_by_repeat.csv"))
    calibration = _maybe_read(ctx.model_output("calibration.csv"))
    features = pd.read_csv(ctx.model_output("features.csv"))
    predictions = pd.read_csv(ctx.model_output("predictions.csv"))

    print("\nloading the modelling dataset for the refit...")
    df = model_module.load_data(cfg_mdl)
    numeric, categorical = model_module.split_column_types(df, cfg_mdl)

    # model.load_data ALWAYS applies sampling.negative_customer_fraction,
    # so `df` above is the sampled frame -- the same one CV ran on. The
    # persisted artifact is meant for scoring new customers, so it wants
    # the whole population instead: a model fit on a 25% negative sample
    # predicts probabilities well above the true base rate. Reload with
    # the fraction forced to 1.0 to get it.
    fraction = cfg_mdl.get("sampling", {}).get("negative_customer_fraction", 1.0)
    refit_full = ctx.cfg["targeting"].get("refit_on_full_population", True)
    if refit_full and fraction < 1.0:
        cfg_full = copy.deepcopy(cfg_mdl)
        cfg_full.setdefault("sampling", {})["negative_customer_fraction"] = 1.0
        print("reloading the full population for the refit "
              "(sampling is for evaluation, not for the deliverable)...")
        refit_frame = model_module.load_data(cfg_full)
        refit_downsampled = False
    else:
        refit_frame = df
        refit_downsampled = fraction < 1.0

    # ---- the deliverable model ---------------------------------------
    artifact_path = ctx.cfg["io"]["model_artifact"]
    print("refitting on %d rows (%d customers) and persisting to %s"
          % (len(refit_frame),
             refit_frame[col["group"]].nunique(), artifact_path))
    pipe, bundle = targeting.refit_and_persist(
        model_module, refit_frame, numeric, categorical, cfg_mdl,
        artifact_path, downsampled=refit_downsampled)
    converged_flag, n_iterations = targeting.converged(pipe)

    # ---- ranking, on out-of-fold scores ------------------------------
    scores = targeting.customer_scores(predictions, cfg_mdl)
    deciles = targeting.decile_table(scores, ctx.cfg["targeting"]["score_deciles"])

    # One feature row per customer. In the hazard design a customer owns
    # many rows carrying the same time-fixed features, so the first row
    # is representative of everything except the interval bookkeeping --
    # which is excluded from reason codes and segments by name below,
    # not by being absent from this frame.
    per_customer = refit_frame.drop_duplicates(subset=[col["group"]],
                                               keep="first")
    per_customer = per_customer.set_index(col["group"])
    n_eligible = int(per_customer.shape[0])

    ranked = scores[scores[col["id"]].isin(per_customer.index)].copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["decile"] = np.minimum(
        (np.arange(len(ranked)) * ctx.cfg["targeting"]["score_deciles"])
        // max(1, len(ranked)) + 1, ctx.cfg["targeting"]["score_deciles"])

    # ---- high-propensity subset --------------------------------------
    cut = max(1, int(round(len(ranked) *
                           ctx.cfg["targeting"]["high_propensity_fraction"])))
    high_ids = ranked[col["id"]].head(cut).tolist()
    high = per_customer.loc[[c for c in high_ids if c in per_customer.index]]
    print("high-propensity subset: %d customers (top %.0f%%)"
          % (len(high), 100.0 * ctx.cfg["targeting"]["high_propensity_fraction"]))

    # ---- segments ----------------------------------------------------
    scores_indexed = ranked.set_index(col["id"])
    segment_result = targeting.segment_population(
        high, numeric, ctx.cfg["targeting"]["segments"],
        scores=scores_indexed["score"])

    # ---- reason codes for the call list ------------------------------
    top_n = min(ctx.cfg["targeting"]["top_n"], len(ranked))
    target = ranked.head(top_n).copy()
    target_rows = per_customer.loc[target[col["id"]].tolist()]
    # The hazard interval index is a real model input but never a
    # reason to call anybody: it is bookkeeping and takes the same value
    # for every row scored here.
    ranked_features = (features["feature"].tolist()
                       if "feature" in features.columns else None)
    reasons, reason_kind = targeting.reason_codes(
        pipe, target_rows[numeric + categorical], numeric, categorical,
        top_k=ctx.cfg["targeting"]["reason_codes"],
        exclude_columns=ctx.schema["hazard_bookkeeping"],
        ranked_features=ranked_features)
    target["reason_codes"] = reasons
    customer_col = ctx.schema["customer"]
    target = target.rename(columns={col["id"]: customer_col})

    names = targeting.segment_names(segment_result)
    if segment_result is not None:
        assigned = segment_result["assignment"].map(names)
        target["segment"] = target[customer_col].map(assigned).fillna(
            "outside the high-propensity subset")
    else:
        target["segment"] = "not segmented"

    csv_path = ctx.report_path("target_list")
    target[["rank", customer_col, "score", "score_min", "score_max",
            "decile", "segment", "reason_codes", "actual"]].to_csv(
                csv_path, index=False)

    # ---- sequences on the high-propensity subset ----------------------
    # Same functions eda.py runs on the whole population.
    print("profiling sequences for the high-propensity subset...")
    raw = pre.load_data(ctx.pre)
    acq, out = pre.split_outcome_rows(raw, ctx.pre)
    cfg_seq = ctx.cfg["sequences"]
    anchors = pre.compute_anchors(acq, out, ctx.pre, mode=cfg_seq["anchor_mode"])
    seqs, labels = sequences.build_sequences(acq, anchors, ctx.pre,
                                             cfg_seq["event_column"],
                                             ctx.schema["anchors"])
    high_profile = sequences.profile(seqs, labels, cfg_seq, customers=high_ids)
    all_profile = sequences.profile(seqs, labels, cfg_seq)
    sequence_block = sequences.render(
        high_profile, cfg_seq, charts, html, ctx.labels,
        heading_note="Population: the %s highest-scoring customers only "
                     "(%.0f%% of those scored). The equivalent table for "
                     "everybody is in eda_report.html — comparing the two "
                     "shows which paths the model is actually keying on. "
                     "Base rate here is %.2f%% against %.2f%% across the "
                     "whole eligible population."
                     % ("{:,}".format(high_profile["n_customers"]),
                        100.0 * ctx.cfg["targeting"]["high_propensity_fraction"],
                        100.0 * (high_profile["base_rate"] or 0),
                        100.0 * (all_profile["base_rate"] or 0)))

    # ---- assemble ----------------------------------------------------
    print("building sections...")
    kpis, config_block = run_summary(ctx, df, cfg_mdl, scores, bundle)
    performance, _lift_columns = performance_section(
        ctx, metrics, per_repeat, calibration, scores)
    feature_block = feature_section(ctx, features, converged_flag, n_iterations)
    score_block = score_section(ctx, scores, deciles)
    segment_block, segment_table = segment_section(ctx, segment_result, high,
                                                   scores_indexed)
    target_block = target_section(ctx, target, reason_kind, csv_path,
                                  customer_col, len(ranked), n_eligible)
    risk_block = risk_section(ctx, cfg_mdl, features, scores, converged_flag,
                              n_iterations, deciles, n_eligible, bundle)

    if segment_table is not None:
        segment_table.to_csv(ctx.report_path("segment_profile"), index=False)

    toc = [("risk", "Model risk"), ("run", "Run summary"),
           ("performance", "Performance"), ("features", "Feature insights"),
           ("scores", "Score profile"), ("segments", "Segments"),
           ("targets", "Next best customers"), ("sequences", "Their paths")]

    body = [
        kpis,
        html.section("risk", "Read this first",
                     "What would stop these numbers being usable, derived "
                     "from the run rather than written by hand.",
                     risk_block),
        html.section("run", "Run summary",
                     "What was fitted, on what, with which settings.",
                     config_block),
        html.section("performance", "Performance",
                     "PR-AUC and lift at the top of the ranking are the "
                     "numbers that matter for a rare event. ROC-AUC is "
                     "shown for reference.",
                     performance),
        html.section("features", "Feature insights",
                     "What the model leans on, and which of it is stable "
                     "enough across folds to be worth believing.",
                     feature_block),
        html.section("scores", "Score profile over the population",
                     "How the scores are distributed, and what a campaign "
                     "gets for calling down the ranking.",
                     score_block),
        html.section("segments", "Segments within the high-propensity population",
                     "The top-scoring customers are not one audience. "
                     "These are the groups inside them, described by what "
                     "makes each different from the rest.",
                     segment_block),
        html.section("targets", "Next best customers",
                     "The call list: who to contact, how confident the "
                     "model is, and why it picked them.",
                     target_block),
        html.section("sequences", "What the high-propensity customers did",
                     "The same sequence profiling eda.py runs on everyone, "
                     "restricted to the customers this model likes.",
                     sequence_block),
    ]

    meta = ("%s · %s design · %s model · %s customers scored"
            % (os.path.basename(cfg_mdl["io"]["input_path"]),
               cfg_mdl["design"], cfg_mdl["model"]["type"],
               "{:,}".format(len(scores))))
    document = html.page(ctx.labels["model_title"],
                         ctx.labels["model_subtitle"], meta, body, toc)

    path = html.write(ctx.report_path("model_report"), document)
    print("\nwrote %s (%.1f KB)" % (path, os.path.getsize(path) / 1024.0))
    print("wrote %s (%d customers)" % (csv_path, len(target)))
    print("wrote %s" % artifact_path)


def _maybe_read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


if __name__ == "__main__":
    main()
