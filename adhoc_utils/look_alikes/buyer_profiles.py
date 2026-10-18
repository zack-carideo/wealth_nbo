"""
Static buyer-type profiles: what each archetype (and buyers overall) looks like
compared to the typical customer.  Produces

  buyer_types.png        one-page card: rows = traits, columns = All buyers | Type 1 | Type 2 ...
                         numerics are shown on a "percentile of all customers" axis so every trait
                         is comparable; categoricals show share of the type vs share of everyone
  buyer_types.csv        the numbers behind the figure
  buyer_types.md         auto-written plain-language description of each type, for reps
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]   # validated categorical order
ALL_COLOR, POP_COLOR, INK, MUTED = "#52514e", "#b8b6b0", "#0b0b0b", "#52514e"


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.3g}"


def _pct_of_pop(pop_sorted: np.ndarray, v: float) -> float:
    return 100 * np.searchsorted(pop_sorted, v, side="right") / len(pop_sorted)


def build_profiles(df: pd.DataFrame, buyers: np.ndarray, arche: np.ndarray, fields: list[str], types: dict,
                   out_dir: Path, max_levels: int = 4) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = int(arche.max()) + 1
    groups = [("All buyers", buyers)] + [(f"Type {t + 1}", buyers[arche == t]) for t in range(k)] if k > 1 \
        else [("All buyers", buyers)]
    colors = [ALL_COLOR] + PALETTE[:k]
    n_all = len(df)

    # ---------- numbers
    rows, spec = [], []   # spec: per field, (kind, levels)
    for f in fields:
        raw = df[f]
        numeric = types.get(f) == "numeric" and pd.api.types.is_numeric_dtype(raw)
        if numeric:
            pop = np.sort(raw.dropna().values.astype(float))
            spec.append((f, "num", None))
            for name, idx in groups:
                b = raw.iloc[idx].dropna().values.astype(float)
                q = np.quantile(b, [.25, .5, .75]) if len(b) else [np.nan] * 3
                rows.append(dict(trait=f, group=name, n=len(idx), level="", value_p25=q[0], value_median=q[1],
                                 value_p75=q[2], pct_of_customers_p25=_pct_of_pop(pop, q[0]),
                                 pct_of_customers_median=_pct_of_pop(pop, q[1]),
                                 pct_of_customers_p75=_pct_of_pop(pop, q[2]),
                                 everyone_median=float(np.median(pop))))
        else:
            sv = raw.astype(str)
            pop_share = sv.value_counts(normalize=True)
            shares = {name: sv.iloc[idx].value_counts(normalize=True) for name, idx in groups}
            # levels to show: most over-represented in any group, then most common overall
            over = pd.concat([s / pop_share.reindex(s.index).fillna(1e-6) for s in shares.values()], axis=1).max(axis=1)
            levels = list(over.sort_values(ascending=False).index[:max_levels])
            levels = sorted(levels, key=lambda lv: -pop_share.get(lv, 0))   # display most common first
            spec.append((f, "cat", levels))
            for name, idx in groups:
                for lv in levels:
                    rows.append(dict(trait=f, group=name, n=len(idx), level=lv,
                                     share=float(shares[name].get(lv, 0)), everyone_share=float(pop_share.get(lv, 0))))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "buyer_types.csv", index=False)

    # ---------- figure
    heights = [1.0 if kind == "num" else 0.42 * len(lv) + 0.35 for _, kind, lv in spec]
    ncol = len(groups)
    fig_h = 1.1 + sum(heights)
    fig, axes = plt.subplots(len(spec), ncol, figsize=(3.1 * ncol + 1.6, fig_h),
                             gridspec_kw=dict(height_ratios=heights, hspace=0.55, wspace=0.25), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for j, ((name, idx), col) in enumerate(zip(groups, colors)):
        share = f"{len(idx) / len(buyers):.0%} of buyers" if name != "All buyers" else f"{len(idx) / n_all:.3%} of all customers"
        axes[0, j].set_title(f"{name}\n{len(idx)} · {share}", fontsize=10.5, color=col, fontweight="bold", pad=10)
    for i, (f, kind, levels) in enumerate(spec):
        sub = table[table.trait == f]
        for j, ((name, idx), col) in enumerate(zip(groups, colors)):
            ax = axes[i, j]
            ax.set_facecolor("#fcfcfb")
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#dddbd5")
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            r = sub[sub.group == name]
            if kind == "num":
                r = r.iloc[0]
                lo, md, hi = r.pct_of_customers_p25, r.pct_of_customers_median, r.pct_of_customers_p75
                ax.axvline(50, color=POP_COLOR, lw=1, ls=(0, (3, 3)), zorder=1)
                ax.barh(0, hi - lo, left=lo, height=0.42, color=col, alpha=0.35, zorder=2)
                ax.plot(md, 0, "o", ms=9, color=col, mec="#fcfcfb", mew=1.5, zorder=3)
                ax.text(md, 0.45, f"median {fmt(r.value_median)}", ha="center", va="bottom", fontsize=8.5, color=INK)
                ax.set_xlim(0, 100); ax.set_ylim(-0.7, 1.0); ax.set_yticks([])
                ax.set_xticks([0, 50, 100])
                ax.set_xticklabels(["low", "typical", "high"] if i == len(spec) - 1 or True else [])
            else:
                ys = np.arange(len(levels))[::-1]
                sh = [float(r[r.level == lv].share.iloc[0]) for lv in levels]
                ev = [float(r[r.level == lv].everyone_share.iloc[0]) for lv in levels]
                ax.barh(ys, sh, height=0.55, color=col, alpha=0.85, zorder=2)
                ax.barh(ys, ev, height=0.55, fill=False, edgecolor=POP_COLOR, lw=1.2, zorder=3)
                for y, s, e in zip(ys, sh, ev):
                    ax.text(max(s, e) + 0.02, y, f"{s:.0%} · all {e:.0%}", va="center", fontsize=8, color=INK)
                ax.set_yticks(ys); ax.set_yticklabels(levels if j == 0 else [""] * len(levels), fontsize=8.5, color=INK)
                ax.set_xlim(0, 1.75); ax.set_xticks([])
                ax.spines["bottom"].set_visible(False)
            if j == 0:
                ax.set_ylabel(f, rotation=0, ha="right", va="center", fontsize=10, color=INK, fontweight="bold", labelpad=14)
    fig.text(0.01, 0.005,
             "Numeric traits: dot = the group's median, band = its middle 50%, placed on a scale of where that falls among ALL customers "
             "(dashed line = the typical customer).\nCategorical traits: filled bar = share of the group, outline = share of all customers ('all').",
             fontsize=8, color=MUTED, va="bottom")
    fig.suptitle("What buyers look like — by buyer type", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.0 - 0.15 / fig_h)
    fig.savefig(out_dir / "buyer_types.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ---------- plain-language summary
    lines = ["# Buyer types\n"]
    for name, idx in groups:
        share = f"{len(idx) / len(buyers):.0%} of buyers" if name != "All buyers" else "all buyers"
        bits = []
        for f, kind, levels in spec:
            r = table[(table.trait == f) & (table.group == name)]
            if kind == "num":
                r = r.iloc[0]
                p = r.pct_of_customers_median
                if p >= 75 or p <= 25:
                    bits.append(f"{f} {'well above' if p >= 90 else 'above' if p >= 75 else 'well below' if p <= 10 else 'below'} typical "
                                f"(median {fmt(r.value_median)} vs {fmt(r.everyone_median)} for everyone)")
            else:
                r = r.assign(ratio=lambda t: t.share / t.everyone_share.clip(lower=0.01)).sort_values("ratio", ascending=False).iloc[0]
                if r.ratio >= 1.5 and r.share >= 0.10:
                    word = "mostly" if r.share >= 0.5 else "over-represented in"
                    bits.append(f"{f} {word} '{r.level}' ({r.share:.0%} vs {r.everyone_share:.0%} of everyone)")
        lines.append(f"## {name} — {len(idx)} customers, {share}\n")
        lines.append(("Stands out on: " + "; ".join(bits) + ".") if bits else "No trait stands out strongly for this group." )
        lines.append("")
    (out_dir / "buyer_types.md").write_text("\n".join(lines), encoding="utf-8")
    return table
