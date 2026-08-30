"""
matplotlib -> base64 <img> helpers.

Every chart is a static PNG embedded straight into the HTML, so a report
is a single file you can mail to someone with no server, no CDN and no
javascript. That choice costs the hover layer, so the rule here is that
**no chart is the only way to read a value** -- each one ships beside a
table in the report body, and the extremes are direct-labelled.

Colour follows a validated palette rather than matplotlib defaults:

    categorical   fixed slot order, never cycled, never generated past 8
    sequential    one hue, light -> dark, for magnitude only
    diverging     blue <-> red with a neutral grey midpoint, for polarity
                  (lift above / below 1.0, WoE above / below zero)
    status        reserved for good / warning / bad, never for a series

Two rules worth stating because they are easy to break later:
  * a value-ramp is never used on nominal categories -- one series is
    one colour, and bar length already encodes the magnitude;
  * there is no dual-axis chart anywhere. Two measures on different
    scales get two charts.
"""

import base64
import html as _stdlib_html
import io

import matplotlib
matplotlib.use("Agg")            # no display; must precede pyplot

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed categorical order. Assign by slot, never by rank.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# One hue, light -> dark.
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
              "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"]

DIVERGING_LOW = "#2a78d6"        # cool pole
DIVERGING_MID = "#f0efec"        # neutral -- reads as "nothing"
DIVERGING_HIGH = "#d03b3b"       # warm pole

STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

FONT = ["Segoe UI", "DejaVu Sans", "sans-serif"]

_seq_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "nbo_seq", SEQUENTIAL)
_div_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "nbo_div", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH])


# ---------------------------------------------------------------------
# figure plumbing
# ---------------------------------------------------------------------

def _new(width=8.0, height=4.0):
    fig, ax = plt.subplots(figsize=(width, height), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _style(ax, xlabel=None, ylabel=None, title=None, grid_axis="y"):
    """Hairline solid grid, recessive axes, no top/right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9, labelpad=8)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=10)


def _legend(ax, **kwargs):
    leg = ax.legend(frameon=False, fontsize=8.5, **kwargs)
    for text in leg.get_texts():
        text.set_color(INK_2)
    return leg


def render(fig, alt=""):
    """Serialise a figure to an <img> tag and release it."""
    plt.rcParams["font.family"] = FONT
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    safe = (alt or "chart").replace('"', "'")
    return ('<img class="chart" alt="%s" src="data:image/png;base64,%s">'
            % (safe, data))


def empty(message):
    """Stand-in when there is nothing to plot. Says why, not nothing."""
    return '<p class="empty">%s</p>' % _stdlib_html.escape(str(message))


def _finite(values):
    """
    Drop non-finite values before any axis maths.

    NaN is TRUTHY in Python, so the usual `span = (hi - lo) or 1.0`
    fallback does not rescue an all-NaN series -- it propagates NaN into
    set_xlim, which raises. Every span and limit below goes through
    this first.
    """
    out = []
    for v in values:
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            out.append(v)
    return out


def _short(labels, width=34):
    out = []
    for text in labels:
        text = str(text)
        out.append(text if len(text) <= width else text[:width - 1] + "…")
    return out


def _fmt_pct(value, _pos=None):
    return "%.1f%%" % (100.0 * value)


# ---------------------------------------------------------------------
# bars
# ---------------------------------------------------------------------

def barh(labels, values, xlabel="", title="", value_fmt="%.2f",
         highlight=None, color=None, height=None, alt=""):
    """
    Horizontal bars, biggest at the top. One series, one colour --
    length already carries the magnitude, so hue stays free.

    `highlight` is an index whose bar is drawn in slot 1 while the rest
    recede, for the "one number is the story" case.
    """
    if len(labels) == 0:
        return empty("No rows met the support threshold, so there is "
                     "nothing to rank here.")
    labels = list(labels)[::-1]
    values = list(values)[::-1]
    finite = _finite(values)
    if not finite:
        return empty("Every value in this chart is missing, so there is "
                     "nothing to draw.")
    n = len(labels)
    fig, ax = _new(8.0, height or max(2.0, 0.34 * n + 1.1))

    if highlight is not None:
        highlight = n - 1 - highlight
        colors = [CATEGORICAL[0] if i == highlight else "#c9d6e5"
                  for i in range(n)]
    else:
        colors = color or CATEGORICAL[0]

    ax.barh(range(n), values, color=colors, height=0.68)
    ax.set_yticks(range(n))
    ax.set_yticklabels(_short(labels), fontsize=8.5)
    _style(ax, xlabel=xlabel, title=title, grid_axis="x")

    span = (max(finite) - min(0, min(finite))) or 1.0
    for i, v in enumerate(values):
        if not _finite([v]):
            continue
        ax.text(v + span * 0.012, i, value_fmt % v, va="center",
                fontsize=8, color=INK_2)
    ax.set_xlim(min(0, min(finite) * 1.05), max(finite) * 1.16)
    return render(fig, alt or title)


def barv(labels, values, ylabel="", title="", value_fmt=None, alt=""):
    """Vertical bars for a small nominal alphabet (product families...)."""
    if len(labels) == 0:
        return empty("No categories to show.")
    fig, ax = _new(8.0, 3.4)
    ax.bar(range(len(labels)), values, color=CATEGORICAL[0], width=0.66)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(_short(labels, 18), rotation=30, ha="right", fontsize=8.5)
    _style(ax, ylabel=ylabel, title=title)
    if value_fmt:
        top = max(values) or 1.0
        for i, v in enumerate(values):
            ax.text(i, v + top * 0.02, value_fmt % v, ha="center",
                    fontsize=8, color=INK_2)
        ax.set_ylim(0, top * 1.14)
    return render(fig, alt or title)


def grouped_barh(labels, series, xlabel="", title="", value_fmt="%.3f", alt=""):
    """
    Two or three series side by side. A legend is always present, and
    with <= 3 series every one is also direct-labelled, so identity is
    never carried by colour alone.
    """
    if len(labels) == 0:
        return empty("Nothing to compare.")
    names = list(series)
    labels = list(labels)[::-1]
    n, k = len(labels), len(names)
    fig, ax = _new(8.0, max(2.2, 0.30 * n * k + 1.2))
    span = 0.80 / k
    for j, name in enumerate(names):
        values = list(series[name])[::-1]
        offset = (j - (k - 1) / 2.0) * span
        ax.barh([i + offset for i in range(n)], values, height=span * 0.86,
                color=CATEGORICAL[j], label=name)
    ax.set_yticks(range(n))
    ax.set_yticklabels(_short(labels), fontsize=8.5)
    _style(ax, xlabel=xlabel, title=title, grid_axis="x")
    _legend(ax, loc="lower right")
    return render(fig, alt or title)


def diverging_barh(labels, values, center=1.0, xlabel="", title="",
                   value_fmt="%.2f", alt=""):
    """
    Polarity around a reference: lift vs 1.0, WoE vs 0. Two hues that
    read as opposite, neutral in the middle.
    """
    if len(labels) == 0:
        return empty("Nothing to compare against the baseline.")
    labels = list(labels)[::-1]
    values = list(values)[::-1]
    finite = _finite(values)
    if not finite:
        return empty("Every value in this chart is missing -- usually a base "
                     "rate of zero, which makes lift undefined.")
    n = len(labels)
    fig, ax = _new(8.0, max(2.0, 0.34 * n + 1.1))
    colors = [DIVERGING_HIGH if (v >= center) else DIVERGING_LOW
              for v in values]
    ax.barh(range(n), [v - center for v in values], left=center,
            color=colors, height=0.68)
    ax.axvline(center, color=AXIS, linewidth=1.0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(_short(labels), fontsize=8.5)
    _style(ax, xlabel=xlabel, title=title, grid_axis="x")

    span = (max(finite) - min(finite)) or 1.0
    for i, v in enumerate(values):
        if not _finite([v]):
            continue
        side = span * 0.02 if v >= center else -span * 0.02
        ax.text(v + side, i, value_fmt % v, va="center",
                ha="left" if v >= center else "right",
                fontsize=8, color=INK_2)
    pad = span * 0.22 + 1e-9
    ax.set_xlim(min(min(finite), center) - pad,
                max(max(finite), center) + pad)
    return render(fig, alt or title)


# ---------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------

def histogram(values, bins=40, xlabel="", title="", log_x=False, alt=""):
    values = np.asarray([v for v in np.asarray(values, dtype="float64")
                         if np.isfinite(v)])
    if values.size == 0:
        return empty("Every value in this column is missing.")
    fig, ax = _new(8.0, 3.2)
    if log_x:
        positive = values[values > 0]
        if positive.size == 0:
            return empty("No positive values, so a log scale cannot be drawn.")
        edges = np.logspace(np.log10(positive.min()),
                            np.log10(positive.max()), bins)
        ax.hist(positive, bins=edges, color=CATEGORICAL[0], rwidth=0.92)
        ax.set_xscale("log")
        xlabel = (xlabel or "") + "  (log scale)"
    else:
        ax.hist(values, bins=bins, color=CATEGORICAL[0], rwidth=0.92)
    _style(ax, xlabel=xlabel, ylabel="customers", title=title)
    median = float(np.median(values if not log_x else values[values > 0]))
    ax.axvline(median, color=CATEGORICAL[1], linewidth=1.6)
    ax.text(median, ax.get_ylim()[1] * 0.94, "  median %s" % _compact(median),
            fontsize=8, color=CATEGORICAL[1], va="top")
    return render(fig, alt or title)


def histogram_by_class(values, labels, class_names, bins=40, xlabel="",
                       title="", alt=""):
    """
    Overlaid densities for converters vs everyone else. Density, not
    count, or the rare class is invisible.
    """
    values = np.asarray(values, dtype="float64")
    labels = np.asarray(labels)
    ok = np.isfinite(values)
    values, labels = values[ok], labels[ok]
    if values.size == 0:
        return empty("No finite scores to plot.")
    fig, ax = _new(8.0, 3.2)
    edges = np.histogram_bin_edges(values, bins=bins)
    for j, (key, name) in enumerate(class_names):
        subset = values[labels == key]
        if subset.size == 0:
            continue
        ax.hist(subset, bins=edges, density=True, histtype="stepfilled",
                color=CATEGORICAL[j], alpha=0.30 if j == 0 else 0.45,
                label="%s (n=%d)" % (name, subset.size))
        ax.hist(subset, bins=edges, density=True, histtype="step",
                color=CATEGORICAL[j], linewidth=1.6)
    _style(ax, xlabel=xlabel, ylabel="density", title=title)
    _legend(ax)
    return render(fig, alt or title)


# ---------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------

def line(x, series, xlabel="", ylabel="", title="", reference=None,
         percent=False, markers=True, alt=""):
    """
    One shared y-axis, always. `reference` draws the diagonal / perfect
    calibration line in muted ink so it reads as chrome, not a series.
    """
    if len(x) == 0:
        return empty("No points to plot.")
    fig, ax = _new(7.2, 3.6)
    if reference is not None:
        ax.plot(reference[0], reference[1], color=MUTED, linewidth=1.0,
                label=reference[2] if len(reference) > 2 else None)
    for j, name in enumerate(series):
        ax.plot(x, series[name], color=CATEGORICAL[j], linewidth=2.0,
                marker="o" if markers else None, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    _style(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_pct))
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
    if len(series) > 1 or reference is not None:
        _legend(ax)
    return render(fig, alt or title)


def spread(labels, means, lows, highs, ylabel="", title="", alt=""):
    """Mean with a min-max whisker per metric -- spread across repeats."""
    if len(labels) == 0 or not _finite(means):
        return empty("No repeats to summarise.")
    fig, ax = _new(7.2, 3.2)
    xs = range(len(labels))
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        ax.plot([i, i], [lo, hi], color=AXIS, linewidth=1.4, zorder=1)
    ax.scatter(list(xs), means, s=64, color=CATEGORICAL[0], zorder=2,
               edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(_short(labels, 16), rotation=20, ha="right", fontsize=8.5)
    _style(ax, ylabel=ylabel, title=title)
    for i, m in enumerate(means):
        if _finite([m]):
            ax.text(i, m, "  %.3f" % m, fontsize=8, color=INK_2, va="center")
    return render(fig, alt or title)


# ---------------------------------------------------------------------
# heatmaps
# ---------------------------------------------------------------------

def heatmap(matrix, row_labels, col_labels, title="", mode="sequential",
            center=0.0, value_fmt="%.2f", cbar_label="", alt=""):
    """
    Transition matrices. `mode="sequential"` for a probability (one hue,
    light -> dark); `mode="diverging"` for a difference against a
    neutral centre. Cells are labelled, so the colour scale is a second
    encoding rather than the only one.
    """
    matrix = np.asarray(matrix, dtype="float64")
    if matrix.size == 0 or not np.isfinite(matrix).any():
        return empty("Not enough transitions to build a matrix.")
    n_rows, n_cols = matrix.shape
    fig, ax = _new(max(5.0, 0.78 * n_cols + 2.6),
                   max(3.0, 0.62 * n_rows + 1.8))

    if mode == "diverging":
        limit = float(np.nanmax(np.abs(matrix - center)))
        limit = limit if np.isfinite(limit) and limit else 1.0
        image = ax.imshow(matrix, cmap=_div_cmap, vmin=center - limit,
                          vmax=center + limit, aspect="auto")
    else:
        top = float(np.nanmax(matrix))
        image = ax.imshow(matrix, cmap=_seq_cmap, vmin=0.0,
                          vmax=top if np.isfinite(top) and top else 1.0,
                          aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(_short(col_labels, 14), rotation=30, ha="right",
                       fontsize=8.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(_short(row_labels, 20), fontsize=8.5)
    ax.tick_params(colors=MUTED, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=10)

    # 2px surface gap between cells instead of a border around each.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)

    span = float(np.nanmax(np.abs(matrix)))
    span = span if np.isfinite(span) and span else 1.0
    for i in range(n_rows):
        for j in range(n_cols):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            strong = abs(value - (center if mode == "diverging" else 0.0)) > 0.55 * span
            ax.text(j, i, value_fmt % value, ha="center", va="center",
                    fontsize=7.6, color="#ffffff" if strong else INK_2)

    bar = fig.colorbar(image, ax=ax, fraction=0.030, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    if cbar_label:
        bar.set_label(cbar_label, color=INK_2, fontsize=8.5)
    return render(fig, alt or title)


# ---------------------------------------------------------------------
# small helpers shared with html.py
# ---------------------------------------------------------------------

def _compact(value):
    """Human-readable magnitude: 12.4k, 3.1M."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= cut:
            return "%.1f%s" % (value / cut, suffix)
    return "%.2f" % value if abs(value) < 100 else "%.0f" % value
