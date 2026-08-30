"""
Self-contained HTML assembly. No jinja, no plotly, no CDN.

The output is one file with the CSS in a <style> block and every chart
inlined as a base64 PNG, so it survives being emailed, dropped on a
share drive, or opened years later with no network. That is worth more
here than interactivity: this is a document that goes to a marketing
team and, eventually, to model risk review.

Everything is built with f-strings and str.join. The only rule to keep
in mind when extending it: `esc()` everything that came from data.
Product names and customer ids are data.

The page commits to a light theme on purpose. The charts are PNGs baked
against the light chart surface, so a dark page would leave them
floating on the wrong ground.
"""

import datetime
import html as _html

import numpy as np
import pandas as pd

CSS = """
:root {
  color-scheme: light;
  --surface:    #fcfcfb;
  --plane:      #f4f4f1;
  --ink:        #0b0b0b;
  --ink-2:      #52514e;
  --muted:      #898781;
  --grid:       #e1e0d9;
  --rule:       #c3c2b7;
  --accent:     #2a78d6;
  --good:       #0ca30c;
  --warning:    #fab219;
  --serious:    #ec835a;
  --critical:   #d03b3b;
  --radius:     10px;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--plane); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px; line-height: 1.55;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 28px 96px; }

header.page {
  background: var(--surface); border-bottom: 1px solid var(--grid);
  padding: 40px 0 30px; margin-bottom: 28px;
}
header.page .wrap { padding-bottom: 0; }
header.page h1 { margin: 0 0 6px; font-size: 27px; letter-spacing: -0.015em; }
header.page p.sub { margin: 0; color: var(--ink-2); font-size: 14.5px; max-width: 76ch; }
header.page p.meta { margin: 14px 0 0; color: var(--muted); font-size: 12.5px; }

nav.toc {
  position: sticky; top: 0; z-index: 5; background: var(--plane);
  border-bottom: 1px solid var(--grid); margin-bottom: 30px;
  padding: 11px 0;
}
nav.toc .wrap { padding-bottom: 0; display: flex; flex-wrap: wrap; gap: 6px; }
nav.toc a {
  color: var(--ink-2); text-decoration: none; font-size: 12.5px;
  padding: 4px 11px; border-radius: 999px; border: 1px solid transparent;
}
nav.toc a:hover { background: var(--surface); border-color: var(--grid); color: var(--ink); }

section { margin: 0 0 46px; }
section > h2 {
  font-size: 19px; margin: 0 0 4px; letter-spacing: -0.01em;
  padding-top: 10px;
}
section > p.lede { margin: 0 0 20px; color: var(--ink-2); max-width: 84ch; }
h3 { font-size: 14.5px; margin: 28px 0 10px; color: var(--ink); }
h3:first-child { margin-top: 0; }

.card {
  background: var(--surface); border: 1px solid var(--grid);
  border-radius: var(--radius); padding: 20px 22px; margin: 0 0 18px;
}
.card > *:first-child { margin-top: 0; }
.card > *:last-child { margin-bottom: 0; }

/* ---- KPI row ---- */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 14px; margin-bottom: 26px; }
.kpi {
  background: var(--surface); border: 1px solid var(--grid);
  border-radius: var(--radius); padding: 16px 18px;
}
.kpi .label { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.kpi .value { font-size: 27px; font-weight: 600; margin: 5px 0 2px; letter-spacing: -0.02em; }
.kpi .note { font-size: 12px; color: var(--ink-2); }

/* ---- charts ---- */
img.chart { max-width: 100%; height: auto; display: block; margin: 6px 0 4px; }
.chart-note { font-size: 12.5px; color: var(--muted); margin: 2px 0 0; }
p.empty {
  font-size: 13px; color: var(--ink-2); background: var(--plane);
  border: 1px dashed var(--rule); border-radius: 8px; padding: 14px 16px;
}

/* ---- tables ---- */
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 12.8px; font-variant-numeric: tabular-nums; }
thead th {
  text-align: left; font-weight: 600; color: var(--ink-2); white-space: nowrap;
  border-bottom: 1.5px solid var(--rule); padding: 8px 12px 8px 0;
}
tbody td { padding: 7px 12px 7px 0; border-bottom: 1px solid var(--grid); white-space: nowrap; }
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; padding-right: 18px; }
td.wrap-cell { white-space: normal; min-width: 260px; }
tbody tr:hover { background: #f7f7f4; }
caption { caption-side: top; text-align: left; color: var(--muted); font-size: 12.5px; padding-bottom: 9px; }

/* ---- badges & callouts ---- */
.badge {
  display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 8px;
  border-radius: 999px; border: 1px solid; white-space: nowrap;
}
.badge.good     { color: #0a7a0a; border-color: #b7e3b7; background: #f0faf0; }
.badge.warning  { color: #8a6100; border-color: #f0dda6; background: #fdf8ec; }
.badge.serious  { color: #a1481f; border-color: #f4cdb9; background: #fdf3ee; }
.badge.critical { color: #a52020; border-color: #f0bcbc; background: #fdf0f0; }
.badge.neutral  { color: var(--ink-2); border-color: var(--grid); background: var(--plane); }

.callout {
  border-left: 3px solid var(--accent); background: var(--surface);
  border-radius: 0 8px 8px 0; padding: 13px 18px; margin: 0 0 12px;
  border-top: 1px solid var(--grid); border-right: 1px solid var(--grid);
  border-bottom: 1px solid var(--grid);
}
.callout.good     { border-left-color: var(--good); }
.callout.warning  { border-left-color: var(--warning); }
.callout.critical { border-left-color: var(--critical); }
.callout .t { font-weight: 600; display: block; margin-bottom: 2px; }
.callout .b { color: var(--ink-2); font-size: 13.2px; }

ul.plain { margin: 6px 0 0; padding-left: 18px; color: var(--ink-2); }
ul.plain li { margin-bottom: 5px; }
code, .mono { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 12.3px; }
.seq { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 12.2px; }
.cols2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 900px) { .cols2 { grid-template-columns: 1fr; } }
footer.page { border-top: 1px solid var(--grid); padding-top: 18px; color: var(--muted); font-size: 12.3px; }
"""


# =====================================================================
# escaping and formatting
# =====================================================================

def esc(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "&mdash;"
    return _html.escape(str(value), quote=True)


def fmt(value, spec=None):
    """Format one cell. None / NaN become an em dash, never 'nan'."""
    if value is None:
        return "&mdash;"
    if isinstance(value, float) and not np.isfinite(value):
        return "&mdash;"
    if isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if spec is None:
        if isinstance(value, float):
            return "%.4g" % value
        if isinstance(value, int):
            return "{:,}".format(value)
        return esc(value)
    if spec == "pct":
        return "%.2f%%" % (100.0 * float(value))
    if spec == "pct1":
        return "%.1f%%" % (100.0 * float(value))
    if spec == "int":
        return "{:,}".format(int(round(float(value))))
    return (spec % value) if "%" in spec else esc(value)


def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and \
        not isinstance(value, bool)


# =====================================================================
# blocks
# =====================================================================

def kpi_row(items):
    """items: list of (label, value, note)."""
    cards = []
    for label, value, note in items:
        cards.append(
            '<div class="kpi"><div class="label">%s</div>'
            '<div class="value">%s</div><div class="note">%s</div></div>'
            % (esc(label), value if isinstance(value, str) else fmt(value),
               esc(note) if note else "&nbsp;"))
    return '<div class="kpis">%s</div>' % "".join(cards)


def table(df, formats=None, caption=None, max_rows=None, wrap_columns=(),
          badges=None, index=False):
    """
    Render a DataFrame. `formats` maps column -> spec understood by
    fmt(); `badges` maps column -> callable(value, row) -> (kind, text)
    for a coloured pill instead of a plain value.

    Every chart in this report has a table like this beside it. That is
    deliberate: a static PNG has no tooltip, so the table is how a
    reader gets the actual number.
    """
    if df is None or len(df) == 0:
        return '<p class="empty">No rows.</p>'
    formats = formats or {}
    badges = badges or {}
    shown = df if max_rows is None else df.head(max_rows)

    columns = list(shown.columns)
    head = []
    if index:
        head.append("<th></th>")
    for name in columns:
        numeric = pd.api.types.is_numeric_dtype(shown[name]) and name not in badges
        head.append('<th class="%s">%s</th>'
                    % ("num" if numeric else "", esc(_pretty(name))))

    body = []
    for _, row in shown.iterrows():
        cells = []
        if index:
            cells.append("<td>%s</td>" % esc(row.name))
        for name in columns:
            value = row[name]
            if name in badges:
                kind, text = badges[name](value, row)
                cells.append('<td><span class="badge %s">%s</span></td>'
                             % (kind, esc(text)))
                continue
            klass = []
            if name in wrap_columns:
                klass.append("wrap-cell")
            elif _is_number(value):
                klass.append("num")
            cells.append('<td class="%s">%s</td>'
                         % (" ".join(klass), fmt(value, formats.get(name))))
        body.append("<tr>%s</tr>" % "".join(cells))

    note = ""
    if max_rows is not None and len(df) > max_rows:
        note = " Showing the first %d of %d rows." % (max_rows, len(df))
    cap = ('<caption>%s%s</caption>' % (esc(caption), esc(note))
           if (caption or note) else "")
    return ('<div class="scroll"><table>%s<thead><tr>%s</tr></thead>'
            '<tbody>%s</tbody></table></div>'
            % (cap, "".join(head), "".join(body)))


def _pretty(name):
    return str(name).replace("_", " ")


def callout(kind, title, body):
    return ('<div class="callout %s"><span class="t">%s</span>'
            '<span class="b">%s</span></div>'
            % (kind, esc(title), body if "<" in str(body) else esc(body)))


def insights(items):
    """
    items: list of (kind, title, body). These are generated from the
    numbers by the calling module, never hand-written -- so they stay
    true when the pipeline is pointed at a different dataset.
    """
    if not items:
        return callout("neutral", "Nothing flagged",
                       "No rule in the report fired against this run.")
    return "".join(callout(k, t, b) for k, t, b in items)


def card(*blocks):
    return '<div class="card">%s</div>' % "".join(b for b in blocks if b)


def note(text):
    return '<p class="chart-note">%s</p>' % esc(text)


def bullets(items):
    return '<ul class="plain">%s</ul>' % "".join(
        "<li>%s</li>" % (i if "<" in str(i) else esc(i)) for i in items)


def h3(text):
    return "<h3>%s</h3>" % esc(text)


def section(anchor, title, lede, *blocks):
    return ('<section id="%s"><h2>%s</h2><p class="lede">%s</p>%s</section>'
            % (esc(anchor), esc(title), esc(lede),
               "".join(b for b in blocks if b)))


# =====================================================================
# page
# =====================================================================

def page(title, subtitle, meta_line, sections, toc):
    """
    sections: list of rendered <section> strings.
    toc:      list of (anchor, label).
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    links = "".join('<a href="#%s">%s</a>' % (esc(a), esc(l)) for a, l in toc)
    return """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%s</title>
<style>%s</style>
</head><body>
<header class="page"><div class="wrap">
  <h1>%s</h1>
  <p class="sub">%s</p>
  <p class="meta">%s &middot; generated %s</p>
</div></header>
<nav class="toc"><div class="wrap">%s</div></nav>
<div class="wrap">%s
<footer class="page">
Generated by <code>nbo_report</code>. Charts are static PNGs embedded in
this file, so it works offline; every chart has a table beside it
carrying the same numbers.
</footer>
</div></body></html>
""" % (esc(title), CSS, esc(title), esc(subtitle), esc(meta_line), stamp,
       links, "".join(sections))


def write(path, content):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path
