"""
Interactive prospect map for sales reps (self-contained HTML, no server).

Buyers (stars), top prospects (colored by score) and optionally a light
background of other customers are projected to 2D from the same weighted
feature space the similarity score uses, so "close on the map" means
"similar on the features that matter".  A rep can:
  * filter to their own book (rep dropdown) or search a customer id
  * click any prospect to see: score band, plain-language reasons, its most
    similar historical buyers (drawn as links), and a profile table comparing
    the prospect's traits to those buyers, to all buyers, and to all customers
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def embed_2d(X: np.ndarray, method: str, perplexity: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
        except ImportError:
            print("[map] umap not installed, falling back to tsne")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        Xp = PCA(n_components=min(30, X.shape[1]), random_state=seed).fit_transform(X) if X.shape[1] > 30 else X
        return TSNE(n_components=2, perplexity=min(perplexity, max(5, len(X) // 4)), init="pca",
                    random_state=seed).fit_transform(Xp)
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def _jsval(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return str(v)


def profile_stats(df: pd.DataFrame, fields: list[str], types: dict, buyers: np.ndarray, cat_max_unique: int) -> dict:
    """Per-field reference stats embedded in the page: buyer / population quantile grids or level shares."""
    out = {}
    grid = np.linspace(0, 1, 101)
    for f in fields:
        raw = df[f]
        numeric = types.get(f) == "numeric" and pd.api.types.is_numeric_dtype(raw)
        if numeric:
            b, p = raw.iloc[buyers].dropna().values.astype(float), raw.dropna().values.astype(float)
            out[f] = dict(kind="num", buyers=np.quantile(b, grid).tolist() if len(b) else [],
                          pop=np.quantile(p, grid).tolist() if len(p) else [])
        else:
            bs = raw.iloc[buyers].astype(str).value_counts(normalize=True)
            ps = raw.astype(str).value_counts(normalize=True)
            out[f] = dict(kind="cat", buyers={k: round(float(v), 3) for k, v in bs.items()},
                          pop={k: round(float(v), 3) for k, v in ps.head(50).items()})
    return out


def build_map(cfg, df, feat_mat, buyers, arche, lead_rows, leads, prospects, out_path: Path, rng,
              profile_fields: list[str], types: dict) -> None:
    import plotly.graph_objects as go
    m, d = cfg["map"], cfg["data"]
    id_col, rep_col = d["id_col"], d.get("rep_col")

    # --- fields shown in the profile table: indicative features first, then extras from config
    extra = [c for c in (m.get("extra_fields") or []) if c in df.columns and c not in profile_fields]
    fields = list(profile_fields) + extra
    ref = profile_stats(df, fields, types, buyers, cfg["preprocess"]["categorical_max_unique"])

    # --- choose points
    top = lead_rows[: m["n_prospects"]]
    n_bg = m.get("n_background", 0) if m.get("show_background", True) else 0
    pool = np.setdiff1d(prospects, top)
    bg = rng.choice(pool, size=min(n_bg, len(pool)), replace=False) if n_bg else np.array([], dtype=int)
    all_rows = np.concatenate([buyers, top, bg])
    kind = np.r_[np.zeros(len(buyers), int), np.ones(len(top), int), np.full(len(bg), 2)]

    # --- 2D projection of the weighted feature space
    Z = embed_2d(feat_mat[all_rows], m["method"], m["perplexity"], m["random_state"])

    # --- nearest buyers per prospect (in feature space, not map space)
    Xb, Xp = feat_mat[buyers], feat_mat[top]
    k = min(m["k_links"], len(buyers))
    D = ((Xp[:, None, :] - Xb[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(D, axis=1)[:, :k]

    ids = df[id_col].astype(str).values
    reps = df[rep_col].astype(str).values if rep_col else np.array([""] * len(df))
    lead_info = leads.set_index(leads[id_col].astype(str))
    zb, zp, zg = Z[kind == 0], Z[kind == 1], Z[kind == 2]

    fig = go.Figure()
    if len(bg):
        fig.add_trace(go.Scattergl(x=zg[:, 0], y=zg[:, 1], mode="markers", name="Other customers",
                                   marker=dict(size=4, color="#d5d8dd", opacity=0.45), hoverinfo="skip"))
    else:  # keep trace indices stable
        fig.add_trace(go.Scattergl(x=[], y=[], mode="markers", name="Other customers", visible="legendonly"))
    pct = lead_info.loc[ids[top], "score_percentile"].values
    band = lead_info.loc[ids[top], "band"].values
    why = lead_info.loc[ids[top], "why_similar"].values
    raw_vals = df[fields]
    custom = [[ids[r], reps[r], band[i], float(pct[i]), why[i], json.dumps(nn[i].tolist()),
               json.dumps({f: _jsval(raw_vals.iat[r, j]) for j, f in enumerate(fields)})]
              for i, r in enumerate(top)]
    fig.add_trace(go.Scattergl(
        x=zp[:, 0], y=zp[:, 1], mode="markers", name="Prospects (click one)",
        marker=dict(size=9, color=pct, colorscale="YlOrRd", cmin=float(pct.min()), cmax=100,
                    colorbar=dict(title="Score<br>percentile", thickness=12), line=dict(width=0.5, color="#555")),
        customdata=custom,
        hovertemplate="<b>%{customdata[0]}</b><br>Rep: %{customdata[1]}<br>%{customdata[2]}<br>"
                      "Score pct: %{customdata[3]:.1f}<extra></extra>"))
    n_types = int(arche.max()) + 1
    palette = ["#1f4e79", "#2e8b57", "#8b2e8b", "#b8860b"]
    for t in range(n_types):
        sel = arche == t
        fig.add_trace(go.Scattergl(
            x=zb[sel, 0], y=zb[sel, 1], mode="markers",
            name="Existing buyers" if n_types == 1 else f"Buyers — type {t + 1}",
            marker=dict(size=13, symbol="star", color=palette[t % 4], line=dict(width=1, color="white")),
            customdata=[[ids[b], f"type {t + 1}"] for b in buyers[sel]],
            hovertemplate="<b>Buyer %{customdata[0]}</b><br>%{customdata[1]}<extra></extra>"))
    fig.update_layout(template="plotly_white", height=m.get("height", 680), autosize=True,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), dragmode="pan",
                      legend=dict(orientation="h", y=1.02, x=0))

    buyer_xy = zb.tolist()
    buyer_ids = ids[buyers].tolist()
    buyer_vals = {f: [_jsval(v) for v in df[f].iloc[buyers].values] for f in fields}
    rep_options = sorted(set(reps[top])) if rep_col else []
    plot_html = fig.to_html(full_html=False, include_plotlyjs=True if m.get("embed_plotlyjs", True) else "cdn",
                            div_id="map", config=dict(scrollZoom=True, displaylogo=False))
    n_ind = len(profile_fields)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Prospect map</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f7f8fa;color:#222}}
 header{{background:#1f4e79;color:#fff;padding:12px 20px;font-size:18px}}
 .wrap{{display:flex;gap:14px;padding:14px;align-items:flex-start}}
 .left{{flex:3;min-width:0;overflow:hidden;background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px #0002}}
 #map{{width:100%}}
 .right{{flex:1.6;min-width:380px;background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 3px #0002}}
 .bar{{display:flex;gap:10px;align-items:center;margin-bottom:8px;font-size:14px;flex-wrap:wrap}}
 select,input{{padding:6px;font-size:14px}}
 .band{{display:inline-block;padding:2px 8px;border-radius:10px;background:#ffe08a;font-weight:600}}
 .reason{{margin:6px 0;padding:8px;background:#f2f5f9;border-left:3px solid #1f4e79;font-size:13px}}
 .nb{{font-size:13px;margin:3px 0}} .muted{{color:#777;font-size:13px}}
 h3{{margin:0 0 8px 0}} h4{{margin:14px 0 6px 0}}
 table.prof{{width:100%;border-collapse:collapse;font-size:12.5px}}
 table.prof th{{text-align:left;background:#eef2f7;padding:5px 6px;font-weight:600}}
 table.prof td{{padding:5px 6px;border-bottom:1px solid #eee;vertical-align:top}}
 table.prof tr.extra td{{color:#555}}
 .pbar{{position:relative;height:8px;background:#e6e9ee;border-radius:4px;margin-top:4px}}
 .pbar .iqr{{position:absolute;top:0;height:8px;background:#b9c8dd;border-radius:4px}}
 .pbar .me{{position:absolute;top:-3px;width:4px;height:14px;background:#d7301f;border-radius:2px}}
 .match{{color:#2e7d32;font-weight:600}} .nomatch{{color:#999}}
 .legend{{font-size:11.5px;color:#777;margin-top:6px}}
</style></head><body>
<header>Prospect map — customers who look like existing buyers</header>
<div class="wrap">
 <div class="left">
  <div class="bar">
   {"<label>Rep:</label><select id='rep'><option value=''>All reps</option>" + "".join(f"<option>{r}</option>" for r in rep_options) + "</select>" if rep_col else ""}
   <label>Find customer:</label><input id="find" placeholder="customer id" size="16">
   <button onclick="findId()">Go</button>
   <span class="muted">Stars = existing buyers. Dots = top {len(top)} prospects, darker = higher score. Click a dot. Scroll to zoom, drag to pan, double-click to reset.</span>
  </div>
  {plot_html}
 </div>
 <div class="right" id="panel">
  <h3>Select a prospect</h3>
  <p class="muted">Click any prospect dot to see how it compares to the buyers it most resembles, to all buyers, and to all customers.
  The map is drawn from the same customer traits the score uses, so a prospect sitting among stars is one whose profile matches past buyers.</p>
 </div>
</div>
<script>
const buyerXY={json.dumps(buyer_xy)}, buyerIds={json.dumps(buyer_ids)}, buyerVals={json.dumps(buyer_vals)};
const REF={json.dumps(ref)}, FIELDS={json.dumps(fields)}, N_IND={n_ind};
const gd=document.getElementById('map');
const PROSPECT=1; let linkTrace=null;
function clearLinks(){{ if(linkTrace!==null){{ Plotly.deleteTraces(gd,linkTrace); linkTrace=null; }} }}
function fmt(v){{ if(v===null||v===undefined) return 'n/a'; if(typeof v!=='number') return v;
  return Math.abs(v)>=100? v.toLocaleString(undefined,{{maximumFractionDigits:0}}) : Number(v.toPrecision(3)).toString(); }}
function pctile(grid,v){{ if(!grid.length||v===null) return null; let lo=0,hi=grid.length-1;
  while(lo<hi){{ const mid=(lo+hi)>>1; if(grid[mid]<v) lo=mid+1; else hi=mid; }} return lo; }}
function bar(grid,v){{ const p=pctile(grid,v); if(p===null) return '';
  return '<div class="pbar"><div class="iqr" style="left:25%;width:50%"></div><div class="me" style="left:calc('+p+'% - 2px)"></div></div>'; }}
function profileRows(vals,nn){{
  let rows='';
  FIELDS.forEach((f,idx)=>{{
    const r=REF[f]; const v=vals[f]; const sim=nn.map(i=>buyerVals[f][i]);
    const cls=idx>=N_IND?' class="extra"':'';
    if(r.kind==='num'){{
      const s=sim.filter(x=>x!==null); const simTxt=s.length? fmt(Math.min(...s))+' – '+fmt(Math.max(...s)) : 'n/a';
      const pb=pctile(r.buyers,v), pp=pctile(r.pop,v);
      rows+='<tr'+cls+'><td><b>'+f+'</b></td><td><b>'+fmt(v)+'</b>'+
        (pb!==null?'<div class="muted">higher than '+pb+'% of buyers, '+pp+'% of all customers</div>'+bar(r.buyers,v):'')+
        '</td><td>'+simTxt+'</td><td>'+fmt(r.buyers[50])+'<div class="muted">'+fmt(r.buyers[25])+' – '+fmt(r.buyers[75])+'</div></td>'+
        '<td>'+fmt(r.pop[50])+'<div class="muted">'+fmt(r.pop[25])+' – '+fmt(r.pop[75])+'</div></td></tr>';
    }} else {{
      const key=String(v); const same=sim.filter(x=>String(x)===key).length;
      const bshare=r.buyers[key]||0, pshare=r.pop[key]||0;
      const top=Object.entries(r.buyers).sort((a,b)=>b[1]-a[1])[0];
      rows+='<tr'+cls+'><td><b>'+f+'</b></td><td><b>'+fmt(v)+'</b></td>'+
        '<td><span class="'+(same>=nn.length/2?'match':'nomatch')+'">'+same+' of '+nn.length+' same</span></td>'+
        '<td>'+Math.round(bshare*100)+'% are "'+key+'"'+(top&&top[0]!==key?'<div class="muted">most common: '+top[0]+' ('+Math.round(top[1]*100)+'%)</div>':'')+'</td>'+
        '<td>'+Math.round(pshare*100)+'% are "'+key+'"</td></tr>';
    }}
  }});
  return rows;
}}
function select(pt){{
  const c=pt.customdata; const nn=JSON.parse(c[5]); const vals=JSON.parse(c[6]);
  clearLinks();
  const xs=[],ys=[]; nn.forEach(i=>{{xs.push(pt.x,buyerXY[i][0],null); ys.push(pt.y,buyerXY[i][1],null);}});
  Plotly.addTraces(gd,{{x:xs,y:ys,mode:'lines',line:{{color:'#1f4e79',width:1.5}},hoverinfo:'skip',showlegend:false}});
  linkTrace=gd.data.length-1;
  const reasons=c[4].split(' | ').map(r=>'<div class="reason">'+r+'</div>').join('');
  document.getElementById('panel').innerHTML=
   '<h3>Customer '+c[0]+'</h3>'+(c[1]?'<div class="muted">Rep: '+c[1]+'</div>':'')+
   '<p><span class="band">'+c[2]+'</span> &nbsp; score percentile '+c[3].toFixed(1)+'</p>'+
   '<b>Why this customer looks like a buyer</b>'+reasons+
   '<h4>Profile: this prospect vs. buyers vs. everyone</h4>'+
   '<table class="prof"><tr><th>Trait</th><th>This prospect</th><th>'+nn.length+' most similar buyers</th><th>All buyers<br><span class="muted">median (middle 50%)</span></th><th>All customers<br><span class="muted">median (middle 50%)</span></th></tr>'+
   profileRows(vals,nn)+'</table>'+
   '<div class="legend">Bar: where the prospect sits among all buyers (shaded = middle 50% of buyers, red = this prospect).'+(FIELDS.length>N_IND?' Grey rows are extra reference fields, not used in scoring.':'')+'</div>'+
   '<h4>Most similar existing buyers</h4>'+nn.map(i=>'<div class="nb">&#9733; '+buyerIds[i]+'</div>').join('')+
   '<p class="muted">Ask the rep who owns these buyers what the conversation looked like — this prospect has the same profile.</p>';
}}
gd.on('plotly_click',ev=>{{ const p=ev.points[0]; if(p.curveNumber===PROSPECT) select(p); }});
const repSel=document.getElementById('rep');
if(repSel) repSel.onchange=()=>{{
  const r=repSel.value; const cd=gd.data[PROSPECT].customdata;
  const op=cd.map(c=>(!r||c[1]===r)?1:0.08);
  Plotly.restyle(gd,{{'marker.opacity':[op]}},[PROSPECT]); clearLinks();
}};
function findId(){{
  const id=document.getElementById('find').value.trim(); const cd=gd.data[PROSPECT].customdata;
  const i=cd.findIndex(c=>c[0]===id);
  if(i<0){{ document.getElementById('panel').innerHTML='<h3>Not found</h3><p class="muted">'+id+' is not among the '+cd.length+' plotted prospects. Check leads.csv for the full ranked list.</p>'; return; }}
  select({{x:gd.data[PROSPECT].x[i],y:gd.data[PROSPECT].y[i],customdata:cd[i]}});
  const xs=gd.data[PROSPECT].x, ys=gd.data[PROSPECT].y; const w=(Math.max(...xs)-Math.min(...xs))*0.15, h=(Math.max(...ys)-Math.min(...ys))*0.15;
  Plotly.relayout(gd,{{'xaxis.range':[xs[i]-w,xs[i]+w],'yaxis.range':[ys[i]-h,ys[i]+h]}});
}}
</script></body></html>"""
    Path(out_path).write_text(html, encoding="utf-8")
