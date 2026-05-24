"""Build a self-contained results.html report from the analysis CSVs.

Reads aggregated tables from results/analysis_id_ood_all_budgets/ and emits
results.html with embedded data, styled tables, and Plotly charts.
"""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis_id_ood_all_budgets"
OUT = ROOT / "results.html"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def load(name, keys):
    rows = []
    with open(ANALYSIS / name) as f:
        for r in csv.DictReader(f):
            row = {}
            for k in keys:
                v = r[k]
                if k in {"family_label", "split", "axis_label", "task_label",
                         "baseline_config", "diverse_config"}:
                    row[k] = v
                elif k in {"budget", "seed_count"}:
                    row[k] = int(v)
                else:
                    row[k] = num(v)
            rows.append(row)
    return rows


def collect():
    return {
        "overall": load("overall_by_family_split_budget.csv",
            ["family_label","split","budget","seed_count",
             "success_at_1cm_pct","success_at_2cm_pct","success_at_5cm_pct",
             "nearest_distance_cm","end_distance_cm"]),
        "ood_axis": load("ood_axis_by_family_budget.csv",
            ["family_label","axis_label","budget","seed_count",
             "success_at_1cm_pct","success_at_2cm_pct","success_at_5cm_pct",
             "nearest_distance_cm","end_distance_cm"]),
        "ood_task": load("ood_task_by_family_budget.csv",
            ["family_label","task_label","budget","seed_count",
             "success_at_1cm_pct","success_at_2cm_pct","success_at_5cm_pct",
             "nearest_distance_cm","end_distance_cm"]),
        "axis_split": load("axis_by_family_split_budget.csv",
            ["family_label","split","axis_label","budget","seed_count",
             "success_at_1cm_pct","success_at_5cm_pct",
             "nearest_distance_cm","end_distance_cm"]),
        "task_split": load("task_by_family_split_budget.csv",
            ["family_label","split","task_label","budget","seed_count",
             "success_at_1cm_pct","success_at_5cm_pct",
             "nearest_distance_cm","end_distance_cm"]),
        "pair_gains": load("diversity_pair_gains_ood_by_family_budget.csv",
            ["family_label","task_label","axis_label","budget",
             "baseline_config","diverse_config",
             "baseline_success_at_1cm_pct","diverse_success_at_1cm_pct",
             "gain_success_at_1cm_pct",
             "baseline_success_at_5cm_pct","diverse_success_at_5cm_pct",
             "gain_success_at_5cm_pct",
             "baseline_nearest_distance_cm","diverse_nearest_distance_cm",
             "gain_nearest_distance_cm"]),
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-BCOOD Results</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {
  --bg: #fafaf7;
  --panel: #ffffff;
  --ink: #1a1a1a;
  --muted: #6b6b6b;
  --line: #e6e3dc;
  --accent: #2b4f81;
  --scratch: #2b4f81;
  --frozen: #2f7a4d;
  --partial: #c46a1d;
  --warn: #b8332a;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
.layout { display: grid; grid-template-columns: 220px minmax(0,1fr); min-height: 100vh; }
nav.toc {
  position: sticky;
  top: 0;
  align-self: start;
  height: 100vh;
  padding: 24px 18px;
  border-right: 1px solid var(--line);
  background: #f4f1ea;
  font-size: 13px;
  overflow-y: auto;
}
nav.toc h2 {
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 14px 0 6px;
}
nav.toc h2:first-child { margin-top: 0; }
nav.toc a {
  display: block;
  padding: 4px 8px;
  border-radius: 4px;
  color: var(--ink);
  text-decoration: none;
}
nav.toc a:hover { background: rgba(43,79,129,0.08); color: var(--accent); }
main {
  padding: 40px 56px 80px;
  max-width: 1100px;
}
header.title { border-bottom: 1px solid var(--line); padding-bottom: 24px; margin-bottom: 32px; }
header.title h1 { margin: 0 0 6px; font-size: 28px; font-weight: 600; letter-spacing: -0.01em; }
header.title .sub { color: var(--muted); font-size: 14px; }
header.title .meta { color: var(--muted); font-size: 12px; margin-top: 8px; }
.findings {
  margin-top: 22px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}
.finding {
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: 12px 14px;
}
.finding .num { font-size: 22px; font-weight: 600; color: var(--accent); }
.finding .label { font-size: 12px; color: var(--muted); margin-bottom: 2px; }
.finding .body { font-size: 13px; }
section { margin: 56px 0 0; scroll-margin-top: 16px; }
section > h2 {
  font-size: 20px;
  margin: 0 0 4px;
  font-weight: 600;
  letter-spacing: -0.01em;
}
section > .lede {
  color: var(--muted);
  margin: 0 0 20px;
  font-size: 14px;
}
.chart {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  margin: 0 0 22px;
}
.chart h3 { margin: 0 0 10px; font-size: 14px; font-weight: 600; }
.chart .pl { width: 100%; height: 380px; }
.chart .pl.tall { height: 460px; }
.row { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px,1fr)); gap: 18px; margin-bottom: 22px; }
.row .chart { margin: 0; }
table.data {
  border-collapse: collapse;
  width: 100%;
  font-size: 12.5px;
  margin: 6px 0 22px;
  background: var(--panel);
}
table.data th, table.data td {
  border-bottom: 1px solid var(--line);
  padding: 7px 10px;
  text-align: right;
}
table.data th:first-child, table.data td:first-child,
table.data th:nth-child(2), table.data td:nth-child(2) { text-align: left; }
table.data th {
  background: #f4f1ea;
  font-weight: 600;
  border-bottom: 1px solid #d8d4c8;
  position: sticky;
  top: 0;
}
table.data tbody tr:hover { background: #f8f6ef; }
table.data .fam-scratch { color: var(--scratch); font-weight: 600; }
table.data .fam-frozen { color: var(--frozen); font-weight: 600; }
table.data .fam-partial { color: var(--partial); font-weight: 600; }
table.data .small { color: var(--muted); font-size: 11px; }
.callout {
  background: #fdf6e3;
  border: 1px solid #e8dca8;
  border-left: 3px solid #c9a73b;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 13px;
  margin: 10px 0 22px;
}
.callout strong { color: #7d5e10; }
.note { font-size: 12px; color: var(--muted); margin-top: -10px; margin-bottom: 18px; }
.tag { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.tag-3 { background: #def0e6; color: #1a5e36; }
.tag-1 { background: #fde6cb; color: #874612; }
ul.tight { margin: 0; padding-left: 18px; }
ul.tight li { margin: 3px 0; }
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  nav.toc { position: static; height: auto; border-right: none; border-bottom: 1px solid var(--line); }
  main { padding: 24px; }
}
</style>
</head>
<body>
<div class="layout">
<nav class="toc">
  <h2>V-BCOOD</h2>
  <a href="#findings">Key findings</a>
  <a href="#methodology">Methodology</a>
  <h2>Results</h2>
  <a href="#overall">ID vs OOD</a>
  <a href="#budget">Budget scaling</a>
  <a href="#family">Model family</a>
  <a href="#axis">OOD visual axis</a>
  <a href="#task">Task basis</a>
  <a href="#threshold">Threshold sweep</a>
  <a href="#pairs">Diversity pair gains</a>
  <h2>Discussion</h2>
  <a href="#caveats">Caveats &amp; open questions</a>
</nav>

<main>
<header class="title">
  <h1>V-BCOOD: Visual Behavior Cloning Under Controlled OOD Shift</h1>
  <div class="sub">128&times;128 pixel control on Panda <em>reach</em> and <em>obstacle-aware reach</em>, sweeping demonstration budget, model family, and visual diversity axis.</div>
  <div class="meta">3 model families &middot; 4 visual axes &middot; 5 budgets &middot; 16 train configs &middot; up to 3 seeds per cell</div>

  <div class="findings" id="findings">
    <div class="finding">
      <div class="label">Headline OOD precision (budget 200)</div>
      <div class="num">43.5%</div>
      <div class="body">Scratch CNN OOD success@1cm. Beats Frozen ResNet-18 (13.8%) and Partial ResNet-18 (19.0%).</div>
    </div>
    <div class="finding">
      <div class="label">Loose threshold inflates success</div>
      <div class="num">2&times;</div>
      <div class="body">Scratch CNN budget 200 OOD: 86.6% @5cm vs 43.5% @1cm. Loose success hides precision failure.</div>
    </div>
    <div class="finding">
      <div class="label">Hardest visual shift</div>
      <div class="num">Spatial</div>
      <div class="body">Even at budget 200, OOD spatial success@1cm stays in single digits across all model families.</div>
    </div>
    <div class="finding">
      <div class="label">Pretrained encoders did not win</div>
      <div class="num">&minus;30pp</div>
      <div class="body">Frozen ResNet-18 trails Scratch CNN by ~30 percentage points on OOD@1cm at budget 200.</div>
    </div>
  </div>
</header>

<section id="methodology">
  <h2>Methodology</h2>
  <p class="lede">Controlled comparison built on PyBullet Panda with dual RGB observations and scripted experts.</p>
  <ul class="tight">
    <li><strong>Tasks:</strong> Reach (direct) and Obstacle-aware reach (route around blocking obstacle).</li>
    <li><strong>Visual axes:</strong> color, camera viewpoint, spatial distribution, lighting direction/intensity.</li>
    <li><strong>Train configs (16):</strong> for each axis, a narrow and a diverse condition, on each of the two tasks.</li>
    <li><strong>Demo budgets:</strong> 5, 20, 50 (3 seeds each); 100, 200 (1 seed, directional).</li>
    <li><strong>Model families:</strong> Scratch CNN, Frozen ResNet-18, Partial ResNet-18 (last block trainable).</li>
    <li><strong>Eval splits:</strong> ID (matched train condition) and OOD (held-out variant on each axis).</li>
    <li><strong>Metrics:</strong> success@<em>k</em>cm computed from <em>nearest distance reached during rollout</em>, plus end-of-episode distance for stability. Early stop at 0.1cm only.</li>
  </ul>
  <div class="callout"><strong>Caveat:</strong> Budgets 100 and 200 currently use seed 0 only; their numbers are directional. The 5/20/50 numbers are the statistically stronger anchor.</div>
</section>

<section id="overall">
  <h2>Overall ID vs OOD by family</h2>
  <p class="lede">How much does each policy degrade when visual conditions shift?</p>
  <div class="chart"><h3>OOD success@1cm vs ID success@1cm, by family and budget</h3><div class="pl tall" id="chart-overall"></div></div>
  <div id="table-overall"></div>
</section>

<section id="budget">
  <h2>Budget scaling</h2>
  <p class="lede">More demonstrations help, but the curve depends sharply on family and split.</p>
  <div class="row">
    <div class="chart"><h3>OOD success@1cm vs budget</h3><div class="pl" id="chart-budget-1cm"></div></div>
    <div class="chart"><h3>OOD nearest distance (cm) vs budget &middot; lower is better</h3><div class="pl" id="chart-budget-nearest"></div></div>
  </div>
</section>

<section id="family">
  <h2>Model family comparison</h2>
  <p class="lede">Frozen ResNet-18 is steadier in end distance; Scratch CNN dominates 1cm OOD precision.</p>
  <div class="row">
    <div class="chart"><h3>OOD success@1cm vs OOD success@5cm, budget 200</h3><div class="pl" id="chart-fam-thresh"></div></div>
    <div class="chart"><h3>OOD end distance (cm) by family/budget &middot; stability proxy</h3><div class="pl" id="chart-fam-end"></div></div>
  </div>
  <div class="callout">Frozen ResNet's lower end distance suggests smoother but less precise policies: the arm settles, but not on the target. Worth following up.</div>
</section>

<section id="axis">
  <h2>OOD visual axis breakdown</h2>
  <p class="lede">Which held-out shift is hardest? Spatial dominates the failure mode.</p>
  <div class="row">
    <div class="chart"><h3>Budget 50 (3 seeds) &middot; OOD success@1cm by axis</h3><div class="pl" id="chart-axis-50"></div></div>
    <div class="chart"><h3>Budget 200 (1 seed, directional) &middot; OOD success@1cm by axis</h3><div class="pl" id="chart-axis-200"></div></div>
  </div>
  <div id="table-axis-50"></div>
  <div id="table-axis-200"></div>
</section>

<section id="task">
  <h2>Task basis: reach vs obstacle-aware reach</h2>
  <p class="lede">Adding the obstacle collapses precision and end stability across every family.</p>
  <div class="row">
    <div class="chart"><h3>OOD success@1cm by task, budget 50 and 200</h3><div class="pl" id="chart-task"></div></div>
    <div class="chart"><h3>OOD end distance by task &middot; obstacle-aware drift</h3><div class="pl" id="chart-task-end"></div></div>
  </div>
  <div id="table-task"></div>
</section>

<section id="threshold">
  <h2>Precision threshold sweep</h2>
  <p class="lede">@5cm headline numbers can disguise large precision failures &mdash; the gap to @1cm is the methodological story.</p>
  <div class="chart"><h3>OOD success vs precision threshold, budget 200</h3><div class="pl" id="chart-thresh"></div></div>
</section>

<section id="pairs">
  <h2>Diversity pair gains: narrow vs diverse training</h2>
  <p class="lede">For each (axis &times; task) pair, training on the diverse condition is compared against the narrow one, evaluated on held-out OOD. Positive gain = diversity helped.</p>
  <div class="chart"><h3>OOD success@1cm gain (diverse &minus; narrow), budget 50 across all pairs</h3><div class="pl tall" id="chart-pairs"></div></div>
  <div id="table-pairs"></div>
  <div class="callout">This is the proposal&rsquo;s headline question. Gains are not uniform across axes &mdash; some shifts are helped by training diversity, others barely move.</div>
</section>

<section id="caveats">
  <h2>Caveats &amp; open questions</h2>
  <ul class="tight">
    <li><strong>Single-seed high budgets.</strong> 100 and 200 currently use seed 0 only. Run seeds 1, 2 before strong claims about the budget-200 step-up.</li>
    <li><strong>Spatial-shift hardness.</strong> All families collapse on spatial OOD &mdash; verify this is not just because the spatial OOD set is farther from training support than the other axes.</li>
    <li><strong>End-distance drift.</strong> Scratch CNN obstacle_reach end&nbsp;distance is large even when nearest distance is small; the policy reaches and then drifts. Frozen ResNet does not show this. Worth follow-up.</li>
    <li><strong>Pair-gain coverage.</strong> Pair gains are computed within each axis; cross-axis transfer (e.g., does spatial diversity help OOD camera?) is not yet reported.</li>
  </ul>
</section>

</main>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const FAMS = ['Scratch CNN','Frozen ResNet-18','Partial ResNet-18'];
const FAM_COLOR = {
  'Scratch CNN':'#2b4f81',
  'Frozen ResNet-18':'#2f7a4d',
  'Partial ResNet-18':'#c46a1d',
};
const FAM_KEY = {
  'Scratch CNN':'fam-scratch',
  'Frozen ResNet-18':'fam-frozen',
  'Partial ResNet-18':'fam-partial',
};
const PLOT_CFG = {displaylogo: false, responsive: true, modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']};
const PLOT_LAYOUT_BASE = {
  margin: {l: 50, r: 16, t: 30, b: 44},
  font: {family: 'system-ui, sans-serif', size: 12, color: '#1a1a1a'},
  paper_bgcolor: 'white',
  plot_bgcolor: 'white',
  legend: {orientation: 'h', yanchor:'bottom', y: 1.02, x: 0, font: {size: 11}},
  xaxis: {gridcolor: '#eee'},
  yaxis: {gridcolor: '#eee'},
};
const layout = (over={}) => Object.assign({}, structuredClone(PLOT_LAYOUT_BASE), over);

const fmtPct = v => (v == null ? '' : (Math.round(v*10)/10).toFixed(1));
const fmtCm  = v => (v == null ? '' : (Math.round(v*100)/100).toFixed(2));
const filt = (rows, pred) => rows.filter(pred);
const by = (rows, key) => { const m={}; rows.forEach(r=>{const k=r[key]; (m[k]=m[k]||[]).push(r);}); return m; };
const seedTag = n => n>=3 ? '<span class="tag tag-3">'+n+' seeds</span>' : '<span class="tag tag-1">'+n+' seed</span>';

/* ---------- Overall: ID vs OOD by family/budget (grouped bars) ---------- */
(function(){
  const budgets = [5,20,50,100,200];
  const traces = [];
  FAMS.forEach(f => {
    ['id','ood'].forEach(split => {
      const ys = budgets.map(b => {
        const r = DATA.overall.find(r => r.family_label===f && r.split===split && r.budget===b);
        return r ? r.success_at_1cm_pct : null;
      });
      traces.push({
        x: budgets.map(String), y: ys,
        type:'bar', name: f + ' · ' + split.toUpperCase(),
        marker:{color: FAM_COLOR[f], opacity: split==='id' ? 1 : 0.5,
                line:{color: FAM_COLOR[f], width: split==='ood' ? 1.5 : 0}},
      });
    });
  });
  Plotly.newPlot('chart-overall', traces, layout({
    barmode:'group',
    xaxis:{title:'Budget (demos)'},
    yaxis:{title:'success@1cm (%)', rangemode:'tozero'},
  }), PLOT_CFG);

  // Table
  const cols = ['Family','Split','Budget','Seeds','succ@1cm','succ@2cm','succ@5cm','nearest (cm)','end (cm)'];
  let html = '<table class="data"><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
  const sorted = [...DATA.overall].sort((a,b)=>{
    const fi = FAMS.indexOf(a.family_label) - FAMS.indexOf(b.family_label);
    if (fi) return fi;
    if (a.split !== b.split) return a.split === 'id' ? -1 : 1;
    return a.budget - b.budget;
  });
  sorted.forEach(r => {
    html += '<tr>'
      +'<td class="'+FAM_KEY[r.family_label]+'">'+r.family_label+'</td>'
      +'<td>'+r.split.toUpperCase()+'</td>'
      +'<td>'+r.budget+'</td>'
      +'<td>'+seedTag(r.seed_count)+'</td>'
      +'<td>'+fmtPct(r.success_at_1cm_pct)+'%</td>'
      +'<td>'+fmtPct(r.success_at_2cm_pct)+'%</td>'
      +'<td>'+fmtPct(r.success_at_5cm_pct)+'%</td>'
      +'<td>'+fmtCm(r.nearest_distance_cm)+'</td>'
      +'<td>'+fmtCm(r.end_distance_cm)+'</td>'
      +'</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('table-overall').innerHTML = html;
})();

/* ---------- Budget scaling: line charts ---------- */
(function(){
  const budgets = [5,20,50,100,200];
  function build(metric, divId, ytitle, splitFilter='ood'){
    const traces = FAMS.map(f => {
      const ys = budgets.map(b => {
        const r = DATA.overall.find(r => r.family_label===f && r.split===splitFilter && r.budget===b);
        return r ? r[metric] : null;
      });
      return {
        x: budgets, y: ys, mode:'lines+markers',
        name: f, line:{color: FAM_COLOR[f], width: 2.5}, marker:{size: 8},
      };
    });
    Plotly.newPlot(divId, traces, layout({
      xaxis:{title:'Budget (demos)', type:'log', tickvals: budgets, ticktext: budgets.map(String)},
      yaxis:{title: ytitle, rangemode:'tozero'},
    }), PLOT_CFG);
  }
  build('success_at_1cm_pct','chart-budget-1cm','OOD success@1cm (%)');
  build('nearest_distance_cm','chart-budget-nearest','OOD nearest distance (cm)');
})();

/* ---------- Family: 1cm vs 5cm at budget 200, end distance over budgets ---------- */
(function(){
  const rows = DATA.overall.filter(r => r.split==='ood' && r.budget===200);
  const families = rows.map(r => r.family_label);
  const traces = [
    {x: families, y: rows.map(r=>r.success_at_1cm_pct), type:'bar', name:'success@1cm', marker:{color:'#2b4f81'}},
    {x: families, y: rows.map(r=>r.success_at_5cm_pct), type:'bar', name:'success@5cm', marker:{color:'#a8b8d0'}},
  ];
  Plotly.newPlot('chart-fam-thresh', traces, layout({
    barmode:'group',
    yaxis:{title:'OOD success (%)', rangemode:'tozero'},
  }), PLOT_CFG);

  // End distance over budgets (OOD), by family
  const budgets = [5,20,50,100,200];
  const t2 = FAMS.map(f => ({
    x: budgets, y: budgets.map(b => {
      const r = DATA.overall.find(r => r.family_label===f && r.split==='ood' && r.budget===b);
      return r ? r.end_distance_cm : null;
    }),
    mode:'lines+markers', name:f,
    line:{color: FAM_COLOR[f], width:2.5}, marker:{size:8},
  }));
  Plotly.newPlot('chart-fam-end', t2, layout({
    xaxis:{title:'Budget (demos)', type:'log', tickvals: budgets, ticktext: budgets.map(String)},
    yaxis:{title:'OOD end distance (cm)', rangemode:'tozero'},
  }), PLOT_CFG);
})();

/* ---------- OOD axis breakdown at budget 50 and 200 ---------- */
(function(){
  function build(budget, divId){
    const axes = [...new Set(DATA.ood_axis.filter(r=>r.budget===budget).map(r=>r.axis_label))];
    axes.sort((a,b)=>a.localeCompare(b));
    const traces = FAMS.map(f => ({
      x: axes,
      y: axes.map(a => {
        const r = DATA.ood_axis.find(r => r.family_label===f && r.axis_label===a && r.budget===budget);
        return r ? r.success_at_1cm_pct : null;
      }),
      type:'bar', name:f,
      marker:{color: FAM_COLOR[f]},
    }));
    Plotly.newPlot(divId, traces, layout({
      barmode:'group',
      yaxis:{title:'OOD success@1cm (%)', rangemode:'tozero'},
      xaxis:{title:''},
    }), PLOT_CFG);
  }
  build(50,'chart-axis-50');
  build(200,'chart-axis-200');

  function table(budget, divId){
    const cols = ['Family','Axis','Seeds','succ@1cm','succ@2cm','succ@5cm','nearest (cm)','end (cm)'];
    let html = '<h3 style="margin-top:8px;font-size:13px;color:#6b6b6b">Budget '+budget+' detail</h3>';
    html += '<table class="data"><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
    const rows = [...DATA.ood_axis.filter(r=>r.budget===budget)].sort((a,b)=>{
      const fi = FAMS.indexOf(a.family_label) - FAMS.indexOf(b.family_label);
      if (fi) return fi;
      return a.axis_label.localeCompare(b.axis_label);
    });
    rows.forEach(r => {
      html += '<tr>'
        +'<td class="'+FAM_KEY[r.family_label]+'">'+r.family_label+'</td>'
        +'<td>'+r.axis_label+'</td>'
        +'<td>'+seedTag(r.seed_count)+'</td>'
        +'<td>'+fmtPct(r.success_at_1cm_pct)+'%</td>'
        +'<td>'+fmtPct(r.success_at_2cm_pct)+'%</td>'
        +'<td>'+fmtPct(r.success_at_5cm_pct)+'%</td>'
        +'<td>'+fmtCm(r.nearest_distance_cm)+'</td>'
        +'<td>'+fmtCm(r.end_distance_cm)+'</td>'
        +'</tr>';
    });
    html += '</tbody></table>';
    document.getElementById(divId).innerHTML = html;
  }
  table(50,'table-axis-50');
  table(200,'table-axis-200');
})();

/* ---------- Task basis ---------- */
(function(){
  const tasks = ['Reach','Obstacle-aware reach'];
  const budgets = [50,200];
  const traces = [];
  FAMS.forEach(f => {
    budgets.forEach(b => {
      traces.push({
        x: tasks,
        y: tasks.map(t => {
          const r = DATA.ood_task.find(r => r.family_label===f && r.task_label===t && r.budget===b);
          return r ? r.success_at_1cm_pct : null;
        }),
        type:'bar', name: f+' · b'+b,
        marker:{color: FAM_COLOR[f], opacity: b===200?1:0.55,
                line:{color: FAM_COLOR[f], width: b===50?1.5:0}},
      });
    });
  });
  Plotly.newPlot('chart-task', traces, layout({
    barmode:'group',
    yaxis:{title:'OOD success@1cm (%)', rangemode:'tozero'},
  }), PLOT_CFG);

  const t2 = [];
  FAMS.forEach(f => {
    budgets.forEach(b => {
      t2.push({
        x: tasks,
        y: tasks.map(t => {
          const r = DATA.ood_task.find(r => r.family_label===f && r.task_label===t && r.budget===b);
          return r ? r.end_distance_cm : null;
        }),
        type:'bar', name: f+' · b'+b,
        marker:{color: FAM_COLOR[f], opacity: b===200?1:0.55,
                line:{color: FAM_COLOR[f], width: b===50?1.5:0}},
      });
    });
  });
  Plotly.newPlot('chart-task-end', t2, layout({
    barmode:'group',
    yaxis:{title:'OOD end distance (cm)', rangemode:'tozero'},
  }), PLOT_CFG);

  // Table
  const cols = ['Family','Task','Budget','Seeds','succ@1cm','succ@5cm','nearest (cm)','end (cm)'];
  let html = '<table class="data"><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
  const rows = [...DATA.ood_task.filter(r=>budgets.includes(r.budget))].sort((a,b)=>{
    const fi = FAMS.indexOf(a.family_label) - FAMS.indexOf(b.family_label);
    if (fi) return fi;
    if (a.task_label !== b.task_label) return a.task_label.localeCompare(b.task_label);
    return a.budget - b.budget;
  });
  rows.forEach(r => {
    html += '<tr>'
      +'<td class="'+FAM_KEY[r.family_label]+'">'+r.family_label+'</td>'
      +'<td>'+r.task_label+'</td>'
      +'<td>'+r.budget+'</td>'
      +'<td>'+seedTag(r.seed_count)+'</td>'
      +'<td>'+fmtPct(r.success_at_1cm_pct)+'%</td>'
      +'<td>'+fmtPct(r.success_at_5cm_pct)+'%</td>'
      +'<td>'+fmtCm(r.nearest_distance_cm)+'</td>'
      +'<td>'+fmtCm(r.end_distance_cm)+'</td>'
      +'</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('table-task').innerHTML = html;
})();

/* ---------- Threshold sweep ---------- */
(function(){
  const thresholds = [1,2,5];
  const labels = ['1cm','2cm','5cm'];
  const keys = ['success_at_1cm_pct','success_at_2cm_pct','success_at_5cm_pct'];
  const traces = FAMS.map(f => {
    const r = DATA.overall.find(r => r.family_label===f && r.split==='ood' && r.budget===200);
    return {
      x: labels,
      y: keys.map(k => r ? r[k] : null),
      mode:'lines+markers', name:f,
      line:{color: FAM_COLOR[f], width:2.5}, marker:{size:9},
    };
  });
  Plotly.newPlot('chart-thresh', traces, layout({
    xaxis:{title:'Precision threshold'},
    yaxis:{title:'OOD success (%)', rangemode:'tozero'},
  }), PLOT_CFG);
})();

/* ---------- Diversity pair gains ---------- */
(function(){
  // Show budget 50 (3-seed, the strongest comparison), all families, all (axis, task) pairs.
  const budget = 50;
  const rows = DATA.pair_gains.filter(r => r.budget===budget);
  const labels = rows.map(r => r.axis_label + ' · ' + (r.task_label==='Reach'?'Reach':'Avoid'));
  // Group bars per family using the same x ordering for each family.
  // Build canonical order: by axis then task.
  const canonical = [];
  const seen = new Set();
  rows.forEach(r => {
    const key = r.axis_label+'|'+r.task_label;
    if (!seen.has(key)) { seen.add(key); canonical.push({axis: r.axis_label, task: r.task_label, label: r.axis_label+' · '+(r.task_label==='Reach'?'Reach':'Avoid')}); }
  });
  canonical.sort((a,b)=> a.axis.localeCompare(b.axis) || a.task.localeCompare(b.task));

  const traces = FAMS.map(f => ({
    x: canonical.map(c => c.label),
    y: canonical.map(c => {
      const r = DATA.pair_gains.find(r => r.family_label===f && r.axis_label===c.axis && r.task_label===c.task && r.budget===budget);
      return r ? r.gain_success_at_1cm_pct : null;
    }),
    type:'bar', name:f, marker:{color: FAM_COLOR[f]},
  }));
  Plotly.newPlot('chart-pairs', traces, layout({
    barmode:'group',
    xaxis:{title:'Axis · Task', tickangle: -25},
    yaxis:{title:'Δ OOD success@1cm (diverse − narrow), pp', zeroline: true, zerolinecolor:'#999'},
  }), PLOT_CFG);

  // Table
  const cols = ['Family','Axis','Task','Budget','Narrow','Diverse','Δ pp @1cm','Δ pp @5cm','Δ nearest (cm)'];
  let html = '<table class="data"><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>';
  const sorted = [...DATA.pair_gains].sort((a,b)=>{
    const fi = FAMS.indexOf(a.family_label) - FAMS.indexOf(b.family_label);
    if (fi) return fi;
    if (a.budget !== b.budget) return a.budget - b.budget;
    if (a.axis_label !== b.axis_label) return a.axis_label.localeCompare(b.axis_label);
    return a.task_label.localeCompare(b.task_label);
  });
  sorted.forEach(r => {
    const sgn = (v) => {
      if (v == null) return '';
      const s = (Math.round(v*10)/10).toFixed(1);
      const cls = v > 0.5 ? 'style="color:#2f7a4d;font-weight:600"' : v < -0.5 ? 'style="color:#b8332a;font-weight:600"' : '';
      return '<span '+cls+'>'+(v>0?'+':'')+s+'</span>';
    };
    html += '<tr>'
      +'<td class="'+FAM_KEY[r.family_label]+'">'+r.family_label+'</td>'
      +'<td>'+r.axis_label+'</td>'
      +'<td>'+r.task_label+'</td>'
      +'<td>'+r.budget+'</td>'
      +'<td>'+fmtPct(r.baseline_success_at_1cm_pct)+'%</td>'
      +'<td>'+fmtPct(r.diverse_success_at_1cm_pct)+'%</td>'
      +'<td>'+sgn(r.gain_success_at_1cm_pct)+'</td>'
      +'<td>'+sgn(r.gain_success_at_5cm_pct)+'</td>'
      +'<td>'+sgn(r.gain_nearest_distance_cm)+'</td>'
      +'</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('table-pairs').innerHTML = html;
})();
</script>
</body>
</html>
"""


def main():
    data = collect()
    payload = json.dumps(data, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__DATA__", payload)
    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    print(f"  overall: {len(data['overall'])}, ood_axis: {len(data['ood_axis'])}, "
          f"pair_gains: {len(data['pair_gains'])}")


if __name__ == "__main__":
    main()
