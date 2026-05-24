#!/usr/bin/env python3
"""Build final HTML analysis pages for the HW8 visual BC study."""

from __future__ import annotations

import csv
import html
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"

PAGES = [
    ("index.html", "Overview"),
    ("01_proposal_to_story.html", "Proposal to Story"),
    ("02_experiment_design.html", "Experiment Design"),
    ("03_visual_diversity_results.html", "Visual Diversity"),
    ("04_model_comparison.html", "Model Comparison"),
    ("05_structured_diagnostic.html", "Structured Diagnostic"),
    ("06_paper_story.html", "Paper Story"),
    ("07_video_rubric_plan.html", "Video Rubric Plan"),
    ("08_video_slides.html", "Video Slides"),
    ("09_video_case_studies.html", "Video Case Studies"),
    ("10_rubric_execution_plan.html", "Rubric Execution"),
]


def load_csv(relative_path: str) -> list[dict[str, str]]:
    path = ROOT / relative_path
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def fmt_pct(value: str | float, already_pct: bool = True) -> str:
    val = fnum(value)
    if not already_pct:
        val *= 100.0
    return f"{val:.1f}%"


def fmt_num(value: str | float, digits: int = 1) -> str:
    return f"{fnum(value):.{digits}f}"


def fmt_int(value: str | float) -> str:
    return f"{int(round(fnum(value))):,}"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def attrs(**kwargs: str | None) -> str:
    parts = []
    for key, value in kwargs.items():
        if value is None:
            continue
        parts.append(f'{key.replace("_", "-")}="{esc(value)}"')
    return " ".join(parts)


def nav(current: str) -> str:
    links = []
    for file_name, label in PAGES:
        cls = "active" if file_name == current else ""
        links.append(f'<a class="{cls}" href="{file_name}">{esc(label)}</a>')
    return '<nav class="topnav">' + "\n".join(links) + "</nav>"


def page(file_name: str, title: str, subtitle: str, body: str) -> None:
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #18202b;
      --muted: #5e6877;
      --line: #d9dee7;
      --accent: #156b75;
      --accent-2: #7a4b00;
      --good: #137548;
      --bad: #a13232;
      --code: #eef2f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.52;
      font-size: 16px;
    }}
    .shell {{
      width: min(1180px, calc(100% - 36px));
      margin: 0 auto;
      padding: 26px 0 56px;
    }}
    .topnav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 26px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--line);
    }}
    .topnav a {{
      color: var(--ink);
      text-decoration: none;
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 7px 10px;
      border-radius: 7px;
      font-size: 14px;
    }}
    .topnav a.active {{
      border-color: var(--accent);
      color: #ffffff;
      background: var(--accent);
    }}
    header {{
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(32px, 5vw, 54px);
      line-height: 1.02;
      letter-spacing: 0;
      max-width: 980px;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 18px;
      max-width: 920px;
    }}
    h2 {{
      margin: 34px 0 12px;
      font-size: 25px;
      line-height: 1.18;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 22px 0 8px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 12px;
      max-width: 980px;
    }}
    a {{ color: var(--accent); }}
    code {{
      background: var(--code);
      padding: 2px 5px;
      border-radius: 5px;
      font-size: 0.92em;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 14px;
      margin: 16px 0 24px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .card h3 {{ margin-top: 0; }}
    .callout {{
      background: #f1f8f8;
      border-left: 4px solid var(--accent);
      padding: 14px 16px;
      margin: 18px 0;
      max-width: 1000px;
    }}
    .warning {{
      background: #fff7ed;
      border-left: 4px solid var(--accent-2);
      padding: 14px 16px;
      margin: 18px 0;
      max-width: 1000px;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 14px 0 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 760px;
      font-size: 14px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef2f7;
      font-weight: 700;
      white-space: nowrap;
    }}
    td.num, th.num {{ text-align: right; }}
    tr:last-child td {{ border-bottom: 0; }}
    .figure {{
      margin: 18px 0 28px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .figure img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 4px;
      background: #ffffff;
    }}
    .figure figcaption {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 8px;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 14px;
      margin: 16px 0 24px;
    }}
    video {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #000000;
    }}
    ul, ol {{ max-width: 980px; }}
    li {{ margin: 5px 0; }}
    .metric {{
      font-weight: 700;
      color: var(--accent);
    }}
    .small {{
      color: var(--muted);
      font-size: 14px;
    }}
    .slide-jump {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 16px 0 24px;
    }}
    .slide-jump a {{
      text-decoration: none;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 13px;
    }}
    .deck {{
      display: grid;
      gap: 28px;
    }}
    .slide {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .slide-screen {{
      min-height: 560px;
      padding: 34px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 20px;
      background: #ffffff;
    }}
    .slide-kicker {{
      color: var(--accent);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 13px;
    }}
    .slide-title {{
      font-size: clamp(30px, 4.5vw, 48px);
      line-height: 1.04;
      margin: 0;
      max-width: 980px;
    }}
    .slide-body {{
      align-self: center;
      display: grid;
      gap: 16px;
    }}
    .slide-body.two-col {{
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      align-items: center;
    }}
    .slide-body img,
    .slide-body video {{
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }}
    .big-points {{
      display: grid;
      gap: 12px;
      margin: 0;
      padding: 0;
      list-style: none;
      font-size: 24px;
      line-height: 1.24;
    }}
    .big-points li {{
      margin: 0;
      padding: 14px 16px;
      background: #f6f8fb;
      border-left: 4px solid var(--accent);
      border-radius: 7px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .metric-box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #f6f8fb;
    }}
    .metric-box strong {{
      display: block;
      font-size: 30px;
      color: var(--accent);
      line-height: 1.05;
    }}
    .metric-box span {{
      color: var(--muted);
      font-size: 14px;
    }}
    .equation {{
      font-size: 23px;
      background: #f6f8fb;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
    }}
    .transcript {{
      border-top: 1px solid var(--line);
      background: #fbfcfe;
      padding: 18px 22px;
    }}
    .transcript h3 {{
      margin-top: 0;
      color: var(--accent);
    }}
    .transcript p {{
      max-width: none;
    }}
    .slide-footer {{
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 13px;
    }}
    footer {{
      margin-top: 36px;
      color: var(--muted);
      font-size: 13px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }}
    @media (max-width: 760px) {{
      .slide-screen {{
        min-height: auto;
        padding: 22px;
      }}
      .slide-body.two-col {{
        grid-template-columns: 1fr;
      }}
      .big-points {{
        font-size: 19px;
      }}
      .equation {{
        font-size: 17px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    {nav(file_name)}
    <header>
      <h1>{esc(title)}</h1>
      <p class="subtitle">{esc(subtitle)}</p>
    </header>
    {body}
    <footer>
      Source artifacts are local to <code>research2/results</code>. Generated by <code>research2/code/build_html_analysis.py</code>.
    </footer>
  </main>
</body>
</html>
"""
    (OUT / file_name).write_text(html_text)


def table(
    rows: Iterable[dict[str, object]],
    columns: list[tuple[str, str, Callable[[object], str] | None, bool]],
) -> str:
    row_list = list(rows)
    head = "".join(
        f'<th class="{"num" if is_num else ""}">{esc(label)}</th>'
        for label, _, _, is_num in columns
    )
    body_rows = []
    for row in row_list:
        cells = []
        for _, key, formatter, is_num in columns:
            value = row.get(key, "")
            rendered = formatter(value) if formatter else esc(value)
            cells.append(f'<td class="{"num" if is_num else ""}">{rendered}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return '<div class="table-wrap"><table><thead><tr>' + head + "</tr></thead><tbody>" + "\n".join(body_rows) + "</tbody></table></div>"


def fig(src: str, caption: str) -> str:
    return f"""<figure class="figure">
  <img src="{esc(src)}" alt="{esc(caption)}">
  <figcaption>{esc(caption)}</figcaption>
</figure>"""


def video(src: str, caption: str) -> str:
    return f"""<div class="card">
  <video controls preload="metadata" src="{esc(src)}"></video>
  <p class="small">{esc(caption)}</p>
</div>"""


def rows_where(rows: list[dict[str, str]], **criteria: object) -> list[dict[str, str]]:
    out = []
    for row in rows:
        ok = True
        for key, value in criteria.items():
            if isinstance(value, (set, tuple, list)):
                if row.get(key) not in {str(v) for v in value}:
                    ok = False
                    break
            elif row.get(key) != str(value):
                ok = False
                break
        if ok:
            out.append(row)
    return out


def sort_by(rows: list[dict[str, str]], *keys: str) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: tuple(row.get(key, "") for key in keys))


def metric_cols(percent_prefix: str = "success_at") -> list[tuple[str, str, Callable[[object], str] | None, bool]]:
    return [
        ("S@1cm", f"{percent_prefix}_1cm_pct", lambda v: fmt_pct(v, True), True),
        ("S@2cm", f"{percent_prefix}_2cm_pct", lambda v: fmt_pct(v, True), True),
        ("S@5cm", f"{percent_prefix}_5cm_pct", lambda v: fmt_pct(v, True), True),
        ("Best cm", "nearest_distance_cm", lambda v: fmt_num(v, 1), True),
        ("Final cm", "end_distance_cm", lambda v: fmt_num(v, 1), True),
    ]


def structured_metric_cols() -> list[tuple[str, str, Callable[[object], str] | None, bool]]:
    return [
        ("S@1cm", "success_1cm", lambda v: fmt_pct(v, False), True),
        ("S@2cm", "success_2cm", lambda v: fmt_pct(v, False), True),
        ("S@5cm", "success_5cm", lambda v: fmt_pct(v, False), True),
        ("Best cm", "mean_best_cm", lambda v: fmt_num(v, 1), True),
        ("Final cm", "mean_final_cm", lambda v: fmt_num(v, 1), True),
    ]


def make_overview() -> str:
    return """
<div class="callout">
  <p><strong>Final thesis.</strong> The original project asked which visual diversity axis most improves closed-loop pixel BC. The result is sharper: visual diversity helps some appearance and viewpoint shifts, but the hard obstacle-aware spatial setting exposes a hidden phase and geometry bottleneck. Explicit phase+geometry conditioning turns that failure into a strong diagnostic result.</p>
</div>

<div class="grid">
  <section class="card">
    <h3>Original commitment</h3>
    <p>Controlled PyBullet Panda imitation learning with fixed data budgets, dual RGB observations, two tasks, and single-factor visual shifts.</p>
  </section>
  <section class="card">
    <h3>Main empirical pattern</h3>
    <p>Scratch CNNs handle color, camera, and lighting better than expected, but spatial OOD and obstacle-aware routing remain brittle.</p>
  </section>
  <section class="card">
    <h3>Final contribution</h3>
    <p>A phase- and geometry-conditioned diagnostic shows that the obstacle-aware task is not merely a visual diversity problem.</p>
  </section>
</div>

<h2>Core Claims</h2>
<ol>
  <li><strong>Spatial distribution is the hardest visual diversity axis.</strong> At budget 50, Scratch CNN OOD S@1cm is 57.9% for camera, 41.1% for color, 20.6% for lighting, but only 2.9% for spatial distribution.</li>
  <li><strong>Loose thresholds hide precision failures.</strong> S@5cm often looks acceptable while S@1cm and final distance show unstable closed-loop behavior.</li>
  <li><strong>Pretrained encoders are mixed in simulation.</strong> Frozen ResNet-18 can improve some loose success rates, but it does not consistently beat scratch CNN precision. Partial fine-tuning is unstable.</li>
  <li><strong>Obstacle-aware reaching needs explicit phase/geometry diagnostics.</strong> On the edge-balanced obstacle-aware setting, visual-only scratch reaches only 1.3% OOD S@1cm, while phase+geometry scratch reaches 53.3% OOD S@1cm.</li>
</ol>

<h2>Report Map</h2>
<div class="grid">
  <a class="card" href="01_proposal_to_story.html"><h3>Proposal to Story</h3><p>How the final paper evolved from the initial proposal.</p></a>
  <a class="card" href="02_experiment_design.html"><h3>Experiment Design</h3><p>Tasks, axes, budgets, models, datasets, and metrics.</p></a>
  <a class="card" href="03_visual_diversity_results.html"><h3>Visual Diversity</h3><p>Budget curves, axis comparisons, task split, and diversity gains.</p></a>
  <a class="card" href="04_model_comparison.html"><h3>Model Comparison</h3><p>Scratch CNN vs frozen ResNet-18 vs partial ResNet-18.</p></a>
  <a class="card" href="05_structured_diagnostic.html"><h3>Structured Diagnostic</h3><p>Phase+geometry results, ablations, bucket breakdown, and videos.</p></a>
  <a class="card" href="06_paper_story.html"><h3>Paper Story</h3><p>Claims, paper structure, limitations, and presentation plan.</p></a>
  <a class="card" href="07_video_rubric_plan.html"><h3>Video Rubric Plan</h3><p>10-minute submission outline mapped directly to the project-video rubric.</p></a>
  <a class="card" href="08_video_slides.html"><h3>Video Slides</h3><p>HTML-only slide deck with speaker transcripts for each slide.</p></a>
  <a class="card" href="09_video_case_studies.html"><h3>Video Case Studies</h3><p>Best/worst rollout clip library for visual axes, tasks, and improvements.</p></a>
  <a class="card" href="10_rubric_execution_plan.html"><h3>Rubric Execution</h3><p>Point-by-point plan for satisfying the project video rubric.</p></a>
</div>
"""


def make_proposal_page() -> str:
    rows = [
        {
            "proposal": "Which visual diversity factors matter under a fixed demo budget?",
            "final": "Still the first question. The clearest ranking is that spatial distribution is the hardest axis, while color/camera/lighting are easier for scratch CNNs.",
        },
        {
            "proposal": "Compare scratch CNN with stronger visual representations.",
            "final": "Completed with scratch CNN, frozen ResNet-18, and partially fine-tuned ResNet-18. Pretraining was not a simple win in simulation.",
        },
        {
            "proposal": "Use closed-loop rollout metrics instead of only supervised loss.",
            "final": "This became central. Success@0.5/1/2/5cm, best distance, and final distance reveal failures that loose S@5cm hides.",
        },
        {
            "proposal": "Study reach and obstacle-aware reach.",
            "final": "Obstacle-aware reach became the scientific center because it exposes hidden stage and geometry inference failures.",
        },
        {
            "proposal": "Optional extension if time remains.",
            "final": "The extension became a structured diagnostic: phase+geometry-conditioned visual BC to test what image-only BC failed to infer.",
        },
    ]
    return f"""
<p>The initial proposal was titled <em>What Visual Diversity Matters for Closed-Loop Behavior Cloning?</em> It proposed a controlled PyBullet Panda benchmark for pixel-based BC under distribution shift. The proposal authors are Aninditaa Chauhan, Jai Kumar Sharma, Manas Ganti, and Shakir Farhan Mohammed.</p>

<h2>What Stayed the Same</h2>
{table(rows, [
    ("Proposal commitment", "proposal", None, False),
    ("Final status", "final", None, False),
])}

<h2>Final Research Questions</h2>
<ol>
  <li><strong>RQ1.</strong> Which visual diversity axes matter most for OOD generalization?</li>
  <li><strong>RQ2.</strong> Do pretrained visual encoders help in simulation, or does sim-real mismatch make frozen features weak?</li>
  <li><strong>RQ3.</strong> Does partial fine-tuning improve over frozen encoders and scratch CNNs?</li>
  <li><strong>RQ4.</strong> Are loose success thresholds hiding poor precision and unstable final behavior?</li>
</ol>

<h2>How the Story Changed</h2>
<p>The proposal expected camera pose and spatial distribution to dominate superficial appearance shifts. The data mostly supports that, but with a stronger twist: spatial OOD is not just another visual axis. In the obstacle-aware task it couples target location, obstacle relation, and task stage. That makes pure visual BC fail even after edge-balanced data collection.</p>
<p>The final story should therefore be written as a controlled empirical study plus a diagnostic extension. The contribution is not that a privileged structured model is the deployable answer. The contribution is that the structured model isolates the missing information: obstacle-aware visual BC is failing to infer phase and geometry reliably enough for precise closed-loop control.</p>

<div class="warning">
  <p><strong>Important framing.</strong> The phase+geometry model is not pure pixel BC. It is a diagnostic that explains why the pixel-only systems fail and what information would need to be inferred or represented by a stronger policy.</p>
</div>
"""


def make_design_page() -> str:
    axes = [
        {"axis": "Color", "narrow": "Fixed cube color, primarily red", "diverse": "Multiple cube colors", "ood": "Held-out color conditions"},
        {"axis": "Spatial distribution", "narrow": "Restricted target XY region", "diverse": "Wider target XY support", "ood": "Held-out spatial edge/corner regions"},
        {"axis": "Camera location / viewpoint", "narrow": "Fixed external camera pose", "diverse": "Multiple external camera poses", "ood": "Held-out camera pose"},
        {"axis": "Lighting direction + intensity", "narrow": "Default/fixed lighting", "diverse": "Explicit light direction and intensity variation", "ood": "Held-out lighting conditions"},
    ]
    models = [
        {"model": "Scratch CNN BC", "role": "Main pixel baseline trained end-to-end from 128px RGB observations plus robot state."},
        {"model": "Frozen ResNet-18 BC", "role": "Pretrained encoder frozen; BC head learns from visual features. Tests whether generic visual features help in sim."},
        {"model": "Partial ResNet-18 BC", "role": "Earlier ResNet layers frozen, later layers and policy head trainable. Tests whether adaptation improves over frozen features."},
        {"model": "Phase+geometry BC", "role": "Diagnostic model with RGB, robot state, phase one-hot, target geometry, obstacle geometry, and relative deltas."},
    ]
    metrics = [
        {"metric": "S@0.5cm, S@1cm, S@2cm, S@5cm", "meaning": "Success thresholds computed from the closest distance reached during the rollout."},
        {"metric": "Best distance", "meaning": "Closest end-effector-to-target distance ever reached. This measures whether the policy can get near the goal at all."},
        {"metric": "Final distance", "meaning": "Distance at rollout end. This measures whether the policy stabilizes or drifts away after getting close."},
        {"metric": "1mm stop threshold", "meaning": "Rollouts only terminate early at 0.001 m, so S@5cm does not clip final distance near 5 cm."},
    ]
    return f"""
<h2>Tasks</h2>
<div class="grid">
  <section class="card">
    <h3>Reach</h3>
    <p>The robot moves the end effector to a cube target. This task isolates visual target localization and closed-loop servoing.</p>
  </section>
  <section class="card">
    <h3>Obstacle-aware reach</h3>
    <p>The robot must reach the same target while routing around a blocking obstacle. This introduces stage structure: hover, side alignment, and final descent.</p>
  </section>
</div>

<h2>Visual Diversity Axes</h2>
{table(axes, [
    ("Axis", "axis", None, False),
    ("Narrow condition", "narrow", None, False),
    ("Diverse condition", "diverse", None, False),
    ("OOD evaluation", "ood", None, False),
])}

<h2>Dataset Structure</h2>
<p>Datasets are 128px behavior-cloning demonstrations collected from scripted experts. Each demonstration contributes image/action samples along the trajectory, not only the start and end states. The primary observations are external RGB and wrist/end-effector RGB, with robot state included in the policy input.</p>
<p>Budgets 5, 20, and 50 use seeds 0, 1, and 2. Budgets 100 and 200 use seed 0 only, so high-budget trends are directional rather than full multi-seed estimates.</p>
{fig("../results/previews_128px_v1/preview_contact_sheet_seed000.png", "Example 128px dataset previews across task and visual-axis conditions.")}

<h2>Model Families</h2>
{table(models, [
    ("Model family", "model", None, False),
    ("Purpose in the study", "role", None, False),
])}

<h2>Evaluation Metrics</h2>
{table(metrics, [
    ("Metric", "metric", None, False),
    ("Interpretation", "meaning", None, False),
])}
"""


def make_visual_results_page(data: dict[str, list[dict[str, str]]]) -> str:
    overall = data["overall"]
    axis = data["ood_axis"]
    task = data["ood_task"]
    gains = data["diversity_gains"]

    scratch_curve = sort_by(rows_where(overall, family_label="Scratch CNN"), "split", "budget")
    scratch_curve = sorted(scratch_curve, key=lambda r: (r["split"], int(r["budget"])))

    axis_50 = sorted(
        rows_where(axis, budget="50"),
        key=lambda r: (r["family_label"], r["axis_label"]),
    )
    axis_200 = sorted(
        rows_where(axis, budget="200"),
        key=lambda r: (r["family_label"], r["axis_label"]),
    )
    scratch_task = sorted(
        rows_where(task, family_label="Scratch CNN"),
        key=lambda r: (int(r["budget"]), r["task_label"]),
    )
    scratch_gains_50 = sorted(
        rows_where(gains, family_label="Scratch CNN", budget="50"),
        key=lambda r: (r["task_label"], r["axis_label"]),
    )
    return f"""
<div class="callout">
  <p><strong>Visual-only takeaway.</strong> Spatial distribution is the hard axis. Color, camera, and lighting shifts can be learned by scratch CNNs in many settings, but spatial OOD collapses precision, especially for obstacle-aware reaching.</p>
</div>

<h2>Scratch CNN Budget Scaling</h2>
<p>Scratch CNN improves with budget, but the improvement is not enough to remove closed-loop instability. The gap between best distance and final distance shows that many policies reach near the target and then drift or oscillate.</p>
{table(scratch_curve, [
    ("Split", "split", lambda v: esc(str(v).upper()), False),
    ("Budget", "budget", lambda v: fmt_int(v), True),
    ("Rollouts", "rollout_count", lambda v: fmt_int(v), True),
] + metric_cols())}

<h2>OOD Axis Ranking at Budget 50</h2>
<p>Budget 50 is the clean multi-seed comparison. Scratch CNN has strong camera and color OOD precision, weaker lighting precision, and a severe spatial collapse.</p>
{table(axis_50, [
    ("Model", "family_label", None, False),
    ("Axis", "axis_label", None, False),
] + metric_cols())}

<h2>OOD Axis Ranking at Budget 200</h2>
<p>Budget 200 is seed-0 only, but it shows the same major pattern: scratch improves on appearance and lighting, while spatial remains hard.</p>
{table(axis_200, [
    ("Model", "family_label", None, False),
    ("Axis", "axis_label", None, False),
] + metric_cols())}

<h2>Task Basis for Scratch CNN</h2>
<p>The direct reach task is much easier than obstacle-aware reach. The obstacle-aware setting is where spatial coverage and hidden task stage matter most.</p>
{table(scratch_task, [
    ("Budget", "budget", lambda v: fmt_int(v), True),
    ("Task", "task_label", None, False),
    ("Rollouts", "rollout_count", lambda v: fmt_int(v), True),
] + metric_cols())}

<h2>Diversity Pair Gains at Budget 50</h2>
<p>These rows compare a narrow training condition against its diverse counterpart under the same task, model, and budget. Positive S@1cm gain means the diverse condition improved precision.</p>
{table(scratch_gains_50, [
    ("Task", "task_label", None, False),
    ("Axis", "axis_label", None, False),
    ("Baseline", "baseline_config", None, False),
    ("Diverse", "diverse_config", None, False),
    ("Gain S@1cm", "gain_success_at_1cm_pct", lambda v: fmt_pct(v, True), True),
    ("Gain S@5cm", "gain_success_at_5cm_pct", lambda v: fmt_pct(v, True), True),
    ("Gain best cm", "gain_nearest_distance_cm", lambda v: fmt_num(v, 2), True),
])}
"""


def make_model_page(data: dict[str, list[dict[str, str]]]) -> str:
    overall = data["overall"]
    ood = rows_where(overall, split="ood")
    ood = sorted(ood, key=lambda r: (r["family_label"], int(r["budget"])))
    budget_50_200 = sorted(
        [r for r in overall if r["budget"] in {"50", "200"}],
        key=lambda r: (int(r["budget"]), r["split"], r["family_label"]),
    )
    return f"""
<div class="callout">
  <p><strong>Encoder takeaway.</strong> Pretrained ResNet features are not automatically better in this simulated manipulation setting. Scratch CNN is often the strongest precision model; frozen ResNet is sometimes stronger at loose thresholds; partial fine-tuning is inconsistent and can be unstable.</p>
</div>

<h2>Overall Budget 50 and 200 Comparison</h2>
<p>Budget 50 is multi-seed. Budget 200 is seed-0 only, but it is useful for seeing whether more demonstrations rescue each model family.</p>
{table(budget_50_200, [
    ("Budget", "budget", lambda v: fmt_int(v), True),
    ("Split", "split", lambda v: esc(str(v).upper()), False),
    ("Model", "family_label", None, False),
    ("Rollouts", "rollout_count", lambda v: fmt_int(v), True),
] + metric_cols())}

<h2>OOD Budget Curves by Model</h2>
<p>The scratch curve improves the most at high budget, but final distance remains large. Frozen ResNet has reasonable loose success but lower precision. Partial ResNet does not provide a stable monotonic improvement.</p>
{table(ood, [
    ("Model", "family_label", None, False),
    ("Budget", "budget", lambda v: fmt_int(v), True),
    ("Rollouts", "rollout_count", lambda v: fmt_int(v), True),
] + metric_cols())}

<h2>Spatial Performance Snapshot</h2>
{fig("../results/analysis_spatial_performance/spatial_performance_table_phase_precise_budget200.png", "Budget 200 spatial performance comparison across Scratch CNN, Frozen ResNet-18, and Partial ResNet-18.")}

<h2>Interpretation</h2>
<ul>
  <li><strong>Scratch CNN.</strong> Best overall precision in many visual-only settings and strongest high-budget OOD curve, but spatial obstacle-aware behavior remains weak.</li>
  <li><strong>Frozen ResNet-18.</strong> Helps some loose success rates and is less catastrophic than partial fine-tuning in the edge-balanced visual-only setting, but precision remains limited.</li>
  <li><strong>Partial ResNet-18.</strong> The most unstable family. In this simulated setting, fine-tuning later blocks did not reliably solve the mismatch between visual features and control-relevant geometry.</li>
</ul>
"""


def aggregate_bucket_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], dict[str, float | str]] = {}
    for row in rows:
        key = (row["group"], row["model"], row["split"], row["bucket_group"])
        if key not in grouped:
            grouped[key] = {
                "group": row["group"],
                "model": row["model"],
                "split": row["split"],
                "bucket_group": row["bucket_group"],
                "rollouts": 0.0,
                "success_1cm_count": 0.0,
                "success_5cm_count": 0.0,
                "best_sum": 0.0,
                "final_sum": 0.0,
            }
        acc = grouped[key]
        n = fnum(row["rollouts"])
        acc["rollouts"] = fnum(acc["rollouts"]) + n
        acc["success_1cm_count"] = fnum(acc["success_1cm_count"]) + fnum(row["success_1cm"]) * n
        acc["success_5cm_count"] = fnum(acc["success_5cm_count"]) + fnum(row["success_5cm"]) * n
        acc["best_sum"] = fnum(acc["best_sum"]) + fnum(row["mean_best_cm"]) * n
        acc["final_sum"] = fnum(acc["final_sum"]) + fnum(row["mean_final_cm"]) * n

    out = []
    for acc in grouped.values():
        n = max(fnum(acc["rollouts"]), 1.0)
        out.append(
            {
                "group": acc["group"],
                "model": acc["model"],
                "split": acc["split"],
                "bucket_group": acc["bucket_group"],
                "rollouts": n,
                "success_1cm": fnum(acc["success_1cm_count"]) / n,
                "success_5cm": fnum(acc["success_5cm_count"]) / n,
                "mean_best_cm": fnum(acc["best_sum"]) / n,
                "mean_final_cm": fnum(acc["final_sum"]) / n,
            }
        )
    return sorted(out, key=lambda r: (str(r["split"]), str(r["group"]), str(r["bucket_group"])))


def make_structured_page(data: dict[str, list[dict[str, str]]]) -> str:
    main = data["main_comparison"]
    ablation = data["ablation"]
    bucket = data["bucket"]

    key_main = [
        r
        for r in main
        if r["group"] in {"Edge-balanced visual-only", "Edge-balanced phase+geometry"}
    ]
    key_main = sorted(key_main, key=lambda r: (r["group"], r["model"], r["split"]))

    ablation_rows = sorted(ablation, key=lambda r: (r["model"], r["split"]))
    bucket_summary = [
        r
        for r in aggregate_bucket_rows(bucket)
        if r["split"] == "OOD"
        and r["model"] == "Scratch CNN"
        and r["group"] in {"Edge-balanced visual-only", "Edge-balanced phase+geometry"}
    ]
    return f"""
<div class="callout">
  <p><strong>Main diagnostic result.</strong> Edge-balanced visual-only training did not solve obstacle-aware spatial OOD. Adding explicit phase and geometry to the visual policy increased Scratch CNN OOD S@1cm from 1.3% to 53.3% and reduced final distance from 19.25 cm to 4.23 cm.</p>
</div>

<h2>Main Edge-Balanced Comparison</h2>
{table(key_main, [
    ("Setup", "group", None, False),
    ("Model", "model", None, False),
    ("Split", "split", None, False),
    ("Rollouts", "rollouts", lambda v: fmt_int(v), True),
] + structured_metric_cols())}
{fig("../results/structured_analysis/main_comparison.png", "Main comparison: visual-only edge-balanced training vs phase+geometry edge-balanced training and prior spatial-wide baselines.")}

<h2>Scratch Structured Ablations</h2>
<p>Phase alone fails because it does not locate the target or obstacle. Geometry alone gives strong loose success and final stability. Phase+geometry gives the best 1cm precision, but with a more bimodal failure pattern.</p>
{table(ablation_rows, [
    ("Variant", "model", None, False),
    ("Split", "split", None, False),
    ("Rollouts", "rollouts", lambda v: fmt_int(v), True),
] + structured_metric_cols())}
{fig("../results/structured_analysis/scratch_structured_ablation.png", "Scratch ablation: phase-only, target-only, full geometry-only, and phase+geometry.")}

<h2>Spatial Bucket Breakdown</h2>
<p>The bucket analysis shows the diagnostic value of structured conditioning most clearly. Visual-only scratch fails badly on OOD corners and edges. Structured scratch solves OOD corners and improves edges, but edge/interior cases still contain failures.</p>
{table(bucket_summary, [
    ("Setup", "group", None, False),
    ("Bucket", "bucket_group", None, False),
    ("Rollouts", "rollouts", lambda v: fmt_int(v), True),
    ("S@1cm", "success_1cm", lambda v: fmt_pct(v, False), True),
    ("S@5cm", "success_5cm", lambda v: fmt_pct(v, False), True),
    ("Best cm", "mean_best_cm", lambda v: fmt_num(v, 1), True),
    ("Final cm", "mean_final_cm", lambda v: fmt_num(v, 1), True),
])}
{fig("../results/structured_analysis/spatial_bucket_groups.png", "Spatial bucket breakdown by corner, edge, and interior target groups.")}

<h2>Representative Videos</h2>
<div class="video-grid">
  {video("../results/structured_analysis/videos/visual_only_scratch_ood_failure.mp4", "Visual-only Scratch CNN OOD failure.")}
  {video("../results/structured_analysis/videos/structured_scratch_ood_success.mp4", "Phase+geometry Scratch CNN OOD success.")}
  {video("../results/structured_analysis/videos/structured_scratch_ood_failure.mp4", "Phase+geometry Scratch CNN OOD failure case.")}
  {video("../results/structured_analysis/videos/frozen_structured_ood_comparison.mp4", "Frozen structured comparison rollout.")}
</div>

<h2>What This Proves</h2>
<p>This does not prove that privileged state should be used at deployment. It proves that the visual-only obstacle-aware policies were missing control-relevant phase and geometry information. That is exactly the diagnostic story: visual diversity alone was insufficient because the task required reliable inference of where the robot is in the route and how the obstacle geometry relates to the target.</p>
"""


def make_paper_story_page() -> str:
    claims = [
        {"claim": "Controlled visual diversity matters, but not equally.", "evidence": "Scratch CNN OOD at budget 50: camera 57.9% S@1cm, color 41.1%, lighting 20.6%, spatial 2.9%."},
        {"claim": "Pretrained encoders are not a guaranteed solution in simulation.", "evidence": "Frozen and partial ResNet-18 do not consistently beat scratch CNN precision; partial fine-tuning is often unstable."},
        {"claim": "Obstacle-aware spatial OOD is a phase/geometry problem, not only an image augmentation problem.", "evidence": "Edge-balanced visual-only scratch OOD S@1cm is 1.3%; phase+geometry scratch OOD S@1cm is 53.3%."},
        {"claim": "Precision metrics change the conclusion.", "evidence": "S@5cm can look acceptable while S@1cm and final distance reveal drift, oscillation, or unstable stopping."},
    ]
    figures = [
        {"slot": "Figure 1", "content": "Benchmark diagram: two tasks, four diversity axes, ID/OOD splits, 128px dual RGB observations."},
        {"slot": "Figure 2", "content": "OOD axis ranking by model at budget 50 and 200."},
        {"slot": "Figure 3", "content": "Budget scaling for Scratch CNN, Frozen ResNet-18, and Partial ResNet-18."},
        {"slot": "Figure 4", "content": "Structured diagnostic main comparison and ablation."},
        {"slot": "Figure 5", "content": "Spatial bucket breakdown plus representative rollout frames or video stills."},
    ]
    limitations = [
        {"limitation": "Simulation-only images", "handling": "Frame as a controlled diagnostic study, not a direct sim-to-real claim."},
        {"limitation": "High budgets use one seed", "handling": "Treat budget 100/200 as directional; rely on budget 50 for full multi-seed visual-axis conclusions."},
        {"limitation": "Structured phase+geometry uses privileged information", "handling": "Call it a diagnostic extension that identifies missing information, not a deployment baseline."},
        {"limitation": "Obstacle-aware task is still simple manipulation", "handling": "Emphasize controlled interpretability over task-suite breadth."},
    ]
    return f"""
<h2>Recommended One-Sentence Story</h2>
<div class="callout">
  <p>In controlled pixel-based robot BC, appearance and viewpoint diversity are not the main obstacle; spatial obstacle-aware generalization fails because image-only policies struggle to infer task phase and geometry, and multi-threshold closed-loop evaluation exposes that failure.</p>
</div>

<h2>Defensible Claims</h2>
{table(claims, [
    ("Claim", "claim", None, False),
    ("Evidence", "evidence", None, False),
])}

<h2>Suggested Paper Structure</h2>
<ol>
  <li><strong>Introduction.</strong> Start from the practical data-collection question: when demos are expensive, which visual diversity is worth collecting?</li>
  <li><strong>Benchmark.</strong> Present the controlled PyBullet setup, reach and obstacle-aware reach, four diversity axes, budgets, seeds, and ID/OOD splits.</li>
  <li><strong>Visual-only results.</strong> Show that spatial distribution dominates the failure modes, while color/camera/lighting are easier for scratch CNNs.</li>
  <li><strong>Representation results.</strong> Show that frozen and partial ResNet-18 do not reliably solve the simulated control problem.</li>
  <li><strong>Diagnostic extension.</strong> Introduce phase+geometry-conditioned visual BC to test what image-only policies fail to infer.</li>
  <li><strong>Discussion.</strong> Argue that future policies need better spatial/temporal/geometric representations, not only more appearance diversity.</li>
</ol>

<h2>Figure Plan</h2>
{table(figures, [
    ("Slot", "slot", None, False),
    ("Content", "content", None, False),
])}

<h2>Limitations to State Explicitly</h2>
{table(limitations, [
    ("Limitation", "limitation", None, False),
    ("How to handle it", "handling", None, False),
])}

<h2>Suggested Abstract Draft</h2>
<p>Behavior cloning from pixels is appealing for robot manipulation, but its robustness depends strongly on what variation appears in the demonstrations. We study this question in a controlled PyBullet Panda benchmark with 128px dual-camera observations, two reaching tasks, four visual diversity axes, and closed-loop ID/OOD evaluation. Across scratch CNN, frozen ResNet-18, and partially fine-tuned ResNet-18 policies, spatial distribution is the most difficult OOD axis, while color, camera, and lighting shifts are often easier than expected. Loose success thresholds obscure this result: policies frequently reach near the target but fail to stabilize precisely. In the obstacle-aware task, additional edge-balanced data does not solve the hardest spatial failures. A phase- and geometry-conditioned diagnostic sharply improves precision, indicating that the image-only policies fail to infer hidden task stage and obstacle-target geometry reliably. These results suggest that data diversity alone is insufficient for obstacle-aware visual imitation; closed-loop robustness also requires representations that expose control-relevant spatial and phase structure.</p>
"""


def make_video_rubric_page() -> str:
    rubric = [
        {
            "category": "Problem Statement",
            "points": "20",
            "what": "Clearly explain the human and robot components, motivation, existing approaches, missing gap, and key formulation.",
            "where": "Minutes 0:00-2:00. Use the BC objective and closed-loop OOD failure formulation.",
        },
        {
            "category": "Method",
            "points": "30",
            "what": "Explain the proposed benchmark, data collection, model families, metrics, algorithms, strengths, and weaknesses.",
            "where": "Minutes 2:00-4:45. Show visual axes, tasks, policy inputs, and evaluation protocol.",
        },
        {
            "category": "Results",
            "points": "20",
            "what": "Include experiments, baselines, motivated comparisons, data analysis, nuances, and surprising outcomes.",
            "where": "Minutes 4:45-8:15. Show axis ranking, model comparison, structured diagnostic, ablations, and videos.",
        },
        {
            "category": "Connection to Class",
            "points": "10",
            "what": "Explicitly list class concepts and explain how the project goes beyond lecture.",
            "where": "Minutes 8:15-9:10. Tie to imitation learning, covariate shift, closed-loop evaluation, and HRI data collection.",
        },
        {
            "category": "Level of Difficulty",
            "points": "20",
            "what": "Show how challenging the project was and how far it went beyond class/homeworks.",
            "where": "Minutes 9:10-10:00. Emphasize full benchmark, 128px datasets, GPU sweeps, ID/OOD evals, and diagnostic extension.",
        },
    ]
    timeline = [
        {
            "time": "0:00-0:30",
            "section": "Hook",
            "content": "Show a visual-only obstacle-aware failure video first. State the practical question: which visual diversity should a human collect when training a robot from demonstrations?",
            "asset": "visual_only_scratch_ood_failure.mp4",
        },
        {
            "time": "0:30-2:00",
            "section": "Problem",
            "content": "Define human demonstrator/data designer, robot learner, pixel BC, distribution shift, and why closed-loop rollout metrics matter.",
            "asset": "BC objective and success metric equations",
        },
        {
            "time": "2:00-3:15",
            "section": "Benchmark",
            "content": "Explain reach and obstacle-aware reach, dual 128px RGB views, fixed budgets, seeds, ID/OOD splits, and visual diversity axes.",
            "asset": "dataset preview contact sheet",
        },
        {
            "time": "3:15-4:45",
            "section": "Methods",
            "content": "Compare Scratch CNN, Frozen ResNet-18, Partial ResNet-18, and the phase+geometry diagnostic model.",
            "asset": "model/input diagram or compact table",
        },
        {
            "time": "4:45-6:15",
            "section": "Visual-only results",
            "content": "Show that spatial distribution is the hardest OOD axis and that S@5cm hides precision instability.",
            "asset": "03_visual_diversity_results.html tables",
        },
        {
            "time": "6:15-7:15",
            "section": "Encoder results",
            "content": "Explain why pretrained ResNet was mixed: frozen helps loose robustness in some cases, partial fine-tuning is unstable, scratch often wins precision.",
            "asset": "spatial performance PNG and model comparison table",
        },
        {
            "time": "7:15-8:15",
            "section": "Diagnostic result",
            "content": "Show edge-balanced visual-only failure, phase+geometry improvement, ablations, and the success/failure videos.",
            "asset": "main_comparison.png, scratch_structured_ablation.png, structured videos",
        },
        {
            "time": "8:15-9:10",
            "section": "Class connection",
            "content": "Connect to imitation learning, behavior cloning, covariate shift, closed-loop control, generalization, and human data collection choices.",
            "asset": "concept checklist slide",
        },
        {
            "time": "9:10-10:00",
            "section": "Difficulty and conclusion",
            "content": "Summarize scale, automation, GPU evaluation, precision metrics, and the final takeaway: diversity matters, but obstacle-aware BC also needs phase/geometric representation.",
            "asset": "one-slide takeaway",
        },
    ]
    equations = [
        {
            "name": "Behavior cloning objective",
            "formula": "min_theta sum_t || pi_theta(o_t, q_t) - a*_t ||_2^2",
            "purpose": "Shows the supervised learning problem: match expert actions from observations and robot state.",
        },
        {
            "name": "Closed-loop rollout",
            "formula": "s_{t+1} = f(s_t, pi_theta(o_t, q_t))",
            "purpose": "Shows why small action errors compound after the policy controls the robot.",
        },
        {
            "name": "Precision success",
            "formula": "success@epsilon = 1[min_t ||p_ee,t - p_target||_2 <= epsilon]",
            "purpose": "Defines S@0.5cm, S@1cm, S@2cm, and S@5cm from closest distance reached.",
        },
        {
            "name": "Final stability",
            "formula": "final_distance = ||p_ee,T - p_target||_2",
            "purpose": "Separates reaching near the target from staying near the target at the end of rollout.",
        },
    ]
    assets = [
        {
            "asset": "Dataset previews",
            "path": "../results/previews_128px_v1/preview_contact_sheet_seed000.png",
            "use": "Use early when explaining what the robot sees.",
        },
        {
            "asset": "Main structured comparison",
            "path": "../results/structured_analysis/main_comparison.png",
            "use": "Use for the strongest result slide.",
        },
        {
            "asset": "Scratch ablation",
            "path": "../results/structured_analysis/scratch_structured_ablation.png",
            "use": "Use to prove phase alone is insufficient and geometry matters.",
        },
        {
            "asset": "Spatial bucket groups",
            "path": "../results/structured_analysis/spatial_bucket_groups.png",
            "use": "Use to discuss nuance: corners improve most, edges/interior still fail sometimes.",
        },
        {
            "asset": "Visual-only failure video",
            "path": "../results/structured_analysis/videos/visual_only_scratch_ood_failure.mp4",
            "use": "Use as opening hook or result contrast.",
        },
        {
            "asset": "Structured success video",
            "path": "../results/structured_analysis/videos/structured_scratch_ood_success.mp4",
            "use": "Use immediately after the diagnostic result.",
        },
    ]
    class_links = [
        {"concept": "Behavior cloning", "connection": "The policy is trained by supervised imitation of scripted expert actions."},
        {"concept": "Covariate shift", "connection": "Closed-loop errors move the robot into states different from demonstrations."},
        {"concept": "Closed-loop evaluation", "connection": "The project evaluates rollouts, not just train/validation loss."},
        {"concept": "Generalization", "connection": "ID/OOD splits isolate color, spatial, camera, and lighting shifts."},
        {"concept": "Human-robot interaction", "connection": "A human data designer decides what demonstrations and diversity to collect so the robot behaves robustly."},
    ]
    return f"""
<div class="callout">
  <p><strong>Rubric source.</strong> The submission video is 10 minutes. It must explain the problem, solution method, results, class connection, and difficulty. The plan below maps directly to the point allocation in <code>project_video_rubric.pdf</code>.</p>
</div>

<h2>Rubric Mapping</h2>
{table(rubric, [
    ("Category", "category", None, False),
    ("Points", "points", lambda v: fmt_int(v), True),
    ("What must be shown", "what", None, False),
    ("Where in video", "where", None, False),
])}

<h2>10-Minute Timeline</h2>
{table(timeline, [
    ("Time", "time", None, False),
    ("Section", "section", None, False),
    ("Content", "content", None, False),
    ("Asset", "asset", None, False),
])}

<h2>Equations and Formulations to Show</h2>
<p>The rubric explicitly asks for key equations/formulations. These are enough for the video without making it overly theoretical.</p>
{table(equations, [
    ("Name", "name", None, False),
    ("Formula", "formula", lambda v: f"<code>{esc(v)}</code>", False),
    ("Purpose", "purpose", None, False),
])}

<h2>Result Assets to Use</h2>
{table(assets, [
    ("Asset", "asset", None, False),
    ("Path", "path", lambda v: f'<a href="{esc(v)}">{esc(v)}</a>', False),
    ("Use in video", "use", None, False),
])}

<h2>Class Concepts to Name Explicitly</h2>
{table(class_links, [
    ("Class concept", "concept", None, False),
    ("Project connection", "connection", None, False),
])}

<h2>Suggested Closing Slide</h2>
<div class="warning">
  <p><strong>Takeaway.</strong> Visual diversity matters, but obstacle-aware robot imitation is not solved by collecting more varied images alone. The hardest failures come from spatial, phase, and geometry inference under closed-loop control. Precision metrics reveal this, and the phase+geometry diagnostic shows what information the visual policy was missing.</p>
</div>
"""


def make_rubric_execution_plan_page() -> str:
    scoring_rows = [
        {
            "category": "Problem Statement",
            "points": "20",
            "grader": "Is the problem clear, motivated, and formulated? Are human and robot components explicit? Are existing approaches and missing gaps stated?",
            "our_answer": "Human: chooses demonstrations and visual diversity. Robot: learns closed-loop manipulation from dual RGB. Gap: large robot datasets entangle diversity axes, so we isolate what matters under fixed budgets.",
            "asset": "Opening failure clip, BC objective, closed-loop rollout equation.",
        },
        {
            "category": "Method",
            "points": "30",
            "grader": "Is the solution reproducible? Are equations/algorithms provided? Are strengths, weaknesses, and comparison to alternatives discussed?",
            "our_answer": "Controlled PyBullet benchmark, two tasks, four axes, fixed budgets, ID/OOD eval, three visual BC families, and phase+geometry diagnostic.",
            "asset": "Dataset preview, model-family table, metric definitions, algorithm checklist.",
        },
        {
            "category": "Results",
            "points": "20",
            "grader": "Are experiments, baselines, data analysis, nuances, and surprising results included?",
            "our_answer": "Show visual-axis ranking, budget/model comparison, spatial OOD failure, pretrained encoder mixed results, and structured diagnostic improvement.",
            "asset": "Axis tables, main comparison PNG, ablation PNG, dual-camera case-study clips.",
        },
        {
            "category": "Connection to Class",
            "points": "10",
            "grader": "Are class concepts listed and integrated? Is it clear how the project goes beyond lecture?",
            "our_answer": "Connect behavior cloning, covariate shift, closed-loop control, distribution shift, HRI data collection, and evaluation metrics.",
            "asset": "Class-concepts slide near the end.",
        },
        {
            "category": "Level of Difficulty",
            "points": "20",
            "grader": "How challenging was the project, and how far beyond class/homework did it go?",
            "our_answer": "Full dataset pipeline, role-lab GPU sweeps, 128px dual-camera BC, multiple model families, precision ID/OOD eval, videos, and diagnostic ablations.",
            "asset": "Difficulty slide with scale numbers and pipeline summary.",
        },
    ]
    timeline_rows = [
        {
            "time": "0:00-0:35",
            "rubric": "Problem",
            "goal": "Hook with a failure case and state the practical HRI question.",
            "say": "When a human collects demonstrations for a robot, which visual diversity should they prioritize so the robot generalizes in closed loop?",
            "asset": "Dual-camera spatial OOD failure clip.",
        },
        {
            "time": "0:35-1:45",
            "rubric": "Problem",
            "goal": "Define human/robot pieces, motivation, existing gap, and formulations.",
            "say": "Pixel BC is simple but brittle because small errors move the robot into states outside the demonstrations.",
            "asset": "BC objective and closed-loop equation.",
        },
        {
            "time": "1:45-3:15",
            "rubric": "Method",
            "goal": "Make the benchmark reproducible.",
            "say": "We vary one visual axis at a time: color, spatial distribution, camera viewpoint, and lighting direction/intensity.",
            "asset": "Dataset preview and design table.",
        },
        {
            "time": "3:15-4:35",
            "rubric": "Method",
            "goal": "Explain policies and evaluation metrics.",
            "say": "We compare Scratch CNN, Frozen ResNet-18, Partial ResNet-18, and a phase+geometry diagnostic.",
            "asset": "Model families and success@epsilon equation.",
        },
        {
            "time": "4:35-6:15",
            "rubric": "Results",
            "goal": "Show visual diversity findings.",
            "say": "Spatial distribution is the hard axis; color/camera/lighting are often easier than expected.",
            "asset": "Visual diversity table and representative/best/worst clips.",
        },
        {
            "time": "6:15-7:10",
            "rubric": "Results",
            "goal": "Discuss pretrained encoder nuance.",
            "say": "Pretrained ResNet features are not automatically better in simulation; partial fine-tuning is unstable.",
            "asset": "Model comparison table.",
        },
        {
            "time": "7:10-8:35",
            "rubric": "Results",
            "goal": "Present the main diagnostic result.",
            "say": "Visual diversity alone did not solve obstacle-aware spatial OOD; phase+geometry exposes the hidden bottleneck.",
            "asset": "Main comparison, ablation, before/after dual-camera clips.",
        },
        {
            "time": "8:35-9:15",
            "rubric": "Class",
            "goal": "Explicitly name course concepts.",
            "say": "This directly uses behavior cloning, covariate shift, closed-loop evaluation, generalization, and HRI data collection decisions.",
            "asset": "Class concept checklist.",
        },
        {
            "time": "9:15-10:00",
            "rubric": "Difficulty",
            "goal": "Close with challenge level and final takeaway.",
            "say": "The project goes beyond the homework by building a controlled benchmark, running large sweeps, and diagnosing failure modes.",
            "asset": "Difficulty and final takeaway slide.",
        },
    ]
    slide_rows = [
        {"slide": "1", "title": "Title and thesis", "rubric": "Problem", "must_do": "State the one-sentence project thesis.", "asset": "No heavy result yet."},
        {"slide": "2", "title": "Pixel BC problem", "rubric": "Problem", "must_do": "Name human and robot components.", "asset": "Failure clip or still."},
        {"slide": "3", "title": "Formulation", "rubric": "Problem/Method", "must_do": "Show BC objective, rollout dynamics, and success metric.", "asset": "Equations."},
        {"slide": "4", "title": "Benchmark", "rubric": "Method", "must_do": "Show tasks, axes, and dual-camera observations.", "asset": "Preview contact sheet."},
        {"slide": "5", "title": "Data and splits", "rubric": "Method", "must_do": "Budgets, seeds, ID/OOD, 400-step rollouts.", "asset": "Compact counts."},
        {"slide": "6", "title": "Model families", "rubric": "Method", "must_do": "Explain scratch, frozen, partial, diagnostic.", "asset": "Model table."},
        {"slide": "7", "title": "Spatial is hardest", "rubric": "Results", "must_do": "Give exact axis ranking numbers.", "asset": "Axis result table."},
        {"slide": "8", "title": "Precision matters", "rubric": "Results", "must_do": "Explain S@5cm vs S@1cm and final distance.", "asset": "Metric boxes."},
        {"slide": "9", "title": "Encoders are mixed", "rubric": "Results", "must_do": "Avoid overclaiming; explain sim visual mismatch.", "asset": "Model comparison."},
        {"slide": "10", "title": "Phase+geometry diagnostic", "rubric": "Results", "must_do": "Main before/after result.", "asset": "Main comparison and clips."},
        {"slide": "11", "title": "Ablation", "rubric": "Results", "must_do": "Show why geometry matters and phase alone fails.", "asset": "Ablation PNG."},
        {"slide": "12", "title": "Rollouts", "rubric": "Results", "must_do": "Use corrected dual-camera contrast-aware clips.", "asset": "480p contrast dual gallery."},
        {"slide": "13", "title": "Class and difficulty", "rubric": "Class/Difficulty", "must_do": "List class concepts and beyond-homework work.", "asset": "Final checklist."},
    ]
    equation_rows = [
        {
            "name": "BC objective",
            "formula": "min_theta sum_t || pi_theta(o_t, q_t) - a*_t ||_2^2",
            "use": "Problem formulation and method reproducibility.",
        },
        {
            "name": "Closed-loop dynamics",
            "formula": "s_{t+1} = f(s_t, pi_theta(o_t, q_t))",
            "use": "Motivates covariate shift and rollout evaluation.",
        },
        {
            "name": "Precision success",
            "formula": "success@epsilon = 1[min_t ||p_ee,t - p_target||_2 <= epsilon]",
            "use": "Explains S@0.5cm, S@1cm, S@2cm, S@5cm.",
        },
        {
            "name": "Final distance",
            "formula": "d_final = ||p_ee,T - p_target||_2",
            "use": "Shows whether the robot stabilizes or drifts after getting close.",
        },
    ]
    asset_rows = [
        {
            "asset": "Current slide deck",
            "path": "08_video_slides.html",
            "rubric": "All",
            "status": "Use as base deck, but trim narration to stay under 10 minutes.",
        },
        {
            "asset": "Corrected video gallery",
            "path": "../results/video_case_studies_480p_contrast_dual/index.html",
            "rubric": "Results",
            "status": "Default gallery: dual-camera, slower, closer, contrast-aware.",
        },
        {
            "asset": "Video case-study index",
            "path": "09_video_case_studies.html",
            "rubric": "Results",
            "status": "Use to choose only 2-4 videos for the final recording.",
        },
        {
            "asset": "Structured main comparison",
            "path": "../results/structured_analysis/main_comparison.png",
            "rubric": "Results",
            "status": "Main result figure.",
        },
        {
            "asset": "Structured ablation",
            "path": "../results/structured_analysis/scratch_structured_ablation.png",
            "rubric": "Results",
            "status": "Explains why phase+geometry is diagnostic.",
        },
        {
            "asset": "Dataset preview",
            "path": "../results/previews_128px_v1/preview_contact_sheet_seed000.png",
            "rubric": "Method",
            "status": "Shows observations and visual axes.",
        },
    ]
    checklist_rows = [
        {"check": "Problem has human and robot components", "done_by": "Slide 2", "status": "Human collects/designs demonstrations; robot learns from pixels."},
        {"check": "Motivation and gap are explicit", "done_by": "Slides 1-2", "status": "Controlled data composition under fixed budgets."},
        {"check": "Equations/formulations are shown", "done_by": "Slide 3", "status": "BC, closed-loop dynamics, success metric."},
        {"check": "Method is reproducible", "done_by": "Slides 4-6", "status": "Tasks, axes, budgets, seeds, models, metrics."},
        {"check": "Baselines are included", "done_by": "Slides 6, 9", "status": "Scratch CNN, Frozen ResNet-18, Partial ResNet-18."},
        {"check": "Nuances and surprising results are covered", "done_by": "Slides 8-11", "status": "Loose thresholds hide failure; pretrained encoders mixed; phase+geometry diagnostic."},
        {"check": "Class concepts are listed", "done_by": "Slide 13", "status": "BC, covariate shift, closed-loop control, generalization, HRI data collection."},
        {"check": "Difficulty is argued", "done_by": "Slide 13", "status": "Full pipeline, GPU sweeps, 128px datasets, ID/OOD eval, ablations, videos."},
        {"check": "Team logistics", "done_by": "After recording", "status": "Submit one video link; each team member submits a one-page contribution document if required."},
    ]

    return f"""
<div class="callout">
  <p><strong>Purpose.</strong> This is the rubric-first execution plan for the final 10-minute HRI project video. Use this page as the checklist before recording. The goal is not to show every experiment; it is to satisfy every rubric item with the strongest story and evidence.</p>
</div>

<h2>Scoring Strategy</h2>
{table(scoring_rows, [
    ("Rubric category", "category", None, False),
    ("Pts", "points", lambda v: fmt_int(v), True),
    ("What the grader asks", "grader", None, False),
    ("Our answer", "our_answer", None, False),
    ("Asset/evidence", "asset", None, False),
])}

<h2>Recommended 10-Minute Flow</h2>
{table(timeline_rows, [
    ("Time", "time", None, False),
    ("Rubric", "rubric", None, False),
    ("Goal", "goal", None, False),
    ("What to say", "say", None, False),
    ("Asset", "asset", None, False),
])}

<h2>Slide Coverage</h2>
{table(slide_rows, [
    ("Slide", "slide", lambda v: fmt_int(v), True),
    ("Title", "title", None, False),
    ("Rubric", "rubric", None, False),
    ("Must do", "must_do", None, False),
    ("Asset", "asset", None, False),
])}

<h2>Required Formulations</h2>
{table(equation_rows, [
    ("Name", "name", None, False),
    ("Formula", "formula", lambda v: f"<code>{esc(v)}</code>", False),
    ("Use", "use", None, False),
])}

<h2>Asset Checklist</h2>
{table(asset_rows, [
    ("Asset", "asset", None, False),
    ("Path", "path", lambda v: f'<a href="{esc(v)}">{esc(v)}</a>', False),
    ("Rubric", "rubric", None, False),
    ("Status", "status", None, False),
])}

<h2>Before Recording</h2>
{table(checklist_rows, [
    ("Check", "check", None, False),
    ("Where handled", "done_by", None, False),
    ("Status", "status", None, False),
])}

<h2>Final Spoken Thesis</h2>
<div class="warning">
  <p>Pixel behavior cloning can tolerate some appearance and viewpoint variation, but obstacle-aware spatial generalization fails when the policy cannot infer task phase and geometry. Controlled visual diversity experiments reveal the failure; phase+geometry diagnostics explain it.</p>
</div>
"""


def slide(slide_no: int, title: str, kicker: str, body: str, transcript: str) -> str:
    return f"""
<section class="slide" id="slide-{slide_no}">
  <div class="slide-screen">
    <div>
      <div class="slide-kicker">{esc(kicker)}</div>
      <h2 class="slide-title">{esc(title)}</h2>
    </div>
    <div class="slide-body">
      {body}
    </div>
    <div class="slide-footer"><span>Slide {slide_no}</span><span>Visual BC under distribution shift</span></div>
  </div>
  <aside class="transcript">
    <h3>Transcript</h3>
    <p>{esc(transcript)}</p>
  </aside>
</section>
"""


def make_video_slides_page() -> str:
    slide_data = [
        (
            "What Visual Diversity Matters for Closed-Loop Behavior Cloning?",
            "Title and thesis",
            """
<ul class="big-points">
  <li>Controlled PyBullet Panda study with 128px dual-camera observations.</li>
  <li>Two tasks: direct reach and obstacle-aware reach.</li>
  <li>Main finding: spatial and phase/geometry structure dominate the hardest failures.</li>
</ul>
""",
            "This project asks a practical question in robot imitation learning: if collecting demonstrations is expensive, what kind of visual diversity should we collect first? We study this with a controlled PyBullet Panda benchmark using pixel observations. The short answer is that color, camera, and lighting shifts matter, but they are not the hardest part. The hardest failures come from spatial distribution shift and from hidden phase and obstacle geometry in the obstacle-aware task. The final story is therefore not just more visual diversity helps. It is that closed-loop visual behavior cloning needs the right control-relevant structure.",
        ),
        (
            "Problem: Pixel BC Breaks Under Closed-Loop Shift",
            "Problem statement",
            """
<div class="metric-grid">
  <div class="metric-box"><strong>Human</strong><span>chooses what demonstrations and visual diversity to collect</span></div>
  <div class="metric-box"><strong>Robot</strong><span>learns a policy from RGB observations and robot state</span></div>
  <div class="metric-box"><strong>Gap</strong><span>large datasets entangle camera, color, lighting, layout, and task variation</span></div>
</div>
<ul class="big-points">
  <li>We isolate one visual factor at a time under fixed demonstration budgets.</li>
</ul>
""",
            "The human-robot interaction component is the data collection decision. A human demonstrator or system designer decides which demonstrations the robot should see, and the robot must use those demonstrations to act robustly. Behavior cloning is attractive because it is simple: learn actions from expert trajectories. But in closed loop, small prediction errors change the robot state, which changes the next image, and errors can compound. Existing robot datasets often scale up by varying many things at once, so it is hard to know whether color, camera pose, lighting, or spatial layout actually caused better generalization.",
        ),
        (
            "Formulation: Imitation Plus Closed-Loop Evaluation",
            "Equations",
            """
<div class="equation"><code>min_theta sum_t || pi_theta(o_t, q_t) - a*_t ||_2^2</code></div>
<div class="equation"><code>s_{t+1} = f(s_t, pi_theta(o_t, q_t))</code></div>
<div class="equation"><code>success@epsilon = 1[min_t ||p_ee,t - p_target||_2 <= epsilon]</code></div>
<ul class="big-points">
  <li>Report S@0.5cm, S@1cm, S@2cm, S@5cm, best distance, and final distance.</li>
</ul>
""",
            "The training objective is standard behavior cloning: minimize the squared error between the policy action and the expert action. The important part is evaluation. Once the robot acts using its own policy, the next state depends on the model's previous actions. That is why we evaluate closed-loop rollouts, not only supervised loss. For success, we compute whether the end effector ever gets within a threshold of the target. We report several thresholds from half a centimeter to five centimeters. We also report the closest distance reached and the final distance, because a policy can pass near the target and then drift away.",
        ),
        (
            "Benchmark: Two Tasks, Four Visual Axes",
            "Experimental setup",
            """
<div class="slide-body two-col">
  <ul class="big-points">
    <li>Reach: move to a cube target.</li>
    <li>Obstacle-aware reach: route around an obstacle, then descend to the target.</li>
    <li>Axes: color, spatial distribution, camera viewpoint, lighting direction and intensity.</li>
  </ul>
  <img src="../results/previews_128px_v1/preview_contact_sheet_seed000.png" alt="Dataset preview contact sheet">
</div>
""",
            "The benchmark uses two related tasks. The first is direct reaching to a cube. The second is obstacle-aware reaching, where the robot needs to route around a blocking obstacle before reaching the same target. We collect 128 by 128 pixel observations from an external camera and a wrist or end-effector camera. The four visual diversity axes are color, spatial target distribution, camera location or viewpoint, and lighting direction plus intensity. The important design choice is that we vary one factor at a time, so the comparison is interpretable.",
        ),
        (
            "Data and Splits",
            "Reproducibility",
            """
<div class="metric-grid">
  <div class="metric-box"><strong>5 / 20 / 50</strong><span>demos, seeds 0, 1, 2</span></div>
  <div class="metric-box"><strong>100 / 200</strong><span>demos, seed 0 directional high-budget runs</span></div>
  <div class="metric-box"><strong>ID + OOD</strong><span>held-out eval seeds and held-out visual conditions</span></div>
  <div class="metric-box"><strong>400</strong><span>rollout step horizon with 1mm early stop</span></div>
</div>
<ul class="big-points">
  <li>Each demo contributes trajectory images and actions, not just start/end frames.</li>
</ul>
""",
            "For the lower budgets, we use three training seeds, which gives a stronger comparison at 5, 20, and 50 demonstrations. We also ran higher-budget 100 and 200 demonstration experiments with seed 0, so those trends should be treated as directional. For evaluation, we separate in-distribution and out-of-distribution conditions. Rollouts run for up to 400 steps and only stop early if the robot gets within one millimeter, so the final distance is not artificially clipped at five centimeters. Each demonstration provides many time steps along the trajectory, which is important for behavior cloning.",
        ),
        (
            "Methods: Three Visual Policies and One Diagnostic",
            "Model families",
            """
<ul class="big-points">
  <li>Scratch CNN BC: end-to-end visual policy baseline.</li>
  <li>Frozen ResNet-18: pretrained encoder with learned policy head.</li>
  <li>Partial ResNet-18: later ResNet layers fine-tuned.</li>
  <li>Phase+geometry BC: diagnostic policy with RGB plus explicit task stage and geometry.</li>
</ul>
""",
            "We compare three visual behavior cloning model families. The scratch CNN is the main end-to-end pixel baseline. The frozen ResNet-18 tests whether pretrained visual features help, especially since the professor's feedback suggested using a visual encoder. The partially fine-tuned ResNet-18 tests whether adapting the later layers improves over frozen features. Finally, after seeing persistent obstacle-aware failures, we add a diagnostic model that still uses RGB, but also receives phase and geometry information. This is not pure pixel BC. It is a diagnostic to identify what information the visual policies failed to infer.",
        ),
        (
            "Result 1: Spatial Distribution Is the Hard Axis",
            "Visual diversity",
            """
<div class="metric-grid">
  <div class="metric-box"><strong>57.9%</strong><span>Scratch CNN camera OOD S@1cm, budget 50</span></div>
  <div class="metric-box"><strong>41.1%</strong><span>Scratch CNN color OOD S@1cm, budget 50</span></div>
  <div class="metric-box"><strong>20.6%</strong><span>Scratch CNN lighting OOD S@1cm, budget 50</span></div>
  <div class="metric-box"><strong>2.9%</strong><span>Scratch CNN spatial OOD S@1cm, budget 50</span></div>
</div>
<ul class="big-points">
  <li>Spatial OOD remains difficult even when appearance shifts are manageable.</li>
</ul>
""",
            "The first major result is the visual-axis ranking. At budget 50, which is a multi-seed comparison, the scratch CNN handles camera and color shifts much better than expected. It gets 57.9 percent success at one centimeter on camera OOD and 41.1 percent on color OOD. Lighting is weaker at 20.6 percent. Spatial distribution is the clear failure, with only 2.9 percent success at one centimeter. This supports the original hypothesis that spatial distribution is more important than superficial appearance, and it also tells us where to focus the deeper analysis.",
        ),
        (
            "Result 2: Loose Success Hides Precision Failure",
            "Metrics matter",
            """
<div class="metric-grid">
  <div class="metric-box"><strong>78.5%</strong><span>Scratch CNN OOD S@5cm at budget 50</span></div>
  <div class="metric-box"><strong>30.6%</strong><span>Scratch CNN OOD S@1cm at budget 50</span></div>
  <div class="metric-box"><strong>4.4 cm</strong><span>mean best distance</span></div>
  <div class="metric-box"><strong>40.4 cm</strong><span>mean final distance</span></div>
</div>
<ul class="big-points">
  <li>Getting close once is not the same as stable closed-loop behavior.</li>
</ul>
""",
            "The second result is about measurement. If we only report success at five centimeters, scratch CNN OOD at budget 50 looks fairly strong, around 78.5 percent. But at one centimeter it is only 30.6 percent. Even more importantly, the mean best distance is 4.4 centimeters, while the final distance is 40.4 centimeters. That means many rollouts get near the target at some point but do not stabilize there. This is why the multi-threshold metric and final distance are central to the story.",
        ),
        (
            "Result 3: Pretrained Encoders Are Mixed in Simulation",
            "Model comparison",
            """
<div class="slide-body two-col">
  <ul class="big-points">
    <li>Scratch CNN often has the best precision.</li>
    <li>Frozen ResNet-18 sometimes helps loose success but not precision.</li>
    <li>Partial ResNet-18 is unstable across axes and tasks.</li>
  </ul>
  <img src="../results/analysis_spatial_performance/spatial_performance_table_phase_precise_budget200.png" alt="Spatial performance table at budget 200">
</div>
""",
            "The third result answers the representation question. Pretrained ResNet features were not a simple solution in this simulated environment. Frozen ResNet-18 can look better at loose thresholds in some settings, but it does not consistently win at one-centimeter precision. Partial fine-tuning is even more unstable. This is an important nuance because using a visual encoder was a reasonable idea, but simulation images and robot control geometry do not necessarily align with features learned from natural images. In this benchmark, representation quality helps only when it captures the geometry needed for control.",
        ),
        (
            "Diagnostic: Visual Diversity Alone Did Not Fix Obstacle-Aware Spatial OOD",
            "Phase and geometry",
            """
<div class="slide-body two-col">
  <div class="metric-grid">
    <div class="metric-box"><strong>1.3%</strong><span>visual-only scratch OOD S@1cm</span></div>
    <div class="metric-box"><strong>53.3%</strong><span>phase+geometry scratch OOD S@1cm</span></div>
    <div class="metric-box"><strong>19.25 cm</strong><span>visual-only final distance</span></div>
    <div class="metric-box"><strong>4.23 cm</strong><span>phase+geometry final distance</span></div>
  </div>
  <img src="../results/structured_analysis/main_comparison.png" alt="Structured main comparison">
</div>
""",
            "After spatial remained weak, we collected edge-balanced obstacle-aware data to cover edges, corners, and interior target regions. Surprisingly, visual-only training still failed badly. That pushed us to the diagnostic model. With the same visual inputs plus explicit phase and geometry, the scratch model improves from 1.3 percent OOD success at one centimeter to 53.3 percent. Final distance also drops from 19.25 centimeters to 4.23 centimeters. This is the strongest result in the project: obstacle-aware spatial OOD is not just a matter of collecting more varied images. The policy also needs reliable phase and geometry information.",
        ),
        (
            "Ablation: Geometry Matters, Phase Alone Is Not Enough",
            "Why it works",
            """
<div class="slide-body two-col">
  <ul class="big-points">
    <li>Phase only fails: it cannot locate target or obstacle.</li>
    <li>Geometry only gives strong coarse reaching and stability.</li>
    <li>Phase+geometry gives the best 1cm precision but remains somewhat bimodal.</li>
  </ul>
  <img src="../results/structured_analysis/scratch_structured_ablation.png" alt="Scratch structured ablation">
</div>
""",
            "The ablation makes the diagnostic more defensible. Phase alone is not useful, because knowing the task stage does not tell the policy where the target or obstacle is. Geometry-only models are surprisingly strong at loose success and final stability, which means spatial relationships are central. The full phase-plus-geometry model gives the best one-centimeter precision, but it is somewhat bimodal: many rollouts become precise, while some still fail. This nuance is useful for the final discussion. The result is not that we solved everything; it is that we identified the bottleneck more clearly.",
        ),
        (
            "Representative Rollouts",
            "Qualitative evidence",
            """
<div class="slide-body two-col">
  <video controls preload="metadata" src="../results/video_case_studies_480p_contrast_dual/visual_axes/avoid_reach/spatial_distribution/avoid_spatial_wide/worst/01__seed101__ep040__best33p3cm__final50p8cm.mp4"></video>
  <video controls preload="metadata" src="../results/video_case_studies_480p_contrast_dual/improvements/avoid_reach/spatial_distribution/phase_geometry_scratch/best/01__seed102__ep020__best0p2cm__final0p3cm.mp4"></video>
</div>
<ul class="big-points">
  <li>Left: visual-only spatial OOD failure. Right: closer/slower phase+geometry OOD success. Both show wrist-camera inset.</li>
</ul>
""",
            "For the video submission, this slide should show the qualitative behavior. The left rollout is the visual-only scratch model failing in the OOD obstacle-aware setting. The right rollout is the phase-plus-geometry scratch model succeeding. These videos make the metric difference concrete. They also help explain why final distance matters: the robot can move in a way that initially looks plausible, but if the policy chooses the wrong stage or spatial route, it never recovers. The diagnostic policy produces behavior that is more directed and more stable.",
        ),
        (
            "Connection to Class and Difficulty",
            "Rubric alignment",
            """
<div class="metric-grid">
  <div class="metric-box"><strong>BC</strong><span>supervised imitation policy</span></div>
  <div class="metric-box"><strong>Shift</strong><span>closed-loop covariate shift</span></div>
  <div class="metric-box"><strong>HRI</strong><span>human data collection choices affect robot behavior</span></div>
  <div class="metric-box"><strong>Beyond</strong><span>benchmark, datasets, GPU sweeps, ID/OOD metrics, diagnostic ablations</span></div>
</div>
<ul class="big-points">
  <li>Final takeaway: collect diversity, but also represent phase and geometry.</li>
</ul>
""",
            "This project connects directly to behavior cloning and covariate shift, which are core imitation-learning ideas. It also has a human-robot interaction angle: the human's data collection choices determine whether the robot can generalize safely and precisely. The project goes beyond class by building a full controlled benchmark, collecting 128 pixel datasets, training three model families, running ID and OOD closed-loop evaluations at multiple precision thresholds, and then adding a diagnostic ablation study. The final takeaway is that visual diversity matters, but for obstacle-aware manipulation, diversity alone is not enough. The policy must also represent task phase and geometry.",
        ),
    ]

    jumps = '<div class="slide-jump">' + "".join(
        f'<a href="#slide-{i}">{i}</a>' for i in range(1, len(slide_data) + 1)
    ) + "</div>"
    slides = "".join(
        slide(i, title, kicker, body, transcript)
        for i, (title, kicker, body, transcript) in enumerate(slide_data, start=1)
    )
    return f"""
<div class="callout">
  <p><strong>Use this page as the recording script.</strong> Each slide is HTML-only and includes a visible transcript. The deck is designed for a 10-minute video; average pacing is about 45-55 seconds per slide.</p>
</div>
{jumps}
<div class="deck">
  {slides}
</div>
"""


def make_video_case_studies_page() -> str:
    preferred_rel = "results/video_case_studies_480p_contrast_dual/recording_manifest.csv"
    fallback_rel = "results/video_case_studies_480p_slow_close_dual/recording_manifest.csv"
    fallback_slow_rel = "results/video_case_studies_480p_slow_close/recording_manifest.csv"
    fallback_1080_rel = "results/video_case_studies_1080p/recording_manifest.csv"
    legacy_rel = "results/video_case_studies/recording_manifest.csv"
    if (ROOT / preferred_rel).exists():
        manifest_rel = preferred_rel
    elif (ROOT / fallback_rel).exists():
        manifest_rel = fallback_rel
    elif (ROOT / fallback_slow_rel).exists():
        manifest_rel = fallback_slow_rel
    elif (ROOT / fallback_1080_rel).exists():
        manifest_rel = fallback_1080_rel
    else:
        manifest_rel = legacy_rel
    gallery_rel = str(Path(manifest_rel).parent / "index.html")
    manifest_path = ROOT / manifest_rel
    if not manifest_path.exists():
        return """
<div class="warning">
  <p>The video case-study manifest has not been generated yet. Run <code>python code/generate_video_case_studies.py --per-case 3</code> on the machine with models and eval datasets.</p>
</div>
"""
    rows = load_csv(manifest_rel)
    status_counts = Counter(row.get("record_status", "") for row in rows)
    group_counts = Counter(row["group"] for row in rows)
    case_type_counts = Counter(row["case_type"] for row in rows)
    available_count = sum(1 for row in rows if (ROOT / row["video_path"]).exists())

    summary_rows = [
        {"item": "Total videos", "count": len(rows), "meaning": "All generated best/worst rollout clips."},
        {"item": "Available MP4s", "count": available_count, "meaning": "Video files present on disk and linked by the manifest."},
        {"item": "Visual-axis videos", "count": group_counts.get("visual_axes", 0), "meaning": "4 axes x 2 tasks x best/worst x 3 clips."},
        {"item": "Improvement videos", "count": group_counts.get("improvements", 0), "meaning": "Diagnostic before/after and ablation clips."},
        {"item": "Best clips", "count": case_type_counts.get("best", 0), "meaning": "Lowest best-distance rollouts within each case."},
        {"item": "Worst clips", "count": case_type_counts.get("worst", 0), "meaning": "Highest best-distance rollouts within each case."},
        {"item": "Recorder status ok/skipped", "count": status_counts.get("ok", 0) + status_counts.get("skipped", 0), "meaning": "Ok means recorded in that pass; skipped means the verified MP4 already existed."},
    ]

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["task"], row["axis"], row["case_label"], row["case_type"])].append(row)

    case_rows = []
    for key, items in sorted(grouped.items()):
        group, task, axis, case_label, case_type = key
        best_values = [fnum(row["best_distance_cm"]) for row in items]
        final_values = [fnum(row["final_distance_cm"]) for row in items]
        first_video = "../" + items[0]["video_path"]
        case_rows.append(
            {
                "group": group,
                "task": task,
                "axis": axis,
                "case_label": case_label,
                "case_type": case_type,
                "count": len(items),
                "best_range": f"{min(best_values):.1f}-{max(best_values):.1f}",
                "final_range": f"{min(final_values):.1f}-{max(final_values):.1f}",
                "first_video": first_video,
            }
        )

    priority_labels = {
        ("visual_axes", "reach", "spatial_distribution", "reach_spatial_wide", "worst"),
        ("visual_axes", "avoid_reach", "spatial_distribution", "avoid_spatial_wide", "worst"),
        ("improvements", "avoid_reach", "spatial_distribution", "edge_balanced_visual_only_scratch", "worst"),
        ("improvements", "avoid_reach", "spatial_distribution", "phase_geometry_scratch", "best"),
        ("improvements", "avoid_reach", "spatial_distribution", "phase_geometry_scratch", "worst"),
        ("improvements", "avoid_reach", "spatial_distribution", "phase_only_ablation", "worst"),
    }
    featured = []
    for key in sorted(priority_labels):
        items = sorted(grouped.get(key, []), key=lambda row: int(row["rank"]))
        for row in items[:1]:
            caption = (
                f"{row['group']} / {row['task']} / {row['axis']} / {row['case_label']} / {row['case_type']} "
                f"rank {row['rank']} | best {row['best_distance_cm']} cm | final {row['final_distance_cm']} cm"
            )
            featured.append(video("../" + row["video_path"], caption))

    case_table = table(
        case_rows,
        [
            ("Group", "group", None, False),
            ("Task", "task", None, False),
            ("Axis", "axis", None, False),
            ("Case", "case_label", None, False),
            ("Type", "case_type", None, False),
            ("Videos", "count", lambda v: fmt_int(v), True),
            ("Best cm range", "best_range", None, True),
            ("Final cm range", "final_range", None, True),
            ("First clip", "first_video", lambda v: f'<a href="{esc(v)}">open</a>', False),
        ],
    )

    return f"""
<div class="callout">
  <p><strong>Video library.</strong> This page indexes the generated rollout case studies. The full standalone gallery is <a href="../{esc(gallery_rel)}">{esc(gallery_rel)}</a>, and the machine-readable manifest is <a href="../{esc(manifest_rel)}">recording_manifest.csv</a>. The preferred set is 854x480, slower playback, a closer external camera view, wrist/end-effector camera inset, and contrast-aware labels: flat cases are representative instead of fake best/worst.</p>
</div>

<h2>Recording Summary</h2>
{table(summary_rows, [
    ("Item", "item", None, False),
    ("Count", "count", lambda v: fmt_int(v), True),
    ("Meaning", "meaning", None, False),
])}

<h2>Featured Clips</h2>
<div class="video-grid">
  {''.join(featured)}
</div>

<h2>All Cases</h2>
{case_table}
"""


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        "overall": load_csv("results/analysis_id_ood_all_budgets/overall_by_family_split_budget.csv"),
        "ood_axis": load_csv("results/analysis_id_ood_all_budgets/ood_axis_by_family_budget.csv"),
        "ood_task": load_csv("results/analysis_id_ood_all_budgets/ood_task_by_family_budget.csv"),
        "diversity_gains": load_csv("results/analysis_id_ood_all_budgets/diversity_pair_gains_ood_by_family_budget.csv"),
        "main_comparison": load_csv("results/structured_analysis/main_comparison.csv"),
        "ablation": load_csv("results/structured_analysis/scratch_structured_ablation.csv"),
        "bucket": load_csv("results/structured_analysis/spatial_bucket_breakdown.csv"),
    }
    page(
        "index.html",
        "Comprehensive Analysis",
        "Final story for the visual diversity and obstacle-aware behavior cloning project.",
        make_overview(),
    )
    page(
        "01_proposal_to_story.html",
        "Proposal to Final Story",
        "How the initial controlled visual-diversity proposal became a phase and geometry diagnostic study.",
        make_proposal_page(),
    )
    page(
        "02_experiment_design.html",
        "Experiment Design",
        "Tasks, diversity axes, datasets, model families, and precision rollout metrics.",
        make_design_page(),
    )
    page(
        "03_visual_diversity_results.html",
        "Visual Diversity Results",
        "Budget scaling, OOD axis ranking, task basis, and diversity pair gains.",
        make_visual_results_page(data),
    )
    page(
        "04_model_comparison.html",
        "Model Comparison",
        "Scratch CNN, frozen ResNet-18, and partially fine-tuned ResNet-18 under ID/OOD evaluation.",
        make_model_page(data),
    )
    page(
        "05_structured_diagnostic.html",
        "Structured Diagnostic",
        "Phase+geometry-conditioned visual BC results, ablations, spatial bucket analysis, and rollout videos.",
        make_structured_page(data),
    )
    page(
        "06_paper_story.html",
        "Paper Story",
        "Claims, paper structure, figure plan, limitations, and abstract draft.",
        make_paper_story_page(),
    )
    page(
        "07_video_rubric_plan.html",
        "Video Rubric Plan",
        "A 10-minute submission outline mapped directly to the HRI project video rubric.",
        make_video_rubric_page(),
    )
    page(
        "08_video_slides.html",
        "Video Slides With Transcripts",
        "HTML-only slide deck for the 10-minute project video, with speaker transcript for each slide.",
        make_video_slides_page(),
    )
    page(
        "09_video_case_studies.html",
        "Video Case Studies",
        "Best and worst rollout video library for visual diversity axes, tasks, and improvement diagnostics.",
        make_video_case_studies_page(),
    )
    page(
        "10_rubric_execution_plan.html",
        "Rubric Execution Plan",
        "Point-by-point checklist for satisfying the HRI project video rubric.",
        make_rubric_execution_plan_page(),
    )


if __name__ == "__main__":
    os.chdir(ROOT)
    build()
