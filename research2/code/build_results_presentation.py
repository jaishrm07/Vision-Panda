#!/usr/bin/env python3
"""Build the results-focused HTML slide deck for the V-BCOOD project."""

from __future__ import annotations

import csv
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANALYSIS = ROOT / "analysis"
OUT = ANALYSIS / "12_presentation.html"

BUDGETS = [5, 20, 50, 100, 200]
FAMILIES = [
    ("scratch_cnn", "Scratch CNN", "#2f6fad"),
    ("frozen_resnet18", "Frozen ResNet-18", "#26845a"),
    ("partial_resnet18", "Partial ResNet-18", "#c66a2b"),
]
AXES = [
    ("color", "Color"),
    ("camera_location_viewpoint", "Camera"),
    ("lighting_direction_intensity", "Lighting"),
    ("spatial_distribution", "Spatial"),
]
TASKS = [
    ("reach", "Reach"),
    ("avoid_reach", "Obstacle-aware"),
]
RAW_AXIS_META = {
    "color": {
        "label": "Color Diversity",
        "description": "cube color changes",
        "color": "#2f6fad",
    },
    "camera": {
        "label": "Camera View Diversity",
        "description": "external viewpoint changes",
        "color": "#26845a",
    },
    "spatial": {
        "label": "Spatial Diversity",
        "description": "target location changes",
        "color": "#c74343",
    },
    "lighting": {
        "label": "Lighting Diversity",
        "description": "light direction and intensity change",
        "color": "#c66a2b",
    },
}
RAW_AXIS_ORDER = ["color", "camera", "spatial", "lighting"]


def read_csv(rel_path: str) -> list[dict[str, str]]:
    path = RESULTS / rel_path
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


OVERALL = read_csv("analysis_id_ood_all_budgets/overall_by_family_split_budget.csv")
AXIS = read_csv("analysis_id_ood_all_budgets/ood_axis_by_family_budget.csv")
AXIS_SPLIT = read_csv("analysis_id_ood_all_budgets/axis_by_family_split_budget.csv")
TASK = read_csv("analysis_id_ood_all_budgets/ood_task_by_family_budget.csv")
TRAIN_CONFIG = read_csv("analysis_id_ood_all_budgets/train_config_by_family_split_budget.csv")
STRUCTURED = read_csv("structured_analysis/main_comparison.csv")
ABLATION = read_csv("structured_analysis/scratch_structured_ablation.csv")
BUCKETS = read_csv("structured_analysis/spatial_bucket_breakdown.csv")


def read_json(rel_path: str) -> dict[str, object]:
    path = ROOT / rel_path
    with path.open(encoding="utf-8") as f:
        return json.load(f)


TRAIN_PRIMARY_SUMMARY = read_json("results/datasets_128px_v1/collection_summary.json")
TRAIN_ALL_SUMMARY = read_json("results/datasets_128px_v1/collection_summary_high_budget_100_200_seed0_20260507T025331Z.json")
EVAL_ID_SUMMARY = read_json("results/eval_id_128px_v1/collection_summary.json")
EVAL_OOD_SUMMARY = read_json("results/eval_ood_128px_v1/collection_summary.json")

FINAL_AVOID_DIRS = {
    "scratch_cnn": "bc_128px_phase_precise_avoid",
    "frozen_resnet18": "frozen_resnet18_bc_128px_phase_precise_avoid",
    "partial_resnet18": "partial_resnet18_bc_128px_phase_precise_avoid",
}
FINAL_AVOID_CONFIGS = {
    "color": [
        ("avoid_color_red_only", "avoid_color_ood_eval"),
        ("avoid_color_multi", "avoid_color_ood_eval"),
    ],
    "spatial_distribution": [
        ("avoid_spatial_narrow", "avoid_spatial_edge_ood_eval"),
        ("avoid_spatial_wide", "avoid_spatial_edge_ood_eval"),
    ],
    "camera_location_viewpoint": [
        ("avoid_camera_fixed", "avoid_camera_extreme_ood_eval"),
        ("avoid_camera_multi_pose", "avoid_camera_extreme_ood_eval"),
    ],
    "lighting_direction_intensity": [
        ("avoid_lighting_fixed", "avoid_lighting_extreme_ood_eval"),
        ("avoid_lighting_diverse", "avoid_lighting_extreme_ood_eval"),
    ],
}


def rel(path: str | Path) -> str:
    return os.path.relpath(Path(path), OUT.parent)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fnum(row: dict[str, str], key: str) -> float:
    return float(row[key])


def find(rows: list[dict[str, str]], **criteria: object) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == str(value) for key, value in criteria.items()):
            return row
    raise KeyError(f"missing row: {criteria}")


def pct(value: float) -> str:
    return f"{value:.1f}%"


def cm(value: float) -> str:
    return f"{value:.1f} cm"


def pp(value: float) -> str:
    return f"{value:+.1f} pp"


def int_fmt(value: int | float | object) -> str:
    return f"{int(value):,}"


def axis_label(axis: str) -> str:
    return dict(AXES)[axis]


def budget_rows_for(family: str, split: str = "ood") -> list[dict[str, str]]:
    return [find(OVERALL, family=family, split=split, budget=b) for b in BUDGETS]


def overall_value(family: str, split: str, budget: int, metric: str) -> float:
    return fnum(find(OVERALL, family=family, split=split, budget=budget), metric)


def axis_value(family: str, budget: int, axis: str, metric: str) -> float:
    return fnum(find(AXIS, family=family, budget=budget, axis=axis), metric)


def axis_split_value(family: str, split: str, budget: int, axis: str, metric: str) -> float:
    return fnum(find(AXIS_SPLIT, family=family, split=split, budget=budget, axis=axis), metric)


SUCCESS_METRIC_TO_JSON_KEY = {
    "success_at_1cm_pct": "success_rate_at_1cm",
    "success_at_2cm_pct": "success_rate_at_2cm",
    "success_at_5cm_pct": "success_rate_at_5cm",
}


def axis_task_rows(family: str, split: str, budget: int, axis: str, task: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in TRAIN_CONFIG
        if row["family"] == family
        and row["split"] == split
        and row["budget"] == str(budget)
        and row["axis"] == axis
        and row["task"] == task
    ]
    if not rows:
        raise KeyError(f"missing task-axis rows: {family=} {split=} {budget=} {axis=} {task=}")
    return rows


def axis_task_rollouts(family: str, split: str, budget: int, axis: str, task: str) -> float:
    return sum(fnum(row, "rollout_count") for row in axis_task_rows(family, split, budget, axis, task))


def axis_task_value(family: str, split: str, budget: int, axis: str, task: str, metric: str) -> float:
    rows = axis_task_rows(family, split, budget, axis, task)
    total_rollouts = sum(fnum(row, "rollout_count") for row in rows)
    return sum(fnum(row, metric) * fnum(row, "rollout_count") for row in rows) / total_rollouts


def final_avoid_config_pairs(axis: str, split: str) -> list[tuple[str, str]]:
    if split == "ood":
        return FINAL_AVOID_CONFIGS[axis]
    if split == "id":
        return [(train_config, train_config) for train_config, _ in FINAL_AVOID_CONFIGS[axis]]
    raise ValueError(f"unsupported split for final avoid metrics: {split}")


def final_avoid_metric_components(family: str, split: str, axis: str, metric: str) -> tuple[float, float]:
    summary_key = SUCCESS_METRIC_TO_JSON_KEY[metric]
    result_dir = FINAL_AVOID_DIRS[family]
    values: list[tuple[float, float]] = []
    for train_config, eval_config in final_avoid_config_pairs(axis, split):
        pattern = f"metrics__*__{train_config}__budget200__seed000__eval_{eval_config}.json"
        matches = sorted((RESULTS / result_dir / f"metrics/{split}").glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one final avoid metric for {family=} {split=} {train_config=}: {matches}")
        payload = read_json(matches[0].relative_to(ROOT))
        rollouts = float(payload["metrics"]["rollout_count"])
        values.append((float(payload["metrics"][summary_key]) * 100.0, rollouts))
    total_rollouts = sum(rollouts for _, rollouts in values)
    return sum(value * rollouts for value, rollouts in values) / total_rollouts, total_rollouts


def final_avoid_metric_value(family: str, split: str, axis: str, metric: str) -> float:
    value, _ = final_avoid_metric_components(family, split, axis, metric)
    return value


def axis_split_slide_value(family: str, split: str, budget: int, axis: str, metric: str) -> float:
    if budget == 200 and metric in SUCCESS_METRIC_TO_JSON_KEY:
        reach_value = axis_task_value(family, split, budget, axis, "reach", metric)
        reach_rollouts = axis_task_rollouts(family, split, budget, axis, "reach")
        avoid_value, avoid_rollouts = final_avoid_metric_components(family, split, axis, metric)
        total_rollouts = reach_rollouts + avoid_rollouts
        return ((reach_value * reach_rollouts) + (avoid_value * avoid_rollouts)) / total_rollouts
    return axis_split_value(family, split, budget, axis, metric)


def task_split_slide_value(family: str, split: str, budget: int, axis: str, task: str, metric: str) -> float:
    if budget == 200 and task == "avoid_reach" and metric in SUCCESS_METRIC_TO_JSON_KEY:
        return final_avoid_metric_value(family, split, axis, metric)
    return axis_task_value(family, split, budget, axis, task, metric)


def task_value(family: str, budget: int, task: str, metric: str) -> float:
    return fnum(find(TASK, family=family, budget=budget, task=task), metric)


def structured_row(group: str, model: str, split: str) -> dict[str, str]:
    return find(STRUCTURED, group=group, model=model, split=split)


def ablation_row(model: str, split: str) -> dict[str, str]:
    return find(ABLATION, model=model, split=split)


def svg_text(x: float, y: float, text: str, cls: str = "", anchor: str = "middle") -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{class_attr}>{esc(text)}</text>'


def y_ticks(max_y: float) -> list[float]:
    if max_y <= 10:
        step = 2
    elif max_y <= 40:
        step = 10
    elif max_y <= 80:
        step = 20
    else:
        step = 25
    ticks = []
    cur = 0
    while cur <= max_y + 1e-6:
        ticks.append(cur)
        cur += step
    if ticks[-1] < max_y:
        ticks.append(max_y)
    return ticks


def grouped_bar_chart(
    groups: list[str],
    series: list[tuple[str, str, list[float]]],
    title: str,
    subtitle: str,
    y_label: str = "%",
    max_y: float | None = 100,
    lower_is_better: bool = False,
) -> str:
    width, height = 1040, 500
    left, right, top, bottom = 82, 185, 74, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    actual_max = max(max(vals) for _, _, vals in series) if series else 1
    if max_y is None:
        max_y = max(1, math.ceil(actual_max / 10.0) * 10)
    max_y = max(max_y, actual_max)
    group_w = plot_w / len(groups)
    bar_gap = 7
    bar_w = min(34, (group_w - 28 - bar_gap * (len(series) - 1)) / len(series))

    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        svg_text(18, 28, title, "chart-title", "start"),
        svg_text(18, 51, subtitle, "chart-subtitle", "start"),
    ]

    for tick in y_ticks(max_y):
        y = top + plot_h - (tick / max_y) * plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" />')
        label = f"{tick:.0f}{y_label}" if y_label == "%" else f"{tick:.0f}"
        parts.append(svg_text(left - 10, y + 4, label, "axis-tick", "end"))

    parts.append(f'<line class="axis" x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}" />')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" />')

    for gi, group in enumerate(groups):
        gx = left + gi * group_w + group_w / 2
        parts.append(svg_text(gx, top + plot_h + 34, group, "x-label"))
        start_x = gx - (len(series) * bar_w + (len(series) - 1) * bar_gap) / 2
        for si, (_, color, values) in enumerate(series):
            value = values[gi]
            bar_h = (value / max_y) * plot_h
            x = start_x + si * (bar_w + bar_gap)
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'rx="4" fill="{color}" />'
            )
            label = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
            parts.append(svg_text(x + bar_w / 2, max(y - 7, top + 12), label, "bar-label"))

    legend_x, legend_y = width - right + 28, top + 8
    for i, (name, color, _) in enumerate(series):
        y = legend_y + i * 26
        parts.append(f'<rect x="{legend_x}" y="{y}" width="13" height="13" rx="3" fill="{color}" />')
        parts.append(svg_text(legend_x + 21, y + 11, name, "legend", "start"))
    if lower_is_better:
        parts.append(svg_text(legend_x, height - 22, "Lower is better", "chart-note", "start"))

    parts.append("</svg>")
    return "\n".join(parts)


def small_line_chart(
    title: str,
    x_labels: list[str],
    series: list[tuple],
    max_y: float = 100,
    show_last_labels: bool = True,
) -> str:
    width, height = 520, 310
    left, right, top, bottom = 48, 104, 48, 52
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(i: int) -> float:
        if len(x_labels) == 1:
            return left + plot_w / 2
        return left + (i / (len(x_labels) - 1)) * plot_w

    parts = [
        f'<svg class="small-chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        svg_text(16, 26, title, "chart-title small", "start"),
    ]
    for tick in y_ticks(max_y):
        y = top + plot_h - (tick / max_y) * plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" />')
        parts.append(svg_text(left - 8, y + 4, f"{tick:.0f}%", "axis-tick", "end"))
    parts.append(f'<line class="axis" x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}" />')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" />')

    for i, label in enumerate(x_labels):
        parts.append(svg_text(x_pos(i), top + plot_h + 31, label, "x-label small"))
    for item in series:
        name, color, values = item[:3]
        dash = item[3] if len(item) > 3 else ""
        dash_attr = f' stroke-dasharray="{esc(dash)}"' if dash else ""
        points = [(x_pos(i), top + plot_h - (value / max_y) * plot_h) for i, value in enumerate(values)]
        point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"{dash_attr} />')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />')
        if show_last_labels:
            last_x, last_y = points[-1]
            parts.append(svg_text(last_x + 8, last_y + 4, f"{values[-1]:.0f}", "bar-label", "start"))

    legend_x, legend_y = width - right + 22, top + 6
    for i, item in enumerate(series):
        name, color = item[:2]
        dash = item[3] if len(item) > 3 else ""
        dash_attr = f' stroke-dasharray="{esc(dash)}"' if dash else ""
        y = legend_y + i * 21
        parts.append(f'<line x1="{legend_x}" x2="{legend_x + 16}" y1="{y + 7}" y2="{y + 7}" stroke="{color}" stroke-width="3"{dash_attr} />')
        parts.append(f'<circle cx="{legend_x + 8}" cy="{y + 7}" r="3.5" fill="{color}" />')
        parts.append(svg_text(legend_x + 23, y + 11, name, "legend small", "start"))
    parts.append("</svg>")
    return "\n".join(parts)


def small_grouped_bar_chart(
    title: str,
    groups: list[str],
    series: list[tuple[str, str, list[float]]],
    max_y: float = 100,
) -> str:
    width, height = 520, 310
    left, right, top, bottom = 48, 104, 48, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / len(groups)
    bar_gap = 3
    bar_w = min(22, (group_w - 16 - bar_gap * (len(series) - 1)) / len(series))
    parts = [
        f'<svg class="small-chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        svg_text(16, 26, title, "chart-title small", "start"),
    ]
    for tick in y_ticks(max_y):
        y = top + plot_h - (tick / max_y) * plot_h
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" />')
        parts.append(svg_text(left - 8, y + 4, f"{tick:.0f}%", "axis-tick", "end"))
    parts.append(f'<line class="axis" x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}" />')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" />')

    for gi, group in enumerate(groups):
        gx = left + gi * group_w + group_w / 2
        parts.append(svg_text(gx, top + plot_h + 31, group, "x-label small"))
        start_x = gx - (len(series) * bar_w + (len(series) - 1) * bar_gap) / 2
        for si, (_, color, values) in enumerate(series):
            value = values[gi]
            bar_h = (value / max_y) * plot_h
            x = start_x + si * (bar_w + bar_gap)
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}" />')
    legend_x, legend_y = width - right + 18, top + 4
    for i, (name, color, _) in enumerate(series):
        y = legend_y + i * 19
        parts.append(f'<rect x="{legend_x}" y="{y}" width="11" height="11" rx="2" fill="{color}" />')
        parts.append(svg_text(legend_x + 17, y + 10, name, "legend small", "start"))
    parts.append("</svg>")
    return "\n".join(parts)


def line_chart(
    x_labels: list[str],
    series: list[tuple[str, str, list[float]]],
    title: str,
    subtitle: str,
    y_label: str = "%",
    max_y: float | None = 100,
    lower_is_better: bool = False,
) -> str:
    width, height = 1040, 500
    left, right, top, bottom = 82, 190, 74, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    actual_max = max(max(vals) for _, _, vals in series) if series else 1
    if max_y is None:
        max_y = max(1, math.ceil(actual_max / 10.0) * 10)
    max_y = max(max_y, actual_max)

    def x_pos(i: int) -> float:
        if len(x_labels) == 1:
            return left + plot_w / 2
        return left + (i / (len(x_labels) - 1)) * plot_w

    def y_pos(value: float) -> float:
        return top + plot_h - (value / max_y) * plot_h

    parts = [
        f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        svg_text(18, 28, title, "chart-title", "start"),
        svg_text(18, 51, subtitle, "chart-subtitle", "start"),
    ]
    for tick in y_ticks(max_y):
        y = y_pos(tick)
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" />')
        label = f"{tick:.0f}{y_label}" if y_label == "%" else f"{tick:.0f}"
        parts.append(svg_text(left - 10, y + 4, label, "axis-tick", "end"))
    parts.append(f'<line class="axis" x1="{left}" x2="{width-right}" y1="{top+plot_h}" y2="{top+plot_h}" />')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" />')
    for i, label in enumerate(x_labels):
        parts.append(svg_text(x_pos(i), top + plot_h + 34, label, "x-label"))

    for name, color, values in series:
        points = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" />')
        for i, value in enumerate(values):
            x = x_pos(i)
            y = y_pos(value)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#fff" stroke="{color}" stroke-width="3" />')
            label = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
            parts.append(svg_text(x, y - 13, label, "bar-label"))

    legend_x, legend_y = width - right + 28, top + 8
    for i, (name, color, _) in enumerate(series):
        y = legend_y + i * 26
        parts.append(f'<line x1="{legend_x}" x2="{legend_x+18}" y1="{y+7}" y2="{y+7}" stroke="{color}" stroke-width="4" stroke-linecap="round" />')
        parts.append(f'<circle cx="{legend_x+9}" cy="{y+7}" r="4" fill="#fff" stroke="{color}" stroke-width="3" />')
        parts.append(svg_text(legend_x + 28, y + 11, name, "legend", "start"))
    if lower_is_better:
        parts.append(svg_text(legend_x, height - 22, "Lower is better", "chart-note", "start"))

    parts.append("</svg>")
    return "\n".join(parts)


def metric_card(label: str, value: str, caption: str, cls: str = "") -> str:
    class_attr = f" metric {cls}" if cls else " metric"
    return (
        f'<div class="{class_attr}">'
        f'<div class="metric-label">{esc(label)}</div>'
        f'<div class="metric-value">{esc(value)}</div>'
        f'<div class="metric-caption">{caption}</div>'
        "</div>"
    )


def image_panel(path: str, alt: str, caption: str = "") -> str:
    cap = f'<div class="caption">{caption}</div>' if caption else ""
    return (
        '<figure class="media-panel">'
        f'<img src="{esc(path)}" alt="{esc(alt)}">'
        f"{cap}"
        "</figure>"
    )


def video_panel(path: str, label: str, cls: str = "") -> str:
    class_attr = f" video-panel {cls}" if cls else " video-panel"
    return (
        f'<figure class="{class_attr}">'
        f'<div class="video-label">{esc(label)}</div>'
        f'<video controls muted playsinline preload="metadata" src="{esc(path)}"></video>'
        "</figure>"
    )


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<table class="result-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def composition_slide_html() -> str:
    primary_datasets = int(TRAIN_PRIMARY_SUMMARY["completed_dataset_count"])
    all_train_datasets = int(TRAIN_ALL_SUMMARY["completed_dataset_count"])
    high_budget_datasets = all_train_datasets - primary_datasets
    train_steps = int(TRAIN_ALL_SUMMARY["total_samples"])
    train_images = train_steps * 2
    id_datasets = int(EVAL_ID_SUMMARY["completed_dataset_count"])
    ood_datasets = int(EVAL_OOD_SUMMARY["completed_dataset_count"])
    eval_steps = int(EVAL_ID_SUMMARY["total_samples"]) + int(EVAL_OOD_SUMMARY["total_samples"])
    eval_images = eval_steps * 2
    cards = "".join(
        [
            metric_card("Visual axes", "4", "color, spatial, camera, lighting"),
            metric_card("Tasks", "2", "direct reach and obstacle-aware reach"),
            metric_card("Train configs", "16", "4 axes x 2 variants x 2 tasks"),
            metric_card("Primary train set", int_fmt(primary_datasets), "budgets 5/20/50 x seeds 0/1/2"),
            metric_card("High-budget extension", f"+{high_budget_datasets}", "budgets 100/200, seed 0"),
            metric_card("Train images", int_fmt(train_images), f"{int_fmt(train_steps)} timesteps x 2 cameras"),
            metric_card("ID eval datasets", int_fmt(id_datasets), "same distributions, held-out seeds 200/201/202"),
            metric_card("OOD eval datasets", int_fmt(ood_datasets), "held-out variants, seeds 100/101/102"),
        ]
    )
    pipeline_steps = [
        (
            "1",
            "Choose config",
            "task, visual axis, variant, budget, seed, and train/ID/OOD split",
        ),
        (
            "2",
            "Sample scene",
            "cube/obstacle placement plus color, camera, spatial, or light parameters",
        ),
        (
            "3",
            "Roll out expert",
            "PyBullet expert moves from sampled start state through reach or obstacle-aware reach",
        ),
        (
            "4",
            "Save demo",
            "actions, robot state, scene metadata, external RGB, and wrist RGB at each timestep",
        ),
    ]
    pipeline_html = "".join(
        f"""
        <div class="pipeline-step">
          <div class="pipeline-num">{num}</div>
          <strong>{esc(title)}</strong>
          <span>{esc(detail)}</span>
        </div>
        """
        for num, title, detail in pipeline_steps
    )
    return f"""
    <div class="composition-layout">
      <div class="metric-grid four composition-cards">{cards}</div>
      <div class="collection-pipeline">
        <div class="pipeline-title">Dataset collection pipeline</div>
        <div class="pipeline-steps">{pipeline_html}</div>
      </div>
      <div class="composition-bottom">
        <div class="callout"><strong>Observation:</strong> 128 x 128 RGB from two cameras: external view plus wrist/end-effector view. Each recorded timestep contributes two raw image frames.</div>
        <div class="callout"><strong>Closed-loop evaluation:</strong> 400-step rollout horizon, precision reported as success@1/2/5 cm from the closest approach to the cube.</div>
        <div class="callout"><strong>Eval image pool:</strong> {int_fmt(id_datasets + ood_datasets)} held-out scene datasets, {int_fmt(eval_steps)} timesteps, {int_fmt(eval_images)} dual-camera RGB images.</div>
      </div>
    </div>
    """


def load_raw_image_records() -> list[dict[str, object]]:
    path = ANALYSIS / "dataset_slides/raw_image_manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("records", [])


def dataset_raw_grid(records: list[dict[str, object]], section: str, columns: int) -> str:
    rows = []
    for axis in RAW_AXIS_ORDER:
        meta = RAW_AXIS_META[axis]
        axis_records = [record for record in records if record["section"] == section and record["axis"] == axis]
        if section.startswith("trajectories_"):
            axis_records.sort(key=lambda record: int(record["step_index"]))
        frame_cards = []
        last_index = len(axis_records) - 1
        for i, record in enumerate(axis_records):
            path = rel(ANALYSIS / "dataset_slides" / str(record["path"]))
            step = int(record["step_index"])
            ep = int(record["episode_index"])
            if section == "overview":
                label = f"episode {ep:03d}"
                sub = str(record["descriptor"])
            elif i == 0:
                label = "start"
                sub = f"step {step}"
            elif i == last_index:
                label = "end"
                sub = f"step {step}"
            else:
                label = f"step {step}"
                sub = f"frame {i + 1}"
            frame_cards.append(
                f"""
                <figure class="raw-card">
                  <img src="{esc(path)}" alt="{esc(meta['label'])} {esc(label)}">
                  <figcaption><strong>{esc(label)}</strong><span>{esc(sub)}</span></figcaption>
                </figure>
                """
            )
        rows.append(
            f"""
            <div class="raw-axis-row" style="--axis-color:{esc(meta['color'])}">
              <div class="raw-axis-label">
                <strong>{esc(meta['label'])}</strong>
                <span>{esc(meta['description'])}</span>
              </div>
              <div class="raw-strip raw-cols-{columns}">
                {''.join(frame_cards)}
              </div>
            </div>
            """
        )
    return f'<div class="raw-dataset-grid">{"".join(rows)}</div>'


def bucket_aggregate(group: str, tag: str, split: str) -> dict[str, dict[str, float]]:
    accum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in BUCKETS:
        if row["group"] != group or row["tag"] != tag or row["split"] != split:
            continue
        if not row["bucket"].startswith("all_"):
            continue
        n = fnum(row, "rollouts")
        bucket = row["bucket_group"]
        accum[bucket]["rollouts"] += n
        for key in ["success_1cm", "success_2cm", "success_5cm"]:
            accum[bucket][key] += fnum(row, key) * n
    for bucket, vals in accum.items():
        n = vals["rollouts"]
        for key in ["success_1cm", "success_2cm", "success_5cm"]:
            vals[key] /= n
    return accum


def slide(kicker: str, title: str, body: str, transcript: str = "", extra_class: str = "") -> str:
    cls = f"slide {extra_class}".strip()
    notes = f'<aside class="transcript"><strong>Transcript.</strong> {transcript}</aside>' if transcript else ""
    return (
        f'<section class="{cls}">\n'
        f'  <div class="kicker">{kicker}</div>\n'
        f'  <h1>{title}</h1>\n'
        f'  <div class="slide-body">{body}</div>\n'
        f'  {notes}\n'
        f'</section>'
    )


def build() -> str:
    slides: list[str] = []

    raw_image_records = load_raw_image_records()
    fail_video = rel(
        ANALYSIS
        / "slide_videos/slide15_visual_only_failure_h264.mp4"
    )
    success_video = rel(
        ANALYSIS
        / "slide_videos/slide15_phase_geometry_success_h264.mp4"
    )
    axis_success_videos = {
        "color": rel(ANALYSIS / "slide_videos/axis_success_720p/avoid_color_success_720p_h264.mp4"),
        "spatial": rel(ANALYSIS / "slide_videos/axis_success_720p/avoid_spatial_success_720p_h264.mp4"),
        "camera": rel(ANALYSIS / "slide_videos/axis_success_720p/avoid_camera_success_720p_h264.mp4"),
        "lighting": rel(ANALYSIS / "slide_videos/axis_success_720p/avoid_lighting_success_720p_h264.mp4"),
    }

    visual_edge = structured_row("Edge-balanced visual-only", "Scratch CNN", "OOD")
    phase_geo = structured_row("Edge-balanced phase+geometry", "Scratch CNN", "OOD")

    slides.append(
        slide(
            "Course project",
            "What Visual Diversity Matters for Closed-Loop Behavior Cloning Under Distribution Shift?",
            """
            <div class="simple-title">
              <p class="subtitle">A Controlled Study of Pixel-Based Robot Manipulation Under Distribution Shift</p>
              <div class="title-meta">
                <div>
                  <span>Students</span>
                  <strong>Aninditaa Chauhan &middot; Jai Kumar Sharma &middot; Manas Ganti &middot; Shakir Farhan Mohammed</strong>
                </div>
                <div>
                  <span>Subject</span>
                  <strong>Human-Robot Interaction</strong>
                </div>
              </div>
            </div>
            """,
            "This is the opening title slide. It states the project name, the student team, and the course subject.",
            "title-slide",
        )
    )

    slides.append(
        slide(
            "Problem statement",
            "Pixel imitation policies must work when the robot sees a different world than the demonstrations.",
            r"""
            <div class="eval-layout problem-statement">
              <div class="eval-card human">
                <div class="eval-kicker">Human + robot setting</div>
                <h2>A human provides demonstrations; the robot must execute them in new scenes.</h2>
                <ul>
                  <li>Expert demonstrations specify how a Panda arm should reach a cube or route around an obstacle.</li>
                  <li>The learned policy receives only dual-camera RGB and robot state at test time.</li>
                  <li>The practical question is whether the robot still reaches when appearance, viewpoint, lighting, or target position changes.</li>
                </ul>
              </div>
              <div class="eval-card missing">
                <div class="eval-kicker">What existing BC misses</div>
                <h2>Offline pixel BC can fit demonstrations but fail under closed-loop visual shift.</h2>
                <ul>
                  <li>Standard behavior cloning minimizes action error on the dataset, not recovery during rollouts.</li>
                  <li>Frozen pretrained visual encoders may not transfer cleanly from real images to simulated manipulation pixels.</li>
                  <li>Prior comparisons often do not isolate which visual diversity axis actually changes control performance.</li>
                </ul>
              </div>
              <div class="eval-card question">
                <div class="eval-kicker">Research question</div>
                <h2>Which visual diversity matters for OOD closed-loop manipulation?</h2>
                <ul>
                  <li>We isolate color, spatial distribution, camera viewpoint, and lighting direction/intensity.</li>
                  <li>We compare scratch CNN, frozen ResNet, and partially fine-tuned ResNet policies.</li>
                  <li>We evaluate ID and OOD rollouts using strict reaching success at 1, 2, and 5 cm.</li>
                </ul>
              </div>
            </div>
            <div class="eval-formula problem-formula">
              <div class="math-block">\[
                a_t = \pi_\theta(I_t^{ext}, I_t^{wrist}, x_t^{ee}), \qquad
                \theta^* = \arg\min_\theta \sum_t \lVert a_t^* - \pi_\theta(o_t) \rVert_2^2
              \]</div>
              <div class="math-block">\[
                S@r = \mathbb{1}\left[\min_t \lVert x_t^{ee} - x_{goal} \rVert_2 \le r\right],
                \quad r \in \{1,2,5\}\text{ cm}
              \]</div>
            </div>
            """,
            "This problem statement is aligned with the rubric. It explains the human and robot components, the motivation for studying visual distribution shift, what standard pixel behavior cloning and pretrained encoders miss, and the formal policy and success metrics used in the project.",
        )
    )

    slides.append(
        slide(
            "Dataset composition",
            "The benchmark is an axis-isolated visual imitation dataset.",
            composition_slide_html(),
            "This slide defines the dataset scale and experimental grid: four visual axes, two tasks, dual 128 pixel camera inputs, ID and OOD eval splits, and the number of datasets and images used.",
        )
    )

    slides.append(
        slide(
            "Dataset evidence 1/3",
            "The collected obstacle-aware dataset visibly varies each visual axis.",
            f"""
            <div class="raw-slide">
              {dataset_raw_grid(raw_image_records, "overview", 4)}
            </div>
            """,
            "This slide is the visual proof that the dataset is diverse. It uses individual raw PNGs from the dataset export, not a precomposed figure.",
        )
    )

    slides.append(
        slide(
            "Dataset evidence 2/3",
            "Full obstacle-aware trajectories from the external camera.",
            f"""
            <div class="raw-slide">
              {dataset_raw_grid(raw_image_records, "trajectories_external_rgb", 8)}
            </div>
            """,
            "This shows the temporal part of the dataset using raw external-camera frames. Each row follows one obstacle-aware demonstration from start to end.",
        )
    )

    slides.append(
        slide(
            "Dataset evidence 3/3",
            "The same trajectories from the wrist camera used by the policy.",
            f"""
            <div class="raw-slide">
              {dataset_raw_grid(raw_image_records, "trajectories_eef_rgb", 8)}
            </div>
            """,
            "The policy observes both camera streams. This slide uses raw wrist-camera frames for the same selected demonstrations.",
        )
    )

    slides.append(
        slide(
            "Model policies",
            "We compare three visual behavior-cloning policies.",
            """
            <div class="policy-layout">
              <div class="policy-card scratch">
                <div class="policy-kicker">Scratch CNN</div>
                <h2>Train the visual encoder from simulation pixels.</h2>
                <p>Small convolutional encoder over external + wrist RGB, concatenated with robot state, then an MLP policy head predicts continuous end-effector action.</p>
                <ul>
                  <li>No pretrained image features.</li>
                  <li>All CNN and policy-head weights trained end-to-end.</li>
                  <li>Baseline for whether the simulated dataset itself is enough.</li>
                </ul>
              </div>
              <div class="policy-card frozen">
                <div class="policy-kicker">Frozen ResNet-18</div>
                <h2>Use ImageNet features without adapting the backbone.</h2>
                <p>External + wrist RGB are encoded by a ResNet-18 backbone initialized from ImageNet; the backbone is frozen and only the projection/policy head is trained.</p>
                <ul>
                  <li>Tests whether pretrained visual features transfer to simulation control.</li>
                  <li>Lower trainable parameter count.</li>
                  <li>Useful contrast for sim-real feature mismatch.</li>
                </ul>
              </div>
              <div class="policy-card partial">
                <div class="policy-kicker">Partial ResNet-18</div>
                <h2>Adapt the later visual features to the simulated task.</h2>
                <p>Early ResNet layers stay frozen; later residual blocks plus the projection/policy head are fine-tuned on the demonstration data.</p>
                <ul>
                  <li>Middle ground between fixed representation and full scratch learning.</li>
                  <li>Tests whether limited adaptation improves OOD generalization.</li>
                  <li>Same BC objective and rollout evaluation as the other policies.</li>
                </ul>
              </div>
            </div>
            <div class="policy-common">
              <div><strong>Shared training:</strong> supervised behavior cloning on expert actions with MSE loss.</div>
              <div><strong>Shared inputs:</strong> 128x128 external RGB + wrist RGB + robot state.</div>
              <div><strong>Shared eval:</strong> 400-step closed-loop rollouts, success@1/2/5 cm from the closest approach.</div>
            </div>
            """,
            "After defining the dataset, this slide defines the three policies we compare: a scratch CNN, a frozen pretrained ResNet, and a partially fine-tuned ResNet. All use the same behavior-cloning objective and evaluation protocol.",
        )
    )

    slides.append(
        slide(
            "Evaluation protocol",
            "We evaluate whether the closed-loop policy reaches the cube under held-out scenes.",
            """
            <div class="eval-layout">
              <div class="eval-card">
                <div class="eval-kicker">Rollout</div>
                <h2>Closed-loop control, not offline prediction.</h2>
                <ul>
                  <li>Reset PyBullet to a held-out scene.</li>
                  <li>Run the learned policy for up to 400 control steps.</li>
                  <li>At every step, feed external RGB, wrist RGB, and robot state back into the policy.</li>
                  <li>Record the closest distance ever reached to the cube target.</li>
                </ul>
              </div>
              <div class="eval-card">
                <div class="eval-kicker">Splits</div>
                <h2>ID tests memorized distributions; OOD tests visual shift.</h2>
                <ul>
                  <li>ID uses the same train distributions with held-out eval seeds.</li>
                  <li>OOD uses held-out variants for the visual axis being tested.</li>
                  <li>Metrics are aggregated across tasks, axes, demo budgets, and model families depending on the result slide.</li>
                </ul>
              </div>
              <div class="eval-card">
                <div class="eval-kicker">Success metrics</div>
                <h2>Success is thresholded by closest approach.</h2>
                <ul>
                  <li><strong>S@1 cm:</strong> strict precision reaching.</li>
                  <li><strong>S@2 cm:</strong> medium tolerance.</li>
                  <li><strong>S@5 cm:</strong> broad reaching, useful but too loose alone.</li>
                  <li>The metric asks whether the policy reaches the cube at any point during the rollout.</li>
                </ul>
              </div>
            </div>
            <div class="eval-formula">
              <code>d_best = min_t || ee_t - target ||</code>
              <code>S@r = 1[d_best <= r], r in {1cm, 2cm, 5cm}</code>
            </div>
            """,
            "Before reading the result plots, this slide defines evaluation. Each policy is rolled out in closed loop for up to 400 steps, and success is computed from the closest approach to the cube at one, two, and five centimeter thresholds.",
        )
    )

    demo_budgets = [20, 50, 100, 200]
    short_families = [
        ("scratch_cnn", "Scratch", "#2f6fad"),
        ("frozen_resnet18", "Frozen", "#26845a"),
        ("partial_resnet18", "Partial", "#c66a2b"),
    ]
    threshold_specs = [
        ("S@1", "success_at_1cm_pct"),
        ("S@2", "success_at_2cm_pct"),
        ("S@5", "success_at_5cm_pct"),
    ]
    split_series = [
        ("S ID", "#8ab7e8", "scratch_cnn", "id"),
        ("S OOD", "#2f6fad", "scratch_cnn", "ood"),
        ("F ID", "#82c9a6", "frozen_resnet18", "id"),
        ("F OOD", "#26845a", "frozen_resnet18", "ood"),
        ("P ID", "#e7ad76", "partial_resnet18", "id"),
        ("P OOD", "#c66a2b", "partial_resnet18", "ood"),
    ]
    task_series_specs = [
        ("S reach", "#8ab7e8", "scratch_cnn", "reach"),
        ("S obs", "#2f6fad", "scratch_cnn", "avoid_reach"),
        ("F reach", "#82c9a6", "frozen_resnet18", "reach"),
        ("F obs", "#26845a", "frozen_resnet18", "avoid_reach"),
        ("P reach", "#e7ad76", "partial_resnet18", "reach"),
        ("P obs", "#c66a2b", "partial_resnet18", "avoid_reach"),
    ]

    scaling_panels = []
    for axis, axis_name in AXES:
        series = []
        for family, label, color in short_families:
            series.append(
                (
                    f"{label} @1",
                    color,
                    [axis_split_slide_value(family, "ood", b, axis, "success_at_1cm_pct") for b in demo_budgets],
                )
            )
            series.append(
                (
                    f"{label} @5",
                    color,
                    [axis_split_slide_value(family, "ood", b, axis, "success_at_5cm_pct") for b in demo_budgets],
                    "6 5",
                )
            )
        scaling_panels.append(
            f'<div class="small-chart-card">{small_line_chart(axis_name, [str(b) for b in demo_budgets], series, show_last_labels=False)}</div>'
        )
    slides.append(
        slide(
            "Result 1 - Visual axes over demos",
            "OOD success changes differently across visual axes as the demo budget increases.",
            f"""
            <div class="small-multiple-grid four">{''.join(scaling_panels)}</div>
            <div class="callout">OOD, aggregated across reach and obstacle-aware tasks. Each panel is one visual axis; x-axis is demo budget 20, 50, 100, 200. Solid lines are S@1 cm; dotted lines are S@5 cm. Budget-200 obstacle-aware values use the final phase-precise avoid metrics.</div>
            """,
            "This comparison shows how each visual axis scales with more demonstrations under both the strict 1 cm metric and the broader 5 cm reaching metric.",
        )
    )

    precision_panels = []
    for axis, axis_name in AXES:
        series = [
            (
                label.replace(" ResNet-18", ""),
                color,
                [axis_split_slide_value(family, "ood", 200, axis, key) for _, key in threshold_specs],
            )
            for family, label, color in short_families
        ]
        precision_panels.append(
            f'<div class="small-chart-card">{small_grouped_bar_chart(axis_name, [name for name, _ in threshold_specs], series)}</div>'
        )
    slides.append(
        slide(
            "Result 2 - Visual axes by precision threshold",
            "The same axis can look solved at 5 cm while still failing at 1 cm.",
            f"""
            <div class="small-multiple-grid four">{''.join(precision_panels)}</div>
            <div class="callout">OOD, budget 200, aggregated across both tasks. The obstacle-aware side uses the final phase-precise avoid metrics; bars compare Scratch CNN, Frozen ResNet, and Partial ResNet.</div>
            """,
            "This slide shows why the report keeps 1, 2, and 5 cm. A broad success threshold can hide poor precision.",
        )
    )

    split_panels = []
    for threshold_name, metric in threshold_specs:
        series = [
            (
                name,
                color,
                [axis_split_slide_value(family, split, 200, axis, metric) for axis, _ in AXES],
            )
            for name, color, family, split in split_series
        ]
        split_panels.append(
            f'<div class="small-chart-card">{small_grouped_bar_chart(threshold_name, [label for _, label in AXES], series)}</div>'
        )
    slides.append(
        slide(
            "Result 3 - ID versus OOD by visual axis",
            "The ID/OOD gap is strongest where visual variation changes control geometry.",
            f"""
            <div class="small-multiple-grid three">{''.join(split_panels)}</div>
            <div class="callout">Budget 200, aggregated across both tasks. The obstacle-aware side uses the final phase-precise avoid metrics. S/F/P denote Scratch CNN, Frozen ResNet, and Partial ResNet; lighter bars are ID and darker bars are OOD.</div>
            """,
            "This slide separates in-distribution reaching from out-of-distribution reaching at all three thresholds.",
        )
    )

    task_panels = []
    for axis, axis_name in AXES:
        series = [
            (
                name,
                color,
                [task_split_slide_value(family, "ood", 200, axis, task, key) for _, key in threshold_specs],
            )
            for name, color, family, task in task_series_specs
        ]
        task_panels.append(
            f'<div class="small-chart-card">{small_grouped_bar_chart(axis_name, [name for name, _ in threshold_specs], series)}</div>'
        )
    slides.append(
        slide(
            "Result 4 - Task split by visual axis",
            "Obstacle-aware reaching is the task where spatial variation becomes the hardest.",
            f"""
            <div class="small-multiple-grid four">{''.join(task_panels)}</div>
            <div class="callout">OOD, budget 200. Reach bars use the all-budget aggregate; obstacle-aware bars use the final phase-precise avoid evaluation metrics for each model family.</div>
            """,
            "This slide introduces the non-obvious failure: the visual axis and the task interact, with obstacle-aware spatial reaching becoming the bottleneck.",
        )
    )

    slides.append(
        slide(
            "Obstacle-aware OOD examples",
            "Successful rollouts across all four visual diversity axes.",
            f"""
            <div class="axis-video-grid">
              {video_panel(axis_success_videos["color"], "Color shift success - S@1 cm", "success")}
              {video_panel(axis_success_videos["spatial"], "Spatial shift success - S@1 cm", "success")}
              {video_panel(axis_success_videos["camera"], "Camera shift success - S@2 cm", "success")}
              {video_panel(axis_success_videos["lighting"], "Lighting shift success - S@1 cm", "success")}
            </div>
            """,
            "This slide shows successful obstacle-aware OOD rollouts for the four visual diversity axes.",
        )
    )

    def axis_chart_for_budget(budget: int) -> str:
        axis_groups = [label for _, label in AXES]
        series = []
        for family, label, color in FAMILIES:
            series.append(
                (
                    label,
                    color,
                    [axis_split_slide_value(family, "ood", budget, axis, "success_at_1cm_pct") for axis, _ in AXES],
                )
            )
        return grouped_bar_chart(axis_groups, series, f"OOD axis comparison at budget {budget}", "Success at 1 cm by visual diversity axis", "%", 70)

    slides.append(
        slide(
            "Result 5 - Axis ranking at 50 and 200 demos",
            "Higher budget helps appearance axes, but spatial remains hard.",
            f"""
            <div class="two-col wide">
              <div class="chart-card">{axis_chart_for_budget(50)}</div>
              <div class="chart-card">{axis_chart_for_budget(200)}</div>
            </div>
            <div class="callout">At 50 demos, spatial OOD is only {pct(axis_split_slide_value("scratch_cnn", "ood", 50, "spatial_distribution", "success_at_1cm_pct"))} at 1 cm for scratch CNN. At 200 demos, color, camera, and lighting improve, but spatial remains low at {pct(axis_split_slide_value("scratch_cnn", "ood", 200, "spatial_distribution", "success_at_1cm_pct"))}.</div>
            """,
            "This compares the original 50-demo regime with the 200-demo high-budget regime. More data improves appearance shifts, but spatial distribution remains the bottleneck.",
        )
    )

    slides.append(
        slide(
            "Method bridge",
            "To address spatial OOD, we add phase + geometry to the visual BC policy.",
            """
            <div class="method-grid">
              <div class="method-card problem">
                <div class="method-kicker">1. Diagnose the failure</div>
                <h2>Spatial OOD is a routing problem, not just an image problem.</h2>
                <ul>
                  <li>We isolated axes while keeping policy, camera inputs, BC loss, and rollout metric fixed.</li>
                  <li>Color, camera, and lighting improve with budget; spatial remains low at strict S@1 cm.</li>
                  <li>The hard subset is obstacle-aware spatial reaching: the policy must route around an obstacle before descending to the cube.</li>
                  <li>So the next test asks whether the policy is missing task-stage and relation signals, not whether it needs a new camera or a new loss.</li>
                </ul>
              </div>
              <div class="method-card">
                <div class="method-kicker">2. Add phase labels</div>
                <h2>Expose which part of the route the demonstrator is executing.</h2>
                <ul>
                  <li>We replay the deterministic avoid-reach expert logic on each stored state.</li>
                  <li>Each timestep gets a phase label: <code>side_align</code>, <code>cube_hover</code>, or <code>final_descent</code>.</li>
                  <li>Example: if the end-effector is still moving to the safe side of the obstacle, the input phase is <code>side_align</code>; once it is above the cube, it becomes <code>cube_hover</code>.</li>
                  <li>The phase is encoded as a 3D one-hot vector and concatenated into the policy input.</li>
                  <li>This separates route selection from fine reaching, which is hidden if every avoid-reach image is treated as the same behavior mode.</li>
                </ul>
              </div>
              <div class="method-card">
                <div class="method-kicker">3. Add geometry features</div>
                <h2>Expose the spatial relations the pixels were not providing reliably.</h2>
                <ul>
                  <li>We compute a compact 18D structured vector at each timestep.</li>
                  <li>It contains phase one-hot, target xyz, obstacle center xyz, and obstacle half-extents xyz.</li>
                  <li>It also includes target-minus-end-effector and target-minus-obstacle vectors for relative control geometry.</li>
                  <li>Example: target=(x,y,z), obstacle=(x_o,y_o,z_o), ee=(x_e,y_e,z_e); the policy receives target-ee and target-obstacle deltas directly.</li>
                  <li>The policy head therefore sees where to move and how the target is positioned relative to the obstacle boundary.</li>
                </ul>
              </div>
              <div class="method-card">
                <div class="method-kicker">4. Run a controlled comparison</div>
                <h2>Change the missing state signal, not the whole experiment.</h2>
                <ul>
                  <li>Same 128px external RGB + wrist RGB, robot state, MSE behavior cloning, and 400-step closed-loop evaluation.</li>
                  <li>Train visual-only, phase-only, target-only, geometry-only, and phase+geometry variants on the hard spatial obstacle-aware setting.</li>
                  <li>Compare S@1/S@2/S@5 on ID and OOD rollouts to test whether explicit structure fixes the bottleneck.</li>
                  <li>If full phase+geometry wins, the result supports a missing-state explanation for the spatial failure.</li>
                </ul>
              </div>
            </div>
            <div class="method-footer">
              <div>
                <strong>Policy input change</strong>
                <code>a_t = pi(I_ext, I_wrist, q_t, z_struct)</code>
              </div>
              <div>
                <strong>Structured vector</strong>
                <code>z_struct = [phase_3, target_3, obstacle_center_3, obstacle_size_3, target-ee_3, target-obstacle_3]</code>
              </div>
              <div>
                <strong>Diagnostic comparison</strong>
                <span>visual-only vs phase-only vs target-only vs geometry-only vs phase+geometry, evaluated with S@1/S@2/S@5 on ID and OOD rollouts.</span>
              </div>
            </div>
            """,
            "After showing that spatial remains hard, this slide introduces the intervention. We keep the visual behavior cloning setup fixed and add a compact structured state containing the route phase and target-obstacle geometry.",
        )
    )

    bucket_visual = bucket_aggregate("Edge-balanced visual-only", "scratch", "OOD")
    bucket_phase = bucket_aggregate("Edge-balanced phase+geometry", "scratch", "OOD")
    bucket_order = ["corner", "edge", "interior"]
    bucket_series = [
        ("Visual-only S@1", "#c74343", [100 * bucket_visual[b]["success_1cm"] for b in bucket_order]),
        ("Phase+geometry S@1", "#26845a", [100 * bucket_phase[b]["success_1cm"] for b in bucket_order]),
    ]
    structured_series = [
        (
            "Visual-only",
            "#c74343",
            [
                100 * fnum(visual_edge, "success_1cm"),
                100 * fnum(visual_edge, "success_2cm"),
                100 * fnum(visual_edge, "success_5cm"),
            ],
        ),
        (
            "Phase+geometry",
            "#26845a",
            [
                100 * fnum(phase_geo, "success_1cm"),
                100 * fnum(phase_geo, "success_2cm"),
                100 * fnum(phase_geo, "success_5cm"),
            ],
        ),
    ]
    slides.append(
        slide(
            "Result 6 - Spatial diagnostic and phase+geometry gain",
            "Phase + geometry improves the hardest spatial obstacle-aware cases.",
            f"""
            <div class="two-col wide">
              <div class="chart-card">{grouped_bar_chart(["Corners", "Edges", "Interior"], bucket_series, "OOD spatial buckets, scratch policy", "Success at 1 cm by target region", "%", 100)}</div>
              <div class="chart-card">{grouped_bar_chart(["S@1", "S@2", "S@5"], structured_series, "Scratch obstacle-aware spatial OOD", "Edge-balanced visual-only versus phase + geometry", "%", 100)}</div>
            </div>
            <div class="callout">Left: visual-only fails across the edge-balanced spatial OOD buckets, while phase + geometry improves corners, edges, and interior targets. Right: with the same RGB stream and BC objective, adding phase + geometry raises OOD success from {pct(100 * fnum(visual_edge, "success_1cm"))} to {pct(100 * fnum(phase_geo, "success_1cm"))} at 1 cm.</div>
            """,
            "This slide combines the bucket-level diagnostic and the main structured comparison. The bucket plot shows that visual-only is weak across the edge-balanced spatial OOD buckets, and the threshold plot shows that phase plus geometry changes the result.",
        )
    )

    slides.append(
        slide(
            "Result 7 - Rollout evidence",
            "Rollout examples.",
            f"""
            <div class="two-col wide">
              {video_panel(fail_video, "Visual-only spatial OOD failure", "danger")}
              {video_panel(success_video, "Phase + geometry spatial OOD success", "success")}
            </div>
            <div class="callout">The videos use the corrected 480p dual-camera render. The visual-only policy misses on a hard spatial case; phase + geometry reaches the cube.</div>
            """,
            "This slide connects the numbers to behavior. It is the video evidence for the structured diagnostic result.",
        )
    )

    slides.append(
        slide(
            "Class connection - equations",
            "The project is behavior cloning from pixels, extended into a controlled OOD benchmark.",
            r"""
            <div class="equation-layout">
              <div class="equation-panel">
                <div class="equation-kicker">Equations used in the project</div>
                <div class="eq-group">
                  <h2>Policy and observation</h2>
                  <div class="math-block">\[
                    o_t = [I_t^{ext}, I_t^{wrist}, x_t^{ee}], \qquad
                    a_t = \pi_\theta(o_t)
                  \]</div>
                  <p>The policy maps dual-camera RGB plus robot state to a 3D end-effector delta action.</p>
                </div>
                <div class="eq-group">
                  <h2>Offline behavior cloning</h2>
                  <div class="math-block">\[
                    \mathcal{D} = \{(o_t, a_t^*)\}_{t=1}^{N}, \qquad
                    \theta^* = \arg\min_\theta {1 \over N}
                    \sum_{(o,a^*) \in \mathcal{D}}
                    \lVert a^* - \pi_\theta(o) \rVert_2^2
                  \]</div>
                  <p>This is the lecture BC objective, using scripted expert actions as labels.</p>
                </div>
                <div class="eq-group">
                  <h2>Closed-loop success and OOD gap</h2>
                  <div class="math-block">\[
                    d_{best} = \min_{0 \le t \le T}
                    \lVert x_t^{ee} - x_{goal} \rVert_2
                  \]</div>
                  <div class="math-block">\[
                    S@\epsilon = \mathbf{1}[d_{best} \le \epsilon],
                    \quad \epsilon \in \{0.01, 0.02, 0.05\}\,\mathrm{m}
                  \]</div>
                  <div class="math-block">\[
                    Gap_\epsilon =
                    \mathbb{E}[S@\epsilon \mid ID] -
                    \mathbb{E}[S@\epsilon \mid OOD]
                  \]</div>
                </div>
                <div class="eq-group">
                  <h2>Phase + geometry diagnostic</h2>
                  <div class="math-block">\[
                    o'_t = [o_t, p_t, g_t], \qquad
                    g_t = [x_{goal}, c_{obs}, h_{obs},
                    x_{goal}-x_t^{ee}, x_{goal}-c_{obs}]
                  \]</div>
                  <p>We keep the BC loss fixed and add task phase plus target/obstacle geometry.</p>
                </div>
              </div>
              <div class="rubric-panel">
                <div class="rubric-card taught">
                  <div class="equation-kicker">What lecture taught</div>
                  <ul>
                    <li>Parameterized control policy \(a = \pi_\theta(s)\).</li>
                    <li>Expert dataset \(\mathcal{D}\) of state-action pairs.</li>
                    <li>Behavior cloning with MSE loss.</li>
                    <li>OOD/covariate-shift problem in offline imitation.</li>
                    <li>DAgger, online correction, RL/SAC, reward learning, and frozen/fine-tuned visual encoders as broader context.</li>
                  </ul>
                </div>
                <div class="rubric-card applied">
                  <div class="equation-kicker">What we applied</div>
                  <ul>
                    <li>Offline BC from expert demonstrations.</li>
                    <li>Vision-to-action policy with external and wrist RGB.</li>
                    <li>Scratch CNN, frozen ResNet-18, and partial ResNet-18 baselines.</li>
                    <li>ID/OOD closed-loop evaluation with precision thresholds.</li>
                    <li>Reproducible dataset, training, evaluation, and analysis scripts.</li>
                  </ul>
                </div>
                <div class="rubric-card beyond">
                  <div class="equation-kicker">Beyond class / rubric difficulty</div>
                  <ul>
                    <li>Factorized 4-axis visual OOD benchmark: color, spatial, camera, lighting.</li>
                    <li>Two tasks, 128px dual-camera observations, 176 train datasets, and 72 eval datasets.</li>
                    <li>158,400 closed-loop rollouts across models, budgets, axes, tasks, and splits.</li>
                    <li>Spatial failure analysis plus phase+geometry intervention for obstacle-aware reaching.</li>
                    <li>We did not use SAC or reward learning; we studied where pure BC fails and what structure fixes.</li>
                  </ul>
                </div>
              </div>
            </div>
            """,
            "This slide explicitly connects the project to the class and the rubric. The equations are the policy, dataset, behavior cloning objective, closed-loop success metrics, OOD gap, and the phase plus geometry diagnostic. The right side lists what was taught, what we applied, and what went beyond the lectures and homework.",
            "equations-slide",
        )
    )

    return render(slides)


def render(slides: list[str]) -> str:
    style = r"""
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #121721;
  --muted: #606a78;
  --line: #dde3ec;
  --line-strong: #c5cfdd;
  --accent: #1f6f8b;
  --scratch: #2f6fad;
  --frozen: #26845a;
  --partial: #c66a2b;
  --danger: #c74343;
  --success: #26845a;
  --warn-bg: #fff7e6;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  width: 100%;
  height: 100%;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  overflow: hidden;
}
.deck { position: fixed; inset: 0; }
.slide {
  position: absolute;
  inset: 0;
  display: none;
  flex-direction: column;
  gap: 16px;
  padding: 38px 54px 58px;
  background: var(--bg);
}
.slide.active { display: flex; }
.kicker {
  color: var(--accent);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
h1 {
  margin: 0;
  max-width: 1220px;
  font-size: clamp(30px, 3.2vw, 56px);
  line-height: 1.04;
  letter-spacing: 0;
  font-weight: 780;
}
.slide-body {
  flex: 1 1 auto;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.title-slide h1 { font-size: clamp(34px, 4.1vw, 68px); }
.title-slide {
  justify-content: center;
}
.title-slide .kicker {
  font-size: 14px;
}
.simple-title {
  flex: 0 0 auto;
  max-width: 1180px;
  display: grid;
  gap: 40px;
}
.subtitle {
  margin: 0;
  max-width: 980px;
  color: var(--muted);
  font-size: clamp(22px, 2.15vw, 36px);
  line-height: 1.24;
}
.title-meta {
  display: grid;
  grid-template-columns: 1.5fr 0.9fr;
  gap: 18px;
}
.title-meta div {
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 5px solid var(--accent);
  border-radius: 8px;
  padding: 18px 20px;
  display: grid;
  gap: 8px;
}
.title-meta span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.title-meta strong {
  color: #172033;
  font-size: clamp(16px, 1.35vw, 23px);
  line-height: 1.3;
}
.hero-layout {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1.04fr 0.96fr;
  gap: 28px;
  align-items: stretch;
}
.lede {
  max-width: 940px;
  margin: 0 0 22px;
  color: var(--muted);
  font-size: clamp(17px, 1.45vw, 24px);
  line-height: 1.42;
}
.authors {
  margin-top: 20px;
  color: var(--muted);
  font-size: 15px;
}
.two-col {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  align-items: stretch;
}
.two-col.wide { grid-template-columns: 1.12fr 0.88fr; }
.single-figure { flex: 1; min-height: 0; }
.three-grid {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}
.metric-grid {
  display: grid;
  gap: 12px;
}
.metric-grid.two { grid-template-columns: repeat(2, 1fr); }
.metric-grid.four { grid-template-columns: repeat(4, 1fr); }
.composition-layout {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto auto;
  gap: 14px;
}
.composition-cards {
  min-height: 0;
}
.composition-cards .metric {
  min-height: 0;
  padding: 13px 15px;
}
.composition-cards .metric-value {
  font-size: clamp(25px, 2.55vw, 42px);
}
.collection-pipeline {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px 14px;
}
.pipeline-title {
  color: #334155;
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.08em;
  margin-bottom: 9px;
  text-transform: uppercase;
}
.pipeline-steps {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}
.pipeline-step {
  position: relative;
  min-height: 94px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 11px 11px 10px 42px;
  background: #f8fafc;
}
.pipeline-num {
  position: absolute;
  left: 11px;
  top: 11px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: var(--accent);
  color: #fff;
  font-size: 12px;
  font-weight: 800;
}
.pipeline-step strong {
  display: block;
  color: #172033;
  font-size: clamp(14px, 1vw, 17px);
  margin-bottom: 4px;
}
.pipeline-step span {
  display: block;
  color: var(--muted);
  font-size: clamp(12px, 0.85vw, 14px);
  line-height: 1.27;
}
.composition-bottom {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}
.metric {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 15px 16px;
  min-height: 126px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.metric.scratch { border-top: 4px solid var(--scratch); }
.metric.frozen { border-top: 4px solid var(--frozen); }
.metric.partial { border-top: 4px solid var(--partial); }
.metric.danger { border-top: 4px solid var(--danger); }
.metric-label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.metric-value {
  color: var(--accent);
  font-size: clamp(28px, 3vw, 50px);
  line-height: 1;
  font-weight: 780;
  font-variant-numeric: tabular-nums;
}
.metric.scratch .metric-value { color: var(--scratch); }
.metric.frozen .metric-value { color: var(--frozen); }
.metric.partial .metric-value { color: var(--partial); }
.metric.danger .metric-value { color: var(--danger); }
.metric-caption {
  color: #2d3541;
  font-size: 14px;
  line-height: 1.32;
}
.callout {
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 5px solid var(--accent);
  border-radius: 8px;
  padding: 14px 17px;
  color: #25303c;
  font-size: clamp(15px, 1.18vw, 19px);
  line-height: 1.42;
}
.callout.strong {
  border-left-color: var(--danger);
  background: #fff4f2;
}
.chart-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 10px 4px;
  min-height: 0;
  height: 100%;
  display: flex;
}
.small-multiple-grid {
  flex: 1;
  min-height: 0;
  display: grid;
  gap: 13px;
}
.small-multiple-grid.four {
  grid-template-columns: repeat(2, 1fr);
}
.small-multiple-grid.three {
  grid-template-columns: repeat(3, 1fr);
}
.small-chart-card {
  min-width: 0;
  min-height: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 6px;
}
.small-chart-svg {
  display: block;
  width: 100%;
  height: 100%;
  min-height: 225px;
}
.chart-svg { width: 100%; height: 100%; min-height: 330px; }
.chart-title {
  font-size: 19px;
  font-weight: 800;
  fill: var(--ink);
}
.chart-subtitle {
  font-size: 13px;
  font-weight: 500;
  fill: var(--muted);
}
.chart-note, .axis-tick, .legend, .x-label {
  font-size: 13px;
  fill: var(--muted);
}
.x-label { font-weight: 700; fill: #384454; }
.bar-label {
  font-size: 12px;
  font-weight: 800;
  fill: #263241;
}
.grid { stroke: #e6ebf2; stroke-width: 1; }
.axis { stroke: var(--line-strong); stroke-width: 1.4; }
.media-panel {
  margin: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  min-height: 0;
  height: 100%;
  display: flex;
  flex-direction: column;
}
.media-panel img {
  width: 100%;
  height: 100%;
  min-height: 0;
  object-fit: contain;
  background: #fff;
}
.caption {
  border-top: 1px solid var(--line);
  color: var(--muted);
  padding: 8px 10px;
  font-size: 12px;
  line-height: 1.25;
}
.raw-slide {
  flex: 1;
  min-height: 0;
  display: flex;
}
.raw-dataset-grid {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-rows: repeat(4, minmax(0, 1fr));
  gap: 10px;
}
.raw-axis-row {
  min-height: 0;
  display: grid;
  grid-template-columns: 210px 1fr;
  gap: 12px;
}
.raw-axis-label {
  border: 1px solid var(--line);
  border-left: 6px solid var(--axis-color);
  background: var(--panel);
  border-radius: 8px;
  padding: 12px 13px;
  min-height: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 8px;
}
.raw-axis-label strong {
  color: var(--axis-color);
  font-size: clamp(17px, 1.35vw, 23px);
  line-height: 1.02;
}
.raw-axis-label span {
  color: var(--muted);
  font-size: clamp(12px, 0.92vw, 15px);
  line-height: 1.28;
}
.raw-strip {
  min-width: 0;
  min-height: 0;
  display: grid;
  gap: 9px;
}
.raw-cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.raw-cols-8 { grid-template-columns: repeat(8, minmax(0, 1fr)); }
.raw-card {
  margin: 0;
  min-width: 0;
  min-height: 0;
  border: 1px solid var(--line);
  border-top: 5px solid var(--axis-color);
  border-radius: 8px;
  background: var(--panel);
  padding: 6px;
  display: flex;
  flex-direction: column;
}
.raw-card img {
  width: 100%;
  min-height: 0;
  flex: 1 1 auto;
  object-fit: contain;
  image-rendering: auto;
  background: #fff;
  border: 1px solid #d7e0ec;
}
.raw-card figcaption {
  flex: 0 0 auto;
  display: flex;
  justify-content: space-between;
  gap: 6px;
  color: var(--muted);
  font-size: clamp(10px, 0.8vw, 12px);
  line-height: 1.2;
  padding-top: 5px;
  font-variant-numeric: tabular-nums;
}
.raw-card figcaption strong {
  color: #273241;
  font-weight: 800;
}
.raw-card figcaption span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.policy-layout {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}
.policy-card {
  min-height: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-top: 7px solid var(--accent);
  border-radius: 8px;
  padding: 20px 21px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.policy-card.scratch { border-top-color: var(--scratch); }
.policy-card.frozen { border-top-color: var(--frozen); }
.policy-card.partial { border-top-color: var(--partial); }
.policy-kicker {
  color: var(--accent);
  font-size: 13px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.policy-card.scratch .policy-kicker { color: var(--scratch); }
.policy-card.frozen .policy-kicker { color: var(--frozen); }
.policy-card.partial .policy-kicker { color: var(--partial); }
.policy-card h2 {
  margin: 0;
  color: var(--ink);
  font-size: clamp(21px, 1.9vw, 31px);
  line-height: 1.08;
  letter-spacing: 0;
}
.policy-card p {
  margin: 0;
  color: #334155;
  font-size: clamp(14px, 1.05vw, 17px);
  line-height: 1.38;
}
.policy-card ul {
  margin: 0;
  padding-left: 20px;
  color: var(--muted);
  font-size: clamp(13px, 0.98vw, 16px);
  line-height: 1.34;
  display: grid;
  gap: 7px;
}
.policy-common {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.policy-common div {
  background: #edf5f7;
  border: 1px solid #c7dce4;
  border-left: 5px solid var(--accent);
  border-radius: 8px;
  padding: 12px 14px;
  color: #22313c;
  font-size: clamp(13px, 1vw, 16px);
  line-height: 1.34;
}
.method-grid {
  flex: 0 1 auto;
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  align-content: start;
  align-items: start;
}
.method-card {
  min-height: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-top: 7px solid var(--accent);
  border-radius: 8px;
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.method-card.problem { border-top-color: var(--danger); }
.method-kicker {
  color: var(--accent);
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.method-card.problem .method-kicker { color: var(--danger); }
.method-card h2 {
  margin: 0;
  color: var(--ink);
  font-size: clamp(17px, 1.18vw, 22px);
  line-height: 1.08;
  letter-spacing: 0;
}
.method-card ul {
  margin: 0;
  padding-left: 20px;
  color: #334155;
  font-size: clamp(11px, 0.82vw, 14px);
  line-height: 1.22;
  display: grid;
  gap: 4px;
}
.method-card code {
  background: #eef3f8;
  border: 1px solid #d7e0ec;
  border-radius: 5px;
  padding: 1px 4px;
  color: #172033;
  font-size: 0.92em;
}
.method-footer {
  background: #edf5f7;
  border: 1px solid #c7dce4;
  border-left: 5px solid var(--accent);
  border-radius: 8px;
  padding: 13px 15px;
  display: grid;
  grid-template-columns: 0.8fr 1.35fr 1.35fr;
  gap: 12px;
  color: #22313c;
}
.method-footer div {
  min-width: 0;
  display: grid;
  gap: 6px;
}
.method-footer strong {
  color: #172033;
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.method-footer code,
.method-footer span {
  color: #25303c;
  font-size: clamp(12px, 0.9vw, 15px);
  line-height: 1.25;
}
.method-footer code {
  white-space: normal;
  overflow-wrap: anywhere;
}
.eval-layout {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 15px;
}
.eval-card {
  min-height: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-top: 7px solid var(--accent);
  border-radius: 8px;
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 11px;
}
.problem-statement .eval-card.human { border-top-color: var(--scratch); }
.problem-statement .eval-card.missing { border-top-color: var(--danger); }
.problem-statement .eval-card.question { border-top-color: var(--partial); }
.problem-statement .eval-card.human .eval-kicker { color: var(--scratch); }
.problem-statement .eval-card.missing .eval-kicker { color: var(--danger); }
.problem-statement .eval-card.question .eval-kicker { color: var(--partial); }
.eval-kicker {
  color: var(--accent);
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.eval-card h2 {
  margin: 0;
  color: var(--ink);
  font-size: clamp(20px, 1.55vw, 29px);
  line-height: 1.08;
}
.eval-card ul {
  margin: 0;
  padding-left: 20px;
  color: #334155;
  font-size: clamp(13px, 1vw, 16px);
  line-height: 1.32;
  display: grid;
  gap: 7px;
}
.eval-formula {
  background: #edf5f7;
  border: 1px solid #c7dce4;
  border-left: 5px solid var(--accent);
  border-radius: 8px;
  padding: 14px 16px;
  display: grid;
  grid-template-columns: 1fr 1.45fr;
  gap: 14px;
}
.eval-formula code {
  color: #172033;
  font-size: clamp(15px, 1.2vw, 20px);
  font-weight: 800;
  overflow-wrap: anywhere;
}
.problem-formula {
  grid-template-columns: 1.2fr 0.95fr;
  align-items: center;
}
.problem-formula .math-block {
  font-size: clamp(12px, 0.98vw, 18px);
}
.axis-video-grid {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}
.axis-video-grid .video-panel {
  min-height: 0;
}
.axis-video-grid .video-label {
  font-size: 11px;
}
.equations-slide h1 {
  max-width: 1320px;
  font-size: clamp(29px, 2.9vw, 50px);
}
.equation-layout {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1.18fr 0.82fr;
  gap: 16px;
  align-items: stretch;
}
.equation-panel,
.rubric-panel {
  min-height: 0;
  display: grid;
  gap: 10px;
}
.equation-panel {
  grid-template-rows: auto repeat(4, minmax(0, 1fr));
}
.rubric-panel {
  grid-template-rows: repeat(3, minmax(0, 1fr));
}
.eq-group,
.rubric-card {
  min-height: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 11px 13px;
}
.eq-group {
  border-left: 5px solid var(--accent);
  display: grid;
  gap: 4px;
}
.rubric-card {
  border-top: 6px solid var(--accent);
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.rubric-card.taught { border-top-color: var(--scratch); }
.rubric-card.applied { border-top-color: var(--frozen); }
.rubric-card.beyond { border-top-color: var(--partial); }
.equation-kicker {
  color: var(--accent);
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.rubric-card.taught .equation-kicker { color: var(--scratch); }
.rubric-card.applied .equation-kicker { color: var(--frozen); }
.rubric-card.beyond .equation-kicker { color: var(--partial); }
.eq-group h2 {
  margin: 0;
  color: #172033;
  font-size: clamp(14px, 1.02vw, 19px);
  line-height: 1.08;
}
.eq-group p {
  margin: 0;
  color: var(--muted);
  font-size: clamp(10px, 0.8vw, 13px);
  line-height: 1.22;
}
.math-block {
  color: #172033;
  font-size: clamp(12px, 0.95vw, 17px);
  line-height: 1.1;
  overflow-x: auto;
  overflow-y: hidden;
}
.math-block mjx-container {
  margin: 0 !important;
}
.rubric-card ul {
  margin: 0;
  padding-left: 18px;
  color: #334155;
  font-size: clamp(10.5px, 0.84vw, 14px);
  line-height: 1.22;
  display: grid;
  gap: 4px;
}
.rubric-card li::marker {
  color: var(--accent);
}
.video-panel {
  position: relative;
  margin: 0;
  height: 100%;
  min-height: 0;
  background: #0e1218;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}
.video-panel video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #111;
}
.video-label {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 2;
  border-radius: 6px;
  padding: 5px 9px;
  background: rgba(20, 25, 32, 0.78);
  color: #fff;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.video-panel.danger .video-label { background: rgba(199, 67, 67, 0.9); }
.video-panel.success .video-label { background: rgba(38, 132, 90, 0.9); }
.table-wrap {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}
.result-table {
  width: 100%;
  border-collapse: collapse;
  font-size: clamp(15px, 1.25vw, 21px);
}
.result-table th, .result-table td {
  border-bottom: 1px solid var(--line);
  padding: 14px 16px;
  text-align: left;
  font-variant-numeric: tabular-nums;
}
.result-table th {
  background: #edf2f7;
  color: #334155;
  font-size: 13px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.compact-table .result-table { font-size: clamp(11px, 0.9vw, 14px); }
.compact-table .result-table th, .compact-table .result-table td { padding: 10px 8px; }
.result-table tr:last-child td { border-bottom: 0; }
.claim-list {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.claim {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 20px;
  display: grid;
  grid-template-columns: 52px 1fr;
  gap: 14px;
  align-items: start;
}
.claim span {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  background: var(--accent);
  color: #fff;
  font-weight: 800;
}
.claim p {
  margin: 0;
  font-size: clamp(17px, 1.45vw, 24px);
  line-height: 1.38;
}
.progress {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  z-index: 20;
  background: transparent;
}
.progress div {
  height: 100%;
  width: 0%;
  background: var(--accent);
  transition: width 160ms ease;
}
.controls {
  position: fixed;
  right: 18px;
  bottom: 14px;
  z-index: 25;
  display: flex;
  align-items: center;
  gap: 9px;
  color: var(--muted);
  font-size: 13px;
}
.controls button {
  border: 1px solid var(--line-strong);
  background: rgba(255, 255, 255, 0.9);
  color: var(--ink);
  border-radius: 6px;
  padding: 5px 9px;
  font: inherit;
  cursor: pointer;
}
.controls button:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
.counter { min-width: 62px; text-align: center; font-variant-numeric: tabular-nums; }
.hint {
  position: fixed;
  left: 18px;
  bottom: 14px;
  z-index: 25;
  color: var(--muted);
  font-size: 12px;
}
.hint kbd {
  border: 1px solid var(--line-strong);
  background: #fff;
  border-radius: 4px;
  padding: 1px 5px;
  margin: 0 2px;
}
.transcript {
  display: none;
  position: absolute;
  right: 54px;
  bottom: 58px;
  max-width: 520px;
  max-height: 180px;
  overflow: auto;
  background: rgba(255,255,255,0.96);
  border: 1px solid var(--line-strong);
  border-radius: 8px;
  padding: 12px 14px;
  color: #263241;
  font-size: 13px;
  line-height: 1.38;
  box-shadow: 0 12px 34px rgba(25, 35, 50, 0.16);
}
body.show-notes .slide.active .transcript { display: block; }
@media (max-width: 900px) {
  .slide { padding: 28px 22px 58px; overflow: auto; }
  .hero-layout, .two-col, .two-col.wide, .three-grid, .claim-list, .composition-bottom, .pipeline-steps, .small-multiple-grid.four, .small-multiple-grid.three, .policy-layout, .policy-common, .method-grid, .method-footer, .eval-layout, .eval-formula, .axis-video-grid, .equation-layout { grid-template-columns: 1fr; }
  .metric-grid.four { grid-template-columns: repeat(2, 1fr); }
  .chart-svg { min-height: 300px; }
  html, body { overflow: auto; }
}
@media print {
  html, body { overflow: visible; height: auto; }
  .slide { position: relative; display: flex !important; page-break-after: always; height: 100vh; }
  .controls, .hint, .progress { display: none; }
  .transcript { display: none !important; }
}
"""

    script = r"""
(function() {
  const slides = Array.from(document.querySelectorAll('.slide'));
  const total = slides.length;
  const cur = document.getElementById('cur');
  const totalEl = document.getElementById('total');
  const bar = document.getElementById('bar');
  totalEl.textContent = total;
  let index = 0;

  function show(next) {
    index = Math.max(0, Math.min(total - 1, next));
    slides.forEach((slide, i) => slide.classList.toggle('active', i === index));
    cur.textContent = index + 1;
    bar.style.width = (((index + 1) / total) * 100).toFixed(2) + '%';
    history.replaceState(null, '', '#' + (index + 1));
    document.querySelectorAll('video').forEach(video => {
      const active = video.closest('.slide') === slides[index];
      if (!active) {
        try { video.pause(); } catch (e) {}
      }
    });
  }

  function fromHash() {
    const n = parseInt(location.hash.replace('#', ''), 10);
    if (Number.isFinite(n)) show(n - 1);
  }

  document.getElementById('prev').addEventListener('click', () => show(index - 1));
  document.getElementById('next').addEventListener('click', () => show(index + 1));
  document.getElementById('fs').addEventListener('click', () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
    else document.exitFullscreen().catch(() => {});
  });
  document.addEventListener('keydown', event => {
    const tag = event.target && event.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (event.key === 'ArrowRight' || event.key === ' ' || event.key === 'PageDown') show(index + 1);
    else if (event.key === 'ArrowLeft' || event.key === 'PageUp') show(index - 1);
    else if (event.key === 'Home') show(0);
    else if (event.key === 'End') show(total - 1);
    else if (event.key.toLowerCase() === 'f') document.getElementById('fs').click();
    else if (event.key.toLowerCase() === 'n') document.body.classList.toggle('show-notes');
    else if (event.key.toLowerCase() === 'p') window.print();
  });
  window.addEventListener('hashchange', fromHash);
  fromHash();
  if (!document.querySelector('.slide.active')) show(0);
})();
"""

    slide_html = "\n\n".join(slides)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>V-BCOOD Results Presentation</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']]
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
{style}
</style>
</head>
<body>
<div class="progress"><div id="bar"></div></div>
<main class="deck">
{slide_html}
</main>
<div class="hint"><kbd>Left</kbd><kbd>Right</kbd> navigate &middot; <kbd>N</kbd> notes &middot; <kbd>F</kbd> fullscreen</div>
<div class="controls">
  <button id="prev" title="Previous">Prev</button>
  <span class="counter"><span id="cur">1</span> / <span id="total">1</span></span>
  <button id="next" title="Next">Next</button>
  <button id="fs" title="Fullscreen">Full</button>
</div>
<script>
{script}
</script>
</body>
</html>
"""


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
