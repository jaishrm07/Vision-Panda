#!/usr/bin/env python3
"""Build analysis tables from completed research2 precision eval metrics."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_DIR = RESULTS / "analysis_id_ood_all_budgets"

FAMILIES = {
    "scratch_cnn": {
        "label": "Scratch CNN",
        "metrics_dir": RESULTS / "bc_128px_v1" / "metrics",
    },
    "frozen_resnet18": {
        "label": "Frozen ResNet-18",
        "metrics_dir": RESULTS / "frozen_resnet18_bc_128px_v1" / "metrics",
    },
    "partial_resnet18": {
        "label": "Partial ResNet-18",
        "metrics_dir": RESULTS / "partial_resnet18_bc_128px_v1" / "metrics",
    },
}

AXIS_LABELS = {
    "color": "Color",
    "spatial_distribution": "Spatial distribution",
    "camera_location_viewpoint": "Camera viewpoint",
    "lighting_direction_intensity": "Lighting",
}

TASK_LABELS = {
    "reach": "Reach",
    "avoid_reach": "Obstacle-aware reach",
}

DIVERSITY_PAIRS = {
    ("reach", "color"): ("color_red_only", "color_multi"),
    ("avoid_reach", "color"): ("avoid_color_red_only", "avoid_color_multi"),
    ("reach", "spatial_distribution"): ("spatial_narrow", "spatial_wide"),
    ("avoid_reach", "spatial_distribution"): ("avoid_spatial_narrow", "avoid_spatial_wide"),
    ("reach", "camera_location_viewpoint"): ("camera_fixed", "camera_multi_pose"),
    ("avoid_reach", "camera_location_viewpoint"): ("avoid_camera_fixed", "avoid_camera_multi_pose"),
    ("reach", "lighting_direction_intensity"): ("lighting_fixed", "lighting_diverse"),
    ("avoid_reach", "lighting_direction_intensity"): ("avoid_lighting_fixed", "avoid_lighting_diverse"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return 100.0 * float(value)


def cm(value: float | None) -> float | None:
    if value is None:
        return None
    return 100.0 * float(value)


def get_metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    return float(value)


def per_file_row(family_key: str, family_label: str, split: str, path: Path) -> dict[str, Any]:
    payload = read_json(path)
    metadata = payload["checkpoint_metadata"]["dataset_metadata"]
    metrics = payload["metrics"]
    eval_datasets = payload.get("eval_datasets", [])
    first_eval = eval_datasets[0] if eval_datasets else {}
    rollout_count = int(metrics["rollout_count"])
    row = {
        "family": family_key,
        "family_label": family_label,
        "split": split,
        "model_family": payload.get("model_family"),
        "train_config": metadata["train_config"],
        "task": metadata["task"],
        "task_label": TASK_LABELS.get(metadata["task"], metadata["task"]),
        "axis": metadata["axis"],
        "axis_label": AXIS_LABELS.get(metadata["axis"], metadata["axis"]),
        "variant": metadata["variant"],
        "budget": int(metadata["budget"]),
        "seed": int(metadata["seed"]),
        "eval_config": first_eval.get("eval_config"),
        "eval_variant": first_eval.get("variant"),
        "rollout_count": rollout_count,
        "success_at_0p5cm_count": int(metrics.get("success_count_at_0p5cm", 0)),
        "success_at_1cm_count": int(metrics.get("success_count_at_1cm", 0)),
        "success_at_2cm_count": int(metrics.get("success_count_at_2cm", 0)),
        "success_at_5cm_count": int(metrics.get("success_count_at_5cm", 0)),
        "success_at_0p5cm": get_metric(metrics, "success_rate_at_0p5cm"),
        "success_at_1cm": get_metric(metrics, "success_rate_at_1cm"),
        "success_at_2cm": get_metric(metrics, "success_rate_at_2cm"),
        "success_at_5cm": get_metric(metrics, "success_rate_at_5cm"),
        "nearest_distance_m": get_metric(metrics, "mean_best_distance"),
        "end_distance_m": get_metric(metrics, "mean_final_distance"),
        "mean_steps": get_metric(metrics, "mean_steps"),
        "metrics_path": str(path.relative_to(ROOT)),
    }
    row["nearest_distance_cm"] = cm(row["nearest_distance_m"])
    row["end_distance_cm"] = cm(row["end_distance_m"])
    return row


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_key, family_info in FAMILIES.items():
        metrics_dir = family_info["metrics_dir"]
        for split in ("id", "ood"):
            for path in sorted((metrics_dir / split).glob("*.json")):
                rows.append(per_file_row(family_key, family_info["label"], split, path))
    return rows


def mean_ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None, None
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0, 0.0
    variance = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
    std = math.sqrt(variance)
    ci95 = 1.96 * std / math.sqrt(len(clean))
    return mean, std, ci95


def aggregate(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        record = dict(zip(group_fields, key))
        rollout_count = sum(int(row["rollout_count"]) for row in group)
        record["metric_file_count"] = len(group)
        record["rollout_count"] = rollout_count
        record["seed_count"] = len({row["seed"] for row in group})
        record["train_config_count"] = len({row["train_config"] for row in group})
        for label in ("0p5cm", "1cm", "2cm", "5cm"):
            count_key = f"success_at_{label}_count"
            count = sum(int(row[count_key]) for row in group)
            record[count_key] = count
            record[f"success_at_{label}"] = count / rollout_count if rollout_count else None
            record[f"success_at_{label}_pct"] = pct(record[f"success_at_{label}"])
        for metric in ("nearest_distance_m", "end_distance_m", "mean_steps"):
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            weighted = None
            if rollout_count and values:
                weighted = sum(float(row[metric]) * int(row["rollout_count"]) for row in group if row.get(metric) is not None) / rollout_count
            mean, std, ci95 = mean_ci(values)
            record[metric] = weighted
            record[f"{metric}_file_mean"] = mean
            record[f"{metric}_file_std"] = std
            record[f"{metric}_file_ci95"] = ci95
        record["nearest_distance_cm"] = cm(record["nearest_distance_m"])
        record["end_distance_cm"] = cm(record["end_distance_m"])
        out.append(record)
    return out


def diversity_gain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["family"], row["split"], row["task"], row["axis"], row["train_config"], row["budget"]): row
        for row in aggregate(rows, ["family", "family_label", "split", "task", "task_label", "axis", "axis_label", "train_config", "variant", "budget"])
    }
    gains: list[dict[str, Any]] = []
    for family_key, family_info in FAMILIES.items():
        for (task, axis), (baseline_config, diverse_config) in DIVERSITY_PAIRS.items():
            budgets = sorted({row["budget"] for row in rows if row["family"] == family_key and row["split"] == "ood" and row["task"] == task and row["axis"] == axis})
            for budget in budgets:
                baseline = by_key.get((family_key, "ood", task, axis, baseline_config, budget))
                diverse = by_key.get((family_key, "ood", task, axis, diverse_config, budget))
                if baseline is None or diverse is None:
                    continue
                gains.append(
                    {
                        "family": family_key,
                        "family_label": family_info["label"],
                        "task": task,
                        "task_label": TASK_LABELS.get(task, task),
                        "axis": axis,
                        "axis_label": AXIS_LABELS.get(axis, axis),
                        "budget": budget,
                        "baseline_config": baseline_config,
                        "diverse_config": diverse_config,
                        "baseline_success_at_1cm_pct": baseline["success_at_1cm_pct"],
                        "diverse_success_at_1cm_pct": diverse["success_at_1cm_pct"],
                        "gain_success_at_1cm_pct": diverse["success_at_1cm_pct"] - baseline["success_at_1cm_pct"],
                        "baseline_success_at_5cm_pct": baseline["success_at_5cm_pct"],
                        "diverse_success_at_5cm_pct": diverse["success_at_5cm_pct"],
                        "gain_success_at_5cm_pct": diverse["success_at_5cm_pct"] - baseline["success_at_5cm_pct"],
                        "baseline_nearest_distance_cm": baseline["nearest_distance_cm"],
                        "diverse_nearest_distance_cm": diverse["nearest_distance_cm"],
                        "gain_nearest_distance_cm": diverse["nearest_distance_cm"] - baseline["nearest_distance_cm"],
                    }
                )
    return gains


def write_markdown_summary(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    overall = tables["overall_by_family_split_budget"]
    scratch_overall = [row for row in overall if row["family"] == "scratch_cnn"]
    ood_axis_50 = [
        row
        for row in tables["ood_axis_by_family_budget"]
        if int(row["budget"]) == 50
    ]
    ood_axis_200 = [
        row
        for row in tables["ood_axis_by_family_budget"]
        if int(row["budget"]) == 200
    ]

    def fmt_pct(value: float | None) -> str:
        return "" if value is None else f"{value:.1f}%"

    def fmt_cm(value: float | None) -> str:
        return "" if value is None else f"{value:.1f} cm"

    lines = [
        "# Completed ID/OOD Metrics Summary",
        "",
        f"Generated: {utc_now()}",
        "",
        "Scope: precision evaluation metrics for ID and OOD splits only. The old Scratch CNN coarse 5cm early-stop split is excluded.",
        "",
        "Budgets 5/20/50 use seeds 0/1/2. Budgets 100/200 use seed 0 only, so treat high-budget trends as directional until repeated with more seeds.",
        "",
        "## Scratch CNN Budget Scaling",
        "",
        "| Split | Budget | success@1cm | success@2cm | success@5cm | nearest | end | metric files | rollouts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(scratch_overall, key=lambda r: (r["split"], int(r["budget"]))):
        lines.append(
            f"| {row['split'].upper()} | {row['budget']} | {fmt_pct(row['success_at_1cm_pct'])} | {fmt_pct(row['success_at_2cm_pct'])} | {fmt_pct(row['success_at_5cm_pct'])} | {fmt_cm(row['nearest_distance_cm'])} | {fmt_cm(row['end_distance_cm'])} | {row['metric_file_count']} | {row['rollout_count']} |"
        )

    lines += [
        "",
        "## OOD Axis, Budget 50",
        "",
        "| Model | Axis | success@1cm | success@2cm | success@5cm | nearest | end |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(ood_axis_50, key=lambda r: (r["family_label"], r["axis_label"])):
        lines.append(
            f"| {row['family_label']} | {row['axis_label']} | {fmt_pct(row['success_at_1cm_pct'])} | {fmt_pct(row['success_at_2cm_pct'])} | {fmt_pct(row['success_at_5cm_pct'])} | {fmt_cm(row['nearest_distance_cm'])} | {fmt_cm(row['end_distance_cm'])} |"
        )

    lines += [
        "",
        "## OOD Axis, Budget 200",
        "",
        "| Model | Axis | success@1cm | success@2cm | success@5cm | nearest | end |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(ood_axis_200, key=lambda r: (r["family_label"], r["axis_label"])):
        lines.append(
            f"| {row['family_label']} | {row['axis_label']} | {fmt_pct(row['success_at_1cm_pct'])} | {fmt_pct(row['success_at_2cm_pct'])} | {fmt_pct(row['success_at_5cm_pct'])} | {fmt_cm(row['nearest_distance_cm'])} | {fmt_cm(row['end_distance_cm'])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    tables = {
        "per_model_metrics": rows,
        "overall_by_family_split_budget": aggregate(rows, ["family", "family_label", "split", "budget"]),
        "ood_axis_by_family_budget": aggregate([row for row in rows if row["split"] == "ood"], ["family", "family_label", "axis", "axis_label", "budget"]),
        "axis_by_family_split_budget": aggregate(rows, ["family", "family_label", "split", "axis", "axis_label", "budget"]),
        "ood_task_by_family_budget": aggregate([row for row in rows if row["split"] == "ood"], ["family", "family_label", "task", "task_label", "budget"]),
        "task_by_family_split_budget": aggregate(rows, ["family", "family_label", "split", "task", "task_label", "budget"]),
        "train_config_by_family_split_budget": aggregate(rows, ["family", "family_label", "split", "task", "task_label", "axis", "axis_label", "train_config", "variant", "budget"]),
    }
    tables["diversity_pair_gains_ood_by_family_budget"] = diversity_gain_rows(rows)

    for name, table_rows in tables.items():
        write_csv(OUTPUT_DIR / f"{name}.csv", table_rows)

    split_counts: dict[str, int] = {}
    budget_counts: dict[str, int] = {}
    for row in rows:
        split_key = f"{row['family']}:{row['split']}"
        budget_key = f"{row['family']}:{row['split']}:budget{row['budget']}"
        split_counts[split_key] = split_counts.get(split_key, 0) + 1
        budget_counts[budget_key] = budget_counts.get(budget_key, 0) + 1

    write_json(
        OUTPUT_DIR / "summary.json",
        {
            "created_at": utc_now(),
            "scope": "precision ID/OOD metrics only; excludes stale coarse early-stop Scratch CNN split",
            "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
            "metric_file_count": len(rows),
            "rollout_count": sum(int(row["rollout_count"]) for row in rows),
            "families": {key: info["label"] for key, info in FAMILIES.items()},
            "split_metric_file_counts": split_counts,
            "budget_metric_file_counts": budget_counts,
            "tables": {name: len(table_rows) for name, table_rows in tables.items()},
        },
    )
    write_markdown_summary(OUTPUT_DIR / "completed_metrics_summary.md", tables)
    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
                "metric_file_count": len(rows),
                "rollout_count": sum(int(row["rollout_count"]) for row in rows),
                "tables": {name: len(table_rows) for name, table_rows in tables.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
