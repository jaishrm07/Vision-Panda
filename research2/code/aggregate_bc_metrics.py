#!/usr/bin/env python3
"""Aggregate research2 BC ID/OOD closed-loop metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_split(path: Path, metrics_dir: Path) -> str:
    try:
        rel = path.relative_to(metrics_dir)
    except ValueError:
        return path.parent.name
    if len(rel.parts) > 1 and rel.parts[0] in {"id", "ood"}:
        return rel.parts[0]
    if "ood" in path.name:
        return "ood"
    if "id" in path.name:
        return "id"
    return path.parent.name


def row_from_metrics(path: Path, metrics_dir: Path) -> dict[str, Any]:
    payload = read_json(path)
    checkpoint_metadata = payload.get("checkpoint_metadata", {})
    dataset_metadata = checkpoint_metadata.get("dataset_metadata", {})
    eval_datasets = payload.get("eval_datasets", [])
    first_eval = eval_datasets[0] if eval_datasets else {}
    metrics = payload.get("metrics", {})
    row = {
        "metrics_path": str(path),
        "eval_split": parse_split(path, metrics_dir),
        "model_family": payload.get("model_family"),
        "train_config": dataset_metadata.get("train_config"),
        "task": dataset_metadata.get("task"),
        "axis": dataset_metadata.get("axis"),
        "variant": dataset_metadata.get("variant"),
        "budget": int(dataset_metadata.get("budget", -1)),
        "seed": int(dataset_metadata.get("seed", -1)),
        "eval_config": first_eval.get("eval_config"),
        "eval_variant": first_eval.get("variant"),
        "rollout_count": int(metrics.get("rollout_count", 0)),
        "success_count": int(metrics.get("success_count", 0)),
        "success_rate": float(metrics.get("success_rate", 0.0)),
        "mean_best_distance": metrics.get("mean_best_distance"),
        "mean_final_distance": metrics.get("mean_final_distance"),
        "mean_steps": metrics.get("mean_steps"),
        "mean_success_steps": metrics.get("mean_success_steps"),
    }
    for key, value in metrics.items():
        if key.startswith(("success_count_at_", "success_rate_at_", "mean_first_step_at_")):
            row[key] = value
    return row


def scalar_summary(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    if len(clean) == 1:
        return {"n": 1, "mean": clean[0], "std": 0.0, "ci95": 0.0}
    std = statistics.stdev(clean)
    return {
        "n": len(clean),
        "mean": float(statistics.mean(clean)),
        "std": float(std),
        "ci95": float(1.96 * std / math.sqrt(len(clean))),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    key_fields = ["eval_split", "task", "axis", "train_config", "variant", "budget", "eval_config", "eval_variant"]
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)

    aggregate_records = []
    for key, group_rows in sorted(groups.items()):
        record = dict(zip(key_fields, key))
        metric_names = ["success_rate", "mean_best_distance", "mean_final_distance", "mean_steps", "mean_success_steps"]
        threshold_metric_names = sorted(
            {
                key
                for row in group_rows
                for key in row
                if key.startswith(("success_rate_at_", "mean_first_step_at_"))
            }
        )
        for metric_name in metric_names + threshold_metric_names:
            summary = scalar_summary([row.get(metric_name) for row in group_rows])
            record[f"{metric_name}_n"] = summary["n"]
            record[f"{metric_name}_mean"] = summary["mean"]
            record[f"{metric_name}_std"] = summary["std"]
            record[f"{metric_name}_ci95"] = summary["ci95"]
        aggregate_records.append(record)
    return aggregate_records


def diversity_gain_rows(aggregate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["eval_split"], row["task"], row["axis"], row["train_config"], row["budget"]): row
        for row in aggregate_records
    }
    gains = []
    for (task, axis), (baseline_config, diversity_config) in DIVERSITY_PAIRS.items():
        budgets = sorted(
            {
                int(row["budget"])
                for row in aggregate_records
                if row["task"] == task and row["axis"] == axis and row["eval_split"] == "ood"
            }
        )
        for budget in budgets:
            baseline = by_key.get(("ood", task, axis, baseline_config, budget))
            diverse = by_key.get(("ood", task, axis, diversity_config, budget))
            if baseline is None or diverse is None:
                continue
            baseline_success = baseline.get("success_rate_mean")
            diverse_success = diverse.get("success_rate_mean")
            if baseline_success is None or diverse_success is None:
                continue
            gains.append(
                {
                    "task": task,
                    "axis": axis,
                    "budget": budget,
                    "baseline_config": baseline_config,
                    "diversity_config": diversity_config,
                    "baseline_ood_success_rate_mean": baseline_success,
                    "diverse_ood_success_rate_mean": diverse_success,
                    "diversity_gain": float(diverse_success - baseline_success),
                }
            )
    return gains


def generalization_gap_rows(aggregate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["eval_split"], row["task"], row["axis"], row["train_config"], row["budget"]): row
        for row in aggregate_records
    }
    gaps = []
    for row in aggregate_records:
        if row["eval_split"] != "id":
            continue
        ood_row = by_key.get(("ood", row["task"], row["axis"], row["train_config"], row["budget"]))
        if ood_row is None:
            continue
        id_success = row.get("success_rate_mean")
        ood_success = ood_row.get("success_rate_mean")
        if id_success is None or ood_success is None:
            continue
        gaps.append(
            {
                "task": row["task"],
                "axis": row["axis"],
                "train_config": row["train_config"],
                "budget": row["budget"],
                "id_success_rate_mean": id_success,
                "ood_success_rate_mean": ood_success,
                "generalization_gap": float(id_success - ood_success),
            }
        )
    return gaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/bc_128px_v1/metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/bc_128px_v1/aggregates"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_paths = [path for path in sorted(args.metrics_dir.rglob("*.json")) if "metrics" in read_json(path)]
    rows = [row_from_metrics(path, args.metrics_dir) for path in metric_paths]
    aggregate_records = aggregate_rows(rows)
    diversity_records = diversity_gain_rows(aggregate_records)
    gap_records = generalization_gap_rows(aggregate_records)

    write_csv(args.output_dir / "per_model_metrics.csv", rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_records)
    write_csv(args.output_dir / "diversity_gains.csv", diversity_records)
    write_csv(args.output_dir / "generalization_gaps.csv", gap_records)
    write_json(
        args.output_dir / "aggregate_summary.json",
        {
            "created_at": utc_now(),
            "metrics_dir": str(args.metrics_dir),
            "output_dir": str(args.output_dir),
            "metric_file_count": len(metric_paths),
            "per_model_row_count": len(rows),
            "aggregate_row_count": len(aggregate_records),
            "diversity_gain_row_count": len(diversity_records),
            "generalization_gap_row_count": len(gap_records),
        },
    )
    print(json.dumps({"metric_file_count": len(metric_paths), "aggregate_row_count": len(aggregate_records)}, indent=2))


if __name__ == "__main__":
    main()
