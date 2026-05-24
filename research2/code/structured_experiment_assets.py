#!/usr/bin/env python3
"""Build tables for the structured phase+geometry BC experiment."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


MetricSpec = tuple[str, str, str, str, str]


MAIN_SPECS: list[MetricSpec] = [
    (
        "Prior spatial-wide visual-only",
        "Scratch CNN",
        "ID",
        "scratch",
        "results/bc_128px_phase_precise_avoid/metrics/id/metrics__scratch_bc_128__avoid_spatial_wide__budget200__seed000__eval_avoid_spatial_wide.json",
    ),
    (
        "Prior spatial-wide visual-only",
        "Scratch CNN",
        "OOD",
        "scratch",
        "results/bc_128px_phase_precise_avoid/metrics/ood/metrics__scratch_bc_128__avoid_spatial_wide__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Prior spatial-wide visual-only",
        "Frozen ResNet18",
        "ID",
        "frozen",
        "results/frozen_resnet18_bc_128px_phase_precise_avoid/metrics/id/metrics__frozen_resnet18_bc_128__avoid_spatial_wide__budget200__seed000__eval_avoid_spatial_wide.json",
    ),
    (
        "Prior spatial-wide visual-only",
        "Frozen ResNet18",
        "OOD",
        "frozen",
        "results/frozen_resnet18_bc_128px_phase_precise_avoid/metrics/ood/metrics__frozen_resnet18_bc_128__avoid_spatial_wide__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Prior spatial-wide visual-only",
        "Partial ResNet18",
        "ID",
        "partial",
        "results/partial_resnet18_bc_128px_phase_precise_avoid/metrics/id/metrics__partial_resnet18_bc_128__avoid_spatial_wide__budget200__seed000__eval_avoid_spatial_wide.json",
    ),
    (
        "Prior spatial-wide visual-only",
        "Partial ResNet18",
        "OOD",
        "partial",
        "results/partial_resnet18_bc_128px_phase_precise_avoid/metrics/ood/metrics__partial_resnet18_bc_128__avoid_spatial_wide__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Edge-balanced visual-only",
        "Scratch CNN",
        "ID",
        "scratch",
        "results/edge_balanced_scratch_bc_128px_phase_precise_avoid/metrics/id/metrics__scratch_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_balanced.json",
    ),
    (
        "Edge-balanced visual-only",
        "Scratch CNN",
        "OOD",
        "scratch",
        "results/edge_balanced_scratch_bc_128px_phase_precise_avoid/metrics/ood/metrics__scratch_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Edge-balanced visual-only",
        "Frozen ResNet18",
        "ID",
        "frozen",
        "results/edge_balanced_frozen_resnet18_bc_128px_phase_precise_avoid/metrics/id/metrics__frozen_resnet18_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_balanced.json",
    ),
    (
        "Edge-balanced visual-only",
        "Frozen ResNet18",
        "OOD",
        "frozen",
        "results/edge_balanced_frozen_resnet18_bc_128px_phase_precise_avoid/metrics/ood/metrics__frozen_resnet18_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Edge-balanced visual-only",
        "Partial ResNet18",
        "ID",
        "partial",
        "results/edge_balanced_partial_resnet18_bc_128px_phase_precise_avoid/metrics/id/metrics__partial_resnet18_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_balanced.json",
    ),
    (
        "Edge-balanced visual-only",
        "Partial ResNet18",
        "OOD",
        "partial",
        "results/edge_balanced_partial_resnet18_bc_128px_phase_precise_avoid/metrics/ood/metrics__partial_resnet18_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Edge-balanced phase+geometry",
        "Scratch CNN",
        "ID",
        "scratch",
        "results/structured_scratch_bc_128px_phase_geo_avoid/metrics/id/metrics__scratch_structured_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_balanced.json",
    ),
    (
        "Edge-balanced phase+geometry",
        "Scratch CNN",
        "OOD",
        "scratch",
        "results/structured_scratch_bc_128px_phase_geo_avoid/metrics/ood/metrics__scratch_structured_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Edge-balanced phase+geometry",
        "Frozen ResNet18",
        "ID",
        "frozen",
        "results/structured_frozen_resnet18_bc_128px_phase_geo_avoid/metrics/id/metrics__frozen_resnet18_structured_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_balanced.json",
    ),
    (
        "Edge-balanced phase+geometry",
        "Frozen ResNet18",
        "OOD",
        "frozen",
        "results/structured_frozen_resnet18_bc_128px_phase_geo_avoid/metrics/ood/metrics__frozen_resnet18_structured_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Edge-balanced phase+geometry",
        "Partial ResNet18",
        "ID",
        "partial",
        "results/structured_partial_resnet18_bc_128px_phase_geo_avoid/metrics/id/metrics__partial_resnet18_structured_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_balanced.json",
    ),
    (
        "Edge-balanced phase+geometry",
        "Partial ResNet18",
        "OOD",
        "partial",
        "results/structured_partial_resnet18_bc_128px_phase_geo_avoid/metrics/ood/metrics__partial_resnet18_structured_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
]


ABLATION_SPECS: list[MetricSpec] = [
    (
        "Scratch ablation",
        "Phase only",
        "ID",
        "phase",
        "results/ablation_scratch_phase_only_bc_128px_phase_geo_avoid/metrics/id/metrics__scratch_phase_only_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_balanced.json",
    ),
    (
        "Scratch ablation",
        "Phase only",
        "OOD",
        "phase",
        "results/ablation_scratch_phase_only_bc_128px_phase_geo_avoid/metrics/ood/metrics__scratch_phase_only_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Scratch ablation",
        "Target geometry only",
        "ID",
        "target",
        "results/ablation_scratch_target_only_bc_128px_phase_geo_avoid/metrics/id/metrics__scratch_target_only_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_balanced.json",
    ),
    (
        "Scratch ablation",
        "Target geometry only",
        "OOD",
        "target",
        "results/ablation_scratch_target_only_bc_128px_phase_geo_avoid/metrics/ood/metrics__scratch_target_only_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Scratch ablation",
        "Full geometry only",
        "ID",
        "geometry",
        "results/ablation_scratch_geometry_only_bc_128px_phase_geo_avoid/metrics/id/metrics__scratch_geometry_only_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_balanced.json",
    ),
    (
        "Scratch ablation",
        "Full geometry only",
        "OOD",
        "geometry",
        "results/ablation_scratch_geometry_only_bc_128px_phase_geo_avoid/metrics/ood/metrics__scratch_geometry_only_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
    (
        "Scratch ablation",
        "Phase + geometry",
        "ID",
        "full",
        "results/structured_scratch_bc_128px_phase_geo_avoid/metrics/id/metrics__scratch_structured_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_balanced.json",
    ),
    (
        "Scratch ablation",
        "Phase + geometry",
        "OOD",
        "full",
        "results/structured_scratch_bc_128px_phase_geo_avoid/metrics/ood/metrics__scratch_structured_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
    ),
]


BUCKET_SPECS: list[MetricSpec] = [
    spec
    for spec in MAIN_SPECS
    if spec[0] in {"Edge-balanced visual-only", "Edge-balanced phase+geometry"}
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rate(metrics: dict[str, Any], label: str) -> float:
    return float(metrics.get(f"success_rate_at_{label}", 0.0))


def cm(value_m: float | None) -> float | None:
    if value_m is None:
        return None
    return float(value_m) * 100.0


def metric_row(root: Path, spec: MetricSpec) -> dict[str, Any] | None:
    group, model, split, tag, rel_path = spec
    path = root / rel_path
    if not path.exists():
        return None
    payload = read_json(path)
    metrics = payload["metrics"]
    return {
        "group": group,
        "model": model,
        "split": split,
        "tag": tag,
        "rollouts": int(metrics.get("rollout_count", 0)),
        "success_0p5cm": rate(metrics, "0p5cm"),
        "success_1cm": rate(metrics, "1cm"),
        "success_2cm": rate(metrics, "2cm"),
        "success_5cm": rate(metrics, "5cm"),
        "mean_best_cm": cm(metrics.get("mean_best_distance")),
        "mean_final_cm": cm(metrics.get("mean_final_distance")),
        "metrics_path": str(path.relative_to(root)),
    }


def collect_rows(root: Path, specs: list[MetricSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        row = metric_row(root, spec)
        if row is not None:
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_percent(value: Any) -> str:
    if value is None:
        return ""
    return f"{100.0 * float(value):.1f}%"


def format_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if key.startswith("success_"):
                cells.append(format_percent(value))
            elif key.endswith("_cm"):
                cells.append(format_float(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, title: str, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"# {title}\n\n"
    if rows:
        body += markdown_table(rows, columns)
    else:
        body += "No rows available yet.\n"
    path.write_text(body, encoding="utf-8")


def table_cell(value: Any, key: str) -> str:
    if key.startswith("success_"):
        return format_percent(value)
    if key.endswith("_cm"):
        return format_float(value)
    text = str(value)
    if key in {"group", "model"}:
        return "\n".join(textwrap.wrap(text, width=24))
    return text


def write_table_png(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str]],
    max_rows: int | None = None,
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    shown = rows if max_rows is None else rows[:max_rows]
    cell_text = [[table_cell(row.get(key), key) for key, _ in columns] for row in shown]
    fig_height = max(2.2, 0.36 * len(shown) + 1.2)
    fig_width = max(8.0, 1.35 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    table = ax.table(
        cellText=cell_text,
        colLabels=[label for _, label in columns],
        loc="upper center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.92],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f3a4a")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f3f5f7")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def classify_spatial_bucket(target_position: list[float] | tuple[float, ...]) -> str:
    x = float(target_position[0])
    y = float(target_position[1])
    left = x <= 0.40
    right = x >= 0.62
    lower = y <= -0.18
    upper = y >= 0.18
    if left and lower:
        return "lower_left_corner"
    if right and lower:
        return "lower_right_corner"
    if left and upper:
        return "upper_left_corner"
    if right and upper:
        return "upper_right_corner"
    if left:
        return "left_edge"
    if right:
        return "right_edge"
    if lower:
        return "lower_edge"
    if upper:
        return "upper_edge"
    return "interior"


def bucket_group(bucket: str) -> str:
    if bucket == "interior":
        return "interior"
    if bucket.endswith("_corner"):
        return "corner"
    return "edge"


def load_bucket_map(root: Path, eval_dataset_path: str) -> dict[int, str]:
    path = Path(eval_dataset_path)
    if not path.is_absolute():
        path = root / path
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    mapping: dict[int, str] = {}
    for fallback_idx, episode in enumerate(payload["episodes"]):
        episode_index = int(episode.get("episode_index", fallback_idx))
        target_position = episode["initial_scene"]["target_position"]
        mapping[episode_index] = classify_spatial_bucket(target_position)
    return mapping


def summarize_result_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    if count == 0:
        return {}
    row: dict[str, Any] = {
        "rollouts": count,
        "success_0p5cm": float(np.mean([bool(item.get("success_at_0p5cm", False)) for item in results])),
        "success_1cm": float(np.mean([bool(item.get("success_at_1cm", False)) for item in results])),
        "success_2cm": float(np.mean([bool(item.get("success_at_2cm", False)) for item in results])),
        "success_5cm": float(np.mean([bool(item.get("success_at_5cm", False)) for item in results])),
        "mean_best_cm": cm(float(np.mean([float(item["best_distance"]) for item in results]))),
        "mean_final_cm": cm(float(np.mean([float(item["final_distance"]) for item in results]))),
    }
    return row


def bucket_rows(root: Path, specs: list[MetricSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bucket_map_cache: dict[str, dict[int, str]] = {}
    for group, model, split, tag, rel_path in specs:
        path = root / rel_path
        if not path.exists():
            continue
        payload = read_json(path)
        bucket_maps = {}
        for dataset in payload.get("eval_datasets", []):
            eval_path = str(dataset["eval_dataset_path"])
            if eval_path not in bucket_map_cache:
                bucket_map_cache[eval_path] = load_bucket_map(root, eval_path)
            bucket_maps[eval_path] = bucket_map_cache[eval_path]
        detailed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in payload["results"]:
            eval_path = str(result["eval_dataset_path"])
            episode_index = int(result["episode_index"])
            bucket = bucket_maps[eval_path][episode_index]
            detailed[(bucket_group(bucket), bucket)].append(result)
            grouped[bucket_group(bucket)].append(result)
        for (coarse, bucket), bucket_results in sorted(detailed.items()):
            rows.append(
                {
                    "group": group,
                    "model": model,
                    "split": split,
                    "tag": tag,
                    "bucket_group": coarse,
                    "bucket": bucket,
                    **summarize_result_group(bucket_results),
                    "metrics_path": str(path.relative_to(root)),
                }
            )
        for coarse, bucket_results in sorted(grouped.items()):
            rows.append(
                {
                    "group": group,
                    "model": model,
                    "split": split,
                    "tag": tag,
                    "bucket_group": coarse,
                    "bucket": f"all_{coarse}",
                    **summarize_result_group(bucket_results),
                    "metrics_path": str(path.relative_to(root)),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results/structured_analysis"))
    parser.add_argument("--skip-buckets", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        ("group", "Training setup"),
        ("model", "Model"),
        ("split", "Split"),
        ("rollouts", "N"),
        ("success_1cm", "S@1cm"),
        ("success_2cm", "S@2cm"),
        ("success_5cm", "S@5cm"),
        ("mean_best_cm", "Best cm"),
        ("mean_final_cm", "Final cm"),
    ]

    main_rows = collect_rows(root, MAIN_SPECS)
    write_csv(output_dir / "main_comparison.csv", main_rows)
    write_markdown(output_dir / "main_comparison.md", "Main Comparison", main_rows, metric_columns)
    write_table_png(output_dir / "main_comparison.png", "Main comparison", main_rows, metric_columns)

    ablation_rows = collect_rows(root, ABLATION_SPECS)
    write_csv(output_dir / "scratch_structured_ablation.csv", ablation_rows)
    write_markdown(output_dir / "scratch_structured_ablation.md", "Scratch Structured Ablation", ablation_rows, metric_columns)
    write_table_png(output_dir / "scratch_structured_ablation.png", "Scratch structured ablation", ablation_rows, metric_columns)

    if not args.skip_buckets:
        bucket_summary_rows = bucket_rows(root, BUCKET_SPECS)
        bucket_columns = [
            ("group", "Training setup"),
            ("model", "Model"),
            ("split", "Split"),
            ("bucket", "Bucket"),
            ("rollouts", "N"),
            ("success_1cm", "S@1cm"),
            ("success_5cm", "S@5cm"),
            ("mean_best_cm", "Best cm"),
            ("mean_final_cm", "Final cm"),
        ]
        write_csv(output_dir / "spatial_bucket_breakdown.csv", bucket_summary_rows)
        write_markdown(output_dir / "spatial_bucket_breakdown.md", "Spatial Bucket Breakdown", bucket_summary_rows, bucket_columns)
        compact_rows = [row for row in bucket_summary_rows if row["bucket"].startswith("all_")]
        write_table_png(
            output_dir / "spatial_bucket_groups.png",
            "Spatial bucket groups",
            compact_rows,
            bucket_columns,
        )


if __name__ == "__main__":
    main()
