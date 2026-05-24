#!/usr/bin/env python3
"""Plot research2 BC aggregate metrics for analysis and project video figures."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = ("reach", "avoid_reach")
AXES = ("color", "spatial_distribution", "camera_location_viewpoint", "lighting_direction_intensity")
AXIS_LABELS = {
    "color": "Color",
    "spatial_distribution": "Spatial",
    "camera_location_viewpoint": "Camera",
    "lighting_direction_intensity": "Lighting",
}
TASK_LABELS = {
    "reach": "Reach",
    "avoid_reach": "Avoid Reach",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def to_int(value: str | None) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def group_label(config_name: str) -> str:
    return (
        config_name.replace("avoid_", "")
        .replace("_red_only", " fixed")
        .replace("_multi_pose", " multi")
        .replace("_multi", " multi")
        .replace("_narrow", " narrow")
        .replace("_wide", " wide")
        .replace("_fixed", " fixed")
        .replace("_diverse", " diverse")
        .replace("_", " ")
        .title()
    )


def plot_success(aggregate_rows: list[dict[str, str]], eval_split: str, output_path: Path) -> None:
    fig, axes = plt.subplots(len(TASKS), len(AXES), figsize=(16, 7), sharex=True, sharey=True)
    for row_idx, task in enumerate(TASKS):
        for col_idx, axis_name in enumerate(AXES):
            ax = axes[row_idx][col_idx]
            subset = [
                row
                for row in aggregate_rows
                if row["eval_split"] == eval_split and row["task"] == task and row["axis"] == axis_name
            ]
            configs = sorted({row["train_config"] for row in subset})
            for config_name in configs:
                rows = sorted([row for row in subset if row["train_config"] == config_name], key=lambda item: to_int(item["budget"]) or 0)
                budgets = [to_int(row["budget"]) for row in rows]
                means = [to_float(row["success_rate_mean"]) for row in rows]
                ci95 = [to_float(row["success_rate_ci95"]) or 0.0 for row in rows]
                if budgets and means:
                    ax.errorbar(budgets, means, yerr=ci95, marker="o", linewidth=1.8, capsize=3, label=group_label(config_name))
            ax.set_title(f"{TASK_LABELS[task]} / {AXIS_LABELS[axis_name]}", fontsize=10)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xticks([5, 20, 50])
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("Success rate")
            if row_idx == len(TASKS) - 1:
                ax.set_xlabel("Training demos")
            if subset:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{eval_split.upper()} Success vs Budget", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gap(gap_rows: list[dict[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(TASKS), len(AXES), figsize=(16, 7), sharex=True)
    for row_idx, task in enumerate(TASKS):
        for col_idx, axis_name in enumerate(AXES):
            ax = axes[row_idx][col_idx]
            subset = [row for row in gap_rows if row["task"] == task and row["axis"] == axis_name]
            configs = sorted({row["train_config"] for row in subset})
            for config_name in configs:
                rows = sorted([row for row in subset if row["train_config"] == config_name], key=lambda item: to_int(item["budget"]) or 0)
                budgets = [to_int(row["budget"]) for row in rows]
                values = [to_float(row["generalization_gap"]) for row in rows]
                if budgets and values:
                    ax.plot(budgets, values, marker="o", linewidth=1.8, label=group_label(config_name))
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{TASK_LABELS[task]} / {AXIS_LABELS[axis_name]}", fontsize=10)
            ax.set_xticks([5, 20, 50])
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("ID - OOD success")
            if row_idx == len(TASKS) - 1:
                ax.set_xlabel("Training demos")
            if subset:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("Generalization Gap vs Budget", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_diversity_gain(gain_rows: list[dict[str, str]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(TASKS), len(AXES), figsize=(16, 7), sharex=True, sharey=True)
    for row_idx, task in enumerate(TASKS):
        for col_idx, axis_name in enumerate(AXES):
            ax = axes[row_idx][col_idx]
            rows = sorted(
                [row for row in gain_rows if row["task"] == task and row["axis"] == axis_name],
                key=lambda item: to_int(item["budget"]) or 0,
            )
            budgets = [to_int(row["budget"]) for row in rows]
            values = [to_float(row["diversity_gain"]) for row in rows]
            if budgets and values:
                ax.plot(budgets, values, marker="o", linewidth=2.2)
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{TASK_LABELS[task]} / {AXIS_LABELS[axis_name]}", fontsize=10)
            ax.set_xticks([5, 20, 50])
            ax.grid(True, alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("OOD diversity gain")
            if row_idx == len(TASKS) - 1:
                ax.set_xlabel("Training demos")
    fig.suptitle("OOD Diversity Gain vs Budget", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, default=Path("results/bc_128px_v1/aggregates"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/bc_128px_v1/plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_rows = read_csv(args.aggregate_dir / "aggregate_metrics.csv")
    gap_rows = read_csv(args.aggregate_dir / "generalization_gaps.csv")
    gain_rows = read_csv(args.aggregate_dir / "diversity_gains.csv")
    outputs = {
        "id_success_vs_budget": args.output_dir / "id_success_vs_budget.png",
        "ood_success_vs_budget": args.output_dir / "ood_success_vs_budget.png",
        "generalization_gap_vs_budget": args.output_dir / "generalization_gap_vs_budget.png",
        "diversity_gain_vs_budget": args.output_dir / "diversity_gain_vs_budget.png",
    }
    plot_success(aggregate_rows, "id", outputs["id_success_vs_budget"])
    plot_success(aggregate_rows, "ood", outputs["ood_success_vs_budget"])
    plot_gap(gap_rows, outputs["generalization_gap_vs_budget"])
    plot_diversity_gain(gain_rows, outputs["diversity_gain_vs_budget"])
    summary = {
        "created_at": utc_now(),
        "aggregate_dir": str(args.aggregate_dir),
        "output_dir": str(args.output_dir),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
