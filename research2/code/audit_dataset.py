#!/usr/bin/env python3
"""Audit the collected 128px visual-diversity datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from preview_simulator import load_yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentiles(values: Iterable[float], qs: tuple[float, ...] = (0, 5, 25, 50, 75, 95, 100)) -> dict[str, float | None]:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size == 0:
        return {f"p{q:g}": None for q in qs}
    return {f"p{q:g}": float(np.percentile(values_array, q)) for q in qs}


def basic_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p0": None,
            "p5": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p100": None,
        }
    stats: dict[str, float | int | None] = {
        "count": int(values_array.size),
        "mean": float(values_array.mean()),
        "std": float(values_array.std()),
        "min": float(values_array.min()),
        "max": float(values_array.max()),
    }
    stats.update(percentiles(values_array))
    return stats


def vector_range(vectors: list[list[float]]) -> dict[str, list[float] | int | None]:
    if not vectors:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    array = np.asarray(vectors, dtype=np.float64)
    return {
        "count": int(array.shape[0]),
        "min": array.min(axis=0).astype(float).tolist(),
        "max": array.max(axis=0).astype(float).tolist(),
        "mean": array.mean(axis=0).astype(float).tolist(),
        "std": array.std(axis=0).astype(float).tolist(),
    }


def deterministic_sample_indices(total: int, limit: int) -> list[int]:
    if total <= 0 or limit <= 0:
        return []
    if total <= limit:
        return list(range(total))
    return sorted(set(np.linspace(0, total - 1, limit, dtype=int).tolist()))


def image_quality(sample: dict[str, Any], view: str) -> dict[str, Any]:
    image = np.asarray(sample[view])
    shape_ok = tuple(image.shape) == (128, 128, 3)
    finite_ok = bool(np.isfinite(image).all()) if np.issubdtype(image.dtype, np.number) else False
    numeric = image.astype(np.float32) if finite_ok else np.asarray([], dtype=np.float32)
    if numeric.size == 0:
        return {
            "shape_ok": shape_ok,
            "finite_ok": finite_ok,
            "dtype": str(image.dtype),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "nonblank": False,
        }
    std = float(numeric.std())
    return {
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "dtype": str(image.dtype),
        "mean": float(numeric.mean()),
        "std": std,
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "nonblank": std > 1.0,
    }


def scene_from_episode(episode: dict[str, Any]) -> dict[str, Any]:
    return dict(episode.get("initial_scene") or {})


def collect_scene_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    target_positions: list[list[float]] = []
    camera_yaws: list[float] = []
    camera_pitches: list[float] = []
    camera_distances: list[float] = []
    lighting_ambient: list[float] = []
    lighting_diffuse: list[float] = []
    lighting_specular: list[float] = []
    lighting_dirs: list[list[float]] = []
    colors = Counter()
    lighting_modes = Counter()
    obstacle_present = Counter()

    for episode in episodes:
        scene = scene_from_episode(episode)
        target_position = scene.get("target_position")
        if target_position is not None:
            target_positions.append([float(item) for item in target_position])
        color_name = scene.get("target_color_name")
        if color_name:
            colors[str(color_name)] += 1

        camera = scene.get("camera") or {}
        for key, target in (
            ("yaw", camera_yaws),
            ("pitch", camera_pitches),
            ("distance", camera_distances),
        ):
            if key in camera:
                target.append(float(camera[key]))

        lighting = scene.get("lighting")
        if lighting is None:
            lighting_modes["pybullet_default"] += 1
        else:
            lighting_modes["explicit"] += 1
            if "ambient" in lighting:
                lighting_ambient.append(float(lighting["ambient"]))
            if "diffuse" in lighting:
                lighting_diffuse.append(float(lighting["diffuse"]))
            if "specular" in lighting:
                lighting_specular.append(float(lighting["specular"]))
            if "light_direction" in lighting:
                lighting_dirs.append([float(item) for item in lighting["light_direction"]])

        obstacle_present["yes" if scene.get("obstacle") else "no"] += 1

    return {
        "target_position": vector_range(target_positions),
        "target_color_counts": dict(sorted(colors.items())),
        "camera_yaw": basic_stats(camera_yaws),
        "camera_pitch": basic_stats(camera_pitches),
        "camera_distance": basic_stats(camera_distances),
        "lighting_mode_counts": dict(sorted(lighting_modes.items())),
        "lighting_ambient": basic_stats(lighting_ambient),
        "lighting_diffuse": basic_stats(lighting_diffuse),
        "lighting_specular": basic_stats(lighting_specular),
        "lighting_direction": vector_range(lighting_dirs),
        "obstacle_present_counts": dict(sorted(obstacle_present.items())),
    }


def audit_dataset_file(path: Path, image_sample_limit: int) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    samples = payload["samples"]
    episodes = payload["episodes"]
    action_magnitudes = [
        float(np.linalg.norm(np.asarray(sample["expert_action"]["delta_position"], dtype=np.float64)))
        for sample in samples
    ]
    task_distances = [float(sample["task_distance"]) for sample in samples]
    trajectory_lengths = [int(episode["num_steps"]) for episode in episodes]
    success_count = int(sum(1 for episode in episodes if episode["success"]))
    success_rate = float(success_count / len(episodes)) if episodes else 0.0

    image_indices = deterministic_sample_indices(len(samples), image_sample_limit)
    view_image_stats: dict[str, list[dict[str, Any]]] = {"external_rgb": [], "eef_rgb": []}
    for sample_index in image_indices:
        sample = samples[sample_index]
        for view in view_image_stats:
            view_image_stats[view].append(image_quality(sample, view))

    image_summary = {}
    flags: list[str] = []
    for view, records in view_image_stats.items():
        means = [record["mean"] for record in records if record["mean"] is not None]
        stds = [record["std"] for record in records if record["std"] is not None]
        bad_shape_count = sum(1 for record in records if not record["shape_ok"])
        bad_finite_count = sum(1 for record in records if not record["finite_ok"])
        blank_count = sum(1 for record in records if not record["nonblank"])
        if bad_shape_count:
            flags.append(f"{view}:bad_shape_count={bad_shape_count}")
        if bad_finite_count:
            flags.append(f"{view}:bad_finite_count={bad_finite_count}")
        if blank_count:
            flags.append(f"{view}:blank_count={blank_count}")
        image_summary[view] = {
            "sampled_image_count": len(records),
            "bad_shape_count": bad_shape_count,
            "bad_finite_count": bad_finite_count,
            "blank_count": blank_count,
            "mean_brightness": basic_stats(means),
            "std_contrast": basic_stats(stds),
        }

    if len(samples) == 0:
        flags.append("empty_samples")
    if len(episodes) != int(payload["budget"]):
        flags.append(f"episode_count_mismatch={len(episodes)}_expected_{payload['budget']}")
    if not math.isclose(success_rate, 1.0):
        flags.append(f"success_rate={success_rate:.4f}")
    if max(action_magnitudes, default=0.0) > 1.0001:
        flags.append(f"action_magnitude_over_one={max(action_magnitudes):.4f}")

    scene_stats = collect_scene_stats(episodes)
    record = {
        "file": str(path),
        "train_config": payload["train_config"],
        "task": payload["task"],
        "axis": payload["axis"],
        "variant": payload["variant"],
        "budget": int(payload["budget"]),
        "seed": int(payload["seed"]),
        "num_samples": len(samples),
        "num_episodes": len(episodes),
        "success_count": success_count,
        "success_rate": success_rate,
        "trajectory_length": basic_stats(trajectory_lengths),
        "action_magnitude": basic_stats(action_magnitudes),
        "task_distance": basic_stats(task_distances),
        "scene": scene_stats,
        "image_quality": image_summary,
        "flags": flags,
    }
    row = {
        "train_config": record["train_config"],
        "task": record["task"],
        "axis": record["axis"],
        "variant": record["variant"],
        "budget": record["budget"],
        "seed": record["seed"],
        "num_samples": record["num_samples"],
        "num_episodes": record["num_episodes"],
        "success_rate": record["success_rate"],
        "trajectory_len_mean": record["trajectory_length"]["mean"],
        "trajectory_len_p50": record["trajectory_length"]["p50"],
        "trajectory_len_p95": record["trajectory_length"]["p95"],
        "action_mag_mean": record["action_magnitude"]["mean"],
        "action_mag_p95": record["action_magnitude"]["p95"],
        "external_brightness_mean": record["image_quality"]["external_rgb"]["mean_brightness"]["mean"],
        "external_contrast_mean": record["image_quality"]["external_rgb"]["std_contrast"]["mean"],
        "eef_brightness_mean": record["image_quality"]["eef_rgb"]["mean_brightness"]["mean"],
        "eef_contrast_mean": record["image_quality"]["eef_rgb"]["std_contrast"]["mean"],
        "flag_count": len(flags),
        "flags": ";".join(flags),
    }
    return record, row, flags


def aggregate_records(records: list[dict[str, Any]], expected_dataset_count: int, metadata_count: int) -> dict[str, Any]:
    sample_counts = [record["num_samples"] for record in records]
    episode_counts = [record["num_episodes"] for record in records]
    success_rates = [record["success_rate"] for record in records]
    flags = {record["train_config"] + f"__budget{record['budget']:03d}__seed{record['seed']:03d}": record["flags"] for record in records if record["flags"]}

    by_axis: dict[str, dict[str, Any]] = {}
    by_task: dict[str, dict[str, Any]] = {}
    by_variant: dict[str, dict[str, Any]] = {}
    for group_name, target in (("axis", by_axis), ("task", by_task), ("variant", by_variant)):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record[group_name])].append(record)
        for key, items in sorted(grouped.items()):
            target[key] = {
                "dataset_count": len(items),
                "total_samples": int(sum(item["num_samples"] for item in items)),
                "total_episodes": int(sum(item["num_episodes"] for item in items)),
                "success_rate_min": float(min(item["success_rate"] for item in items)),
                "success_rate_max": float(max(item["success_rate"] for item in items)),
                "sample_count": basic_stats(item["num_samples"] for item in items),
            }

    return {
        "dataset_file_count": len(records),
        "metadata_file_count": metadata_count,
        "expected_dataset_count": expected_dataset_count,
        "total_samples": int(sum(sample_counts)),
        "total_episodes": int(sum(episode_counts)),
        "sample_count_per_dataset": basic_stats(sample_counts),
        "episode_count_per_dataset": basic_stats(episode_counts),
        "success_rate": basic_stats(success_rates),
        "all_success_rates_are_one": all(math.isclose(rate, 1.0) for rate in success_rates),
        "datasets_with_flags": flags,
        "flagged_dataset_count": len(flags),
        "by_axis": by_axis,
        "by_task": by_task,
        "by_variant": by_variant,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text_summary(path: Path, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    lines = [
        "Dataset Audit Summary",
        f"generated_at: {summary['generated_at']}",
        f"dataset_dir: {summary['dataset_dir']}",
        "",
        f"dataset files: {aggregate['dataset_file_count']} / {aggregate['expected_dataset_count']}",
        f"metadata files: {aggregate['metadata_file_count']}",
        f"total samples: {aggregate['total_samples']}",
        f"total episodes: {aggregate['total_episodes']}",
        f"all success rates are 1.0: {aggregate['all_success_rates_are_one']}",
        f"flagged datasets: {aggregate['flagged_dataset_count']}",
        "",
        "Sample Count Per Dataset:",
        json.dumps(aggregate["sample_count_per_dataset"], indent=2),
        "",
        "By Axis:",
        json.dumps(aggregate["by_axis"], indent=2),
        "",
        "By Task:",
        json.dumps(aggregate["by_task"], indent=2),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_names(config: dict[str, Any], requested_names: list[str] | None) -> list[str]:
    all_names = [item["name"] for item in config["train_configs"]]
    all_names.extend(item["name"] for item in config.get("eval_configs", []))
    if requested_names is None:
        return [item["name"] for item in config["train_configs"]]
    unknown = sorted(set(requested_names) - set(all_names))
    if unknown:
        raise ValueError(f"Unknown config names: {unknown}")
    return requested_names


def audit(
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    image_sample_limit: int,
    requested_config_names: list[str] | None,
    budgets: list[int] | None,
    seeds: list[int] | None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = sorted(dataset_dir.glob("dataset__*.pkl"))
    metadata_count = len(list(dataset_dir.glob("dataset__*.json")))
    names = config_names(config, requested_config_names)
    budgets = budgets if budgets is not None else [int(item) for item in config["collection"]["budgets"]]
    seeds = seeds if seeds is not None else [int(item) for item in config["collection"]["seeds"]]
    expected_dataset_count = len(names) * len(budgets) * len(seeds)

    records = []
    rows = []
    all_flags: list[dict[str, Any]] = []
    for path in dataset_paths:
        record, row, flags = audit_dataset_file(path, image_sample_limit=image_sample_limit)
        records.append(record)
        rows.append(row)
        if flags:
            all_flags.append({"dataset": path.name, "flags": flags})

    summary = {
        "generated_at": utc_now(),
        "config_path": str(config_path),
        "dataset_dir": str(dataset_dir),
        "image_sample_limit_per_dataset": int(image_sample_limit),
        "aggregate": aggregate_records(records, expected_dataset_count, metadata_count),
        "flags": all_flags,
        "datasets": records,
    }
    (output_dir / "dataset_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "dataset_audit_per_dataset.csv", rows)
    write_text_summary(output_dir / "dataset_audit_summary.txt", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_128px_v1.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("results/datasets_128px_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dataset_audit_128px_v1"))
    parser.add_argument("--image-sample-limit-per-dataset", type=int, default=200)
    parser.add_argument("--train-configs", nargs="+")
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit(
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        image_sample_limit=args.image_sample_limit_per_dataset,
        requested_config_names=args.train_configs,
        budgets=args.budgets,
        seeds=args.seeds,
    )
    aggregate = summary["aggregate"]
    print(
        json.dumps(
            {
                "dataset_file_count": aggregate["dataset_file_count"],
                "expected_dataset_count": aggregate["expected_dataset_count"],
                "metadata_file_count": aggregate["metadata_file_count"],
                "total_samples": aggregate["total_samples"],
                "total_episodes": aggregate["total_episodes"],
                "all_success_rates_are_one": aggregate["all_success_rates_are_one"],
                "flagged_dataset_count": aggregate["flagged_dataset_count"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
