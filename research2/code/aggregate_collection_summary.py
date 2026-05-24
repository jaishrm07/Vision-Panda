#!/usr/bin/env python3
"""Aggregate per-dataset metadata into the final collection summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preview_simulator import load_yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_key(record: dict[str, Any]) -> tuple[str, int, int]:
    return (str(record["train_config"]), int(record["budget"]), int(record["seed"]))


def available_config_names(config: dict[str, Any]) -> list[str]:
    names = [item["name"] for item in config["train_configs"]]
    names.extend(item["name"] for item in config.get("eval_configs", []))
    return names


def build_expected(
    config: dict[str, Any],
    names: list[str] | None,
    budgets: list[int] | None,
    seeds: list[int] | None,
) -> list[tuple[str, int, int]]:
    if names is None:
        names = [item["name"] for item in config["train_configs"]]
    unknown = sorted(set(names) - set(available_config_names(config)))
    if unknown:
        raise ValueError(f"Unknown config names: {unknown}")
    budgets = budgets if budgets is not None else [int(item) for item in config["collection"]["budgets"]]
    seeds = seeds if seeds is not None else [int(item) for item in config["collection"]["seeds"]]
    return [(name, budget, seed) for name in names for budget in budgets for seed in seeds]


def load_records(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for metadata_path in sorted(output_dir.glob("dataset__*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records.append(
            {
                "train_config": metadata["train_config"],
                "task": metadata["task"],
                "axis": metadata["axis"],
                "variant": metadata["variant"],
                "budget": int(metadata["budget"]),
                "seed": int(metadata["seed"]),
                "num_samples": int(metadata["num_samples"]),
                "num_episodes": int(metadata["num_episodes"]),
                "success_rate": float(metadata["success_rate"]),
                "sample_file": metadata["sample_file"],
                "metadata_file": str(metadata_path),
                "quality_checks": metadata["quality_checks"],
            }
        )
    return records


def aggregate(
    config_path: Path,
    output_dir: Path,
    summary_path: Path,
    allow_incomplete: bool,
    train_config_names: list[str] | None,
    budgets: list[int] | None,
    seeds: list[int] | None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    expected = build_expected(config, names=train_config_names, budgets=budgets, seeds=seeds)
    expected_set = set(expected)
    records = load_records(output_dir)
    observed_set = {record_key(record) for record in records}
    missing = [list(item) for item in expected if item not in observed_set]
    unexpected = [list(item) for item in sorted(observed_set - expected_set)]

    ordered_names = [item["name"] for item in config["train_configs"]]
    ordered_names.extend(item["name"] for item in config.get("eval_configs", []))
    config_order = {name: index for index, name in enumerate(ordered_names)}
    records.sort(key=lambda item: (config_order.get(item["train_config"], 999), item["budget"], item["seed"]))
    all_quality_passed = all(all(bool(value) for value in record["quality_checks"].values()) for record in records)

    summary = {
        "version": config["version"],
        "created_at": utc_now(),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "expected_dataset_count": len(expected),
        "completed_dataset_count": len(records),
        "missing_dataset_count": len(missing),
        "unexpected_dataset_count": len(unexpected),
        "budgets": sorted({int(item[1]) for item in expected}),
        "seeds": sorted({int(item[2]) for item in expected}),
        "train_configs": list(dict.fromkeys(item[0] for item in expected)),
        "total_samples": int(sum(record["num_samples"] for record in records)),
        "all_quality_checks_passed": all_quality_passed,
        "all_success_rates_are_one": all(record["success_rate"] == 1.0 for record in records),
        "missing": missing,
        "unexpected": unexpected,
        "records": records,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if (missing or unexpected or not all_quality_passed) and not allow_incomplete:
        raise SystemExit(1)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_128px_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/datasets_128px_v1"))
    parser.add_argument("--summary-path", type=Path, default=Path("results/datasets_128px_v1/collection_summary.json"))
    parser.add_argument("--train-configs", nargs="+")
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate(
        config_path=args.config,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        allow_incomplete=args.allow_incomplete,
        train_config_names=args.train_configs,
        budgets=args.budgets,
        seeds=args.seeds,
    )
    print(
        json.dumps(
            {
                "completed_dataset_count": summary["completed_dataset_count"],
                "missing_dataset_count": summary["missing_dataset_count"],
                "total_samples": summary["total_samples"],
                "summary_path": str(args.summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
