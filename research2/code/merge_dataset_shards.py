#!/usr/bin/env python3
"""Merge sharded research2 dataset pickles back into canonical dataset files."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_stem(train_config: str, budget: int, seed: int) -> str:
    return f"dataset__{train_config}__budget{int(budget):03d}__seed{int(seed):03d}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def merge_one(input_dir: Path, output_dir: Path, train_config: str, budget: int, seed: int, shard_count: int) -> dict[str, Any]:
    stem = dataset_stem(train_config, budget, seed)
    shard_paths = [
        input_dir / f"{stem}__shard{shard_index:02d}_of{shard_count:02d}.pkl"
        for shard_index in range(shard_count)
    ]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard(s): {missing}")

    shard_payloads = [load_pickle(path) for path in shard_paths]
    first = shard_payloads[0]
    samples = []
    episodes = []
    phase_counts: Counter[str] = Counter()
    for payload in shard_payloads:
        if payload.get("train_config") != train_config or int(payload.get("budget")) != int(budget) or int(payload.get("seed")) != int(seed):
            raise ValueError(f"Shard metadata mismatch for {train_config} budget={budget} seed={seed}")
        samples.extend(payload.get("samples", []))
        episodes.extend(payload.get("episodes", []))
    samples.sort(key=lambda item: (int(item["episode_index"]), int(item["step_index"])))
    episodes.sort(key=lambda item: int(item["episode_index"]))
    episode_indices = [int(item["episode_index"]) for item in episodes]
    expected_indices = list(range(int(budget)))
    if episode_indices != expected_indices:
        raise ValueError(f"Merged episodes for {train_config} do not cover 0..{int(budget) - 1}.")

    for sample in samples:
        if sample.get("avoid_phase") is not None:
            phase_counts[str(sample["avoid_phase"])] += 1

    merged = {
        key: first.get(key)
        for key in ("version", "train_config", "task", "axis", "variant", "budget", "seed", "resolution")
    }
    merged["samples"] = samples
    merged["episodes"] = episodes
    merged["phase_annotation"] = {
        "created_at": utc_now(),
        "method": "direct_collection_with_avoid_phase_logging_and_shard_merge",
        "source_shards": [str(path) for path in shard_paths],
        "phase_counts": dict(sorted(phase_counts.items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pkl = output_dir / f"{stem}.pkl"
    with output_pkl.open("wb") as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)

    success_count = int(sum(1 for item in episodes if item.get("success")))
    metadata = {
        "version": merged["version"],
        "created_at": utc_now(),
        "train_config": train_config,
        "task": merged["task"],
        "axis": merged["axis"],
        "variant": merged["variant"],
        "budget": int(budget),
        "seed": int(seed),
        "num_samples": len(samples),
        "num_episodes": len(episodes),
        "success_count": success_count,
        "success_rate": float(success_count / len(episodes)) if episodes else 0.0,
        "sample_file": str(output_pkl),
        "source_shards": [str(path) for path in shard_paths],
        "phase_counts": dict(sorted(phase_counts.items())),
        "episodes": episodes,
    }
    output_json = output_dir / f"{stem}.json"
    write_json(output_json, metadata)
    return {
        "train_config": train_config,
        "budget": int(budget),
        "seed": int(seed),
        "num_samples": len(samples),
        "num_episodes": len(episodes),
        "success_rate": metadata["success_rate"],
        "phase_counts": dict(sorted(phase_counts.items())),
        "sample_file": str(output_pkl),
        "metadata_file": str(output_json),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-configs", nargs="+", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--summary-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        merge_one(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            train_config=train_config,
            budget=args.budget,
            seed=args.seed,
            shard_count=args.shard_count,
        )
        for train_config in args.train_configs
    ]
    summary = {
        "created_at": utc_now(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "budget": int(args.budget),
        "seed": int(args.seed),
        "shard_count": int(args.shard_count),
        "dataset_count": len(records),
        "records": records,
    }
    summary_path = args.summary_path or args.output_dir / "shard_merge_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"dataset_count": len(records), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
