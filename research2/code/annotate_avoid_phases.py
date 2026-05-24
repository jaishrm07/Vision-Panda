#!/usr/bin/env python3
"""Annotate existing avoid_reach datasets with expert waypoint phase metadata."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from collect_dataset import expert_action


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def annotate_dataset(input_path: Path, output_path: Path, overwrite: bool = False) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        with output_path.open("rb") as handle:
            existing = pickle.load(handle)
        return {
            "status": "skipped_existing",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "sample_count": len(existing.get("samples", [])),
        }

    with input_path.open("rb") as handle:
        payload = pickle.load(handle)

    if payload.get("task") != "avoid_reach":
        raise ValueError(f"Expected avoid_reach dataset, got {payload.get('task')}: {input_path}")

    phase_counts: Counter[str] = Counter()
    stage_counts: Counter[int] = Counter()
    controller_state_by_episode: dict[int, dict[str, Any]] = {}
    samples = payload.get("samples", [])
    for sample in samples:
        episode_index = int(sample["episode_index"])
        controller_state = controller_state_by_episode.setdefault(episode_index, {})
        scene = sample["scene"]
        ee_position = np.asarray(sample["robot_state"]["ee_position"], dtype=np.float32)
        cube_position = np.asarray(scene["target_position"], dtype=np.float32)
        obstacle = scene.get("obstacle")
        if obstacle is None:
            raise ValueError(f"avoid_reach sample has no obstacle: {input_path}")

        # Re-run the deterministic expert phase logic on the stored state.
        expert_action("avoid_reach", ee_position, cube_position, obstacle, controller_state)

        phase_payload = {
            "avoid_stage": int(controller_state.get("avoid_stage", -1)),
            "avoid_phase": str(controller_state.get("avoid_phase", "unknown")),
            "avoid_waypoint": controller_state.get("avoid_waypoint"),
            "avoid_route_sign": controller_state.get("avoid_route_sign"),
            "avoid_route_y": controller_state.get("avoid_route_y"),
        }
        sample.update(phase_payload)
        sample.setdefault("expert_action", {}).update(phase_payload)
        phase_counts[phase_payload["avoid_phase"]] += 1
        stage_counts[phase_payload["avoid_stage"]] += 1

    payload["phase_annotation"] = {
        "created_at": utc_now(),
        "source_dataset_path": str(input_path),
        "method": "deterministic_replay_of_avoid_reach_expert_phase_logic",
        "phase_counts": dict(sorted(phase_counts.items())),
        "stage_counts": {str(key): value for key, value in sorted(stage_counts.items())},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_path = output_path.with_suffix(".phase_annotation.json")
    write_json(metadata_path, payload["phase_annotation"])
    return {
        "status": "ok",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "sample_count": len(samples),
        "phase_counts": dict(sorted(phase_counts.items())),
        "stage_counts": {str(key): value for key, value in sorted(stage_counts.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/datasets_128px_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/datasets_128px_v1_phase_annotated"))
    parser.add_argument("--train-configs", nargs="+", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    for train_config in args.train_configs:
        for budget in args.budgets:
            for seed in args.seeds:
                stem = f"dataset__{train_config}__budget{int(budget):03d}__seed{int(seed):03d}.pkl"
                records.append(
                    annotate_dataset(
                        input_path=args.input_dir / stem,
                        output_path=args.output_dir / stem,
                        overwrite=args.overwrite,
                    )
                )

    summary = {
        "created_at": utc_now(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "train_configs": args.train_configs,
        "budgets": [int(value) for value in args.budgets],
        "seeds": [int(value) for value in args.seeds],
        "dataset_count": len(records),
        "records": records,
    }
    summary_path = args.summary_path or args.output_dir / "phase_annotation_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"dataset_count": len(records), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
