#!/usr/bin/env python3
"""Run a worker shard of the research2 BC ID/OOD closed-loop eval matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_bc import evaluate_policy
from preview_simulator import load_yaml
from train_bc import MODEL_FAMILY, dataset_stem, model_stem


OOD_EVAL_BY_TRAIN_CONFIG = {
    "color_red_only": "color_ood_eval",
    "color_multi": "color_ood_eval",
    "avoid_color_red_only": "avoid_color_ood_eval",
    "avoid_color_multi": "avoid_color_ood_eval",
    "spatial_narrow": "spatial_edge_ood_eval",
    "spatial_wide": "spatial_edge_ood_eval",
    "avoid_spatial_narrow": "avoid_spatial_edge_ood_eval",
    "avoid_spatial_wide": "avoid_spatial_edge_ood_eval",
    "camera_fixed": "camera_extreme_ood_eval",
    "camera_multi_pose": "camera_extreme_ood_eval",
    "avoid_camera_fixed": "avoid_camera_extreme_ood_eval",
    "avoid_camera_multi_pose": "avoid_camera_extreme_ood_eval",
    "lighting_fixed": "lighting_extreme_ood_eval",
    "lighting_diverse": "lighting_extreme_ood_eval",
    "avoid_lighting_fixed": "avoid_lighting_extreme_ood_eval",
    "avoid_lighting_diverse": "avoid_lighting_extreme_ood_eval",
}


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def enumerate_models(config: dict[str, Any], train_config_names: list[str] | None, budgets: list[int] | None, seeds: list[int] | None) -> list[dict[str, Any]]:
    names = train_config_names if train_config_names is not None else [item["name"] for item in config["train_configs"]]
    available = {item["name"] for item in config["train_configs"]}
    unknown = sorted(set(names) - available)
    if unknown:
        raise ValueError(f"Unknown train configs: {unknown}")
    resolved_budgets = budgets if budgets is not None else [int(item) for item in config["collection"]["budgets"]]
    resolved_seeds = seeds if seeds is not None else [int(item) for item in config["collection"]["seeds"]]
    return [
        {"train_config": name, "budget": int(budget), "seed": int(seed)}
        for name in names
        for budget in resolved_budgets
        for seed in resolved_seeds
    ]


def eval_config_for(split: str, train_config: str) -> str:
    if split == "id":
        return train_config
    if split == "ood":
        return OOD_EVAL_BY_TRAIN_CONFIG[train_config]
    raise ValueError(f"Unsupported eval split: {split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_128px_v1.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("results/bc_128px_v1/models"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/bc_128px_v1/metrics"))
    parser.add_argument("--model-family", default=MODEL_FAMILY)
    parser.add_argument("--eval-split", choices=("id", "ood"), required=True)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--eval-seeds", type=int, nargs="+")
    parser.add_argument("--train-configs", nargs="+")
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps-per-episode", type=int, default=400)
    parser.add_argument("--max-episodes-per-dataset", type=int)
    parser.add_argument("--success-thresholds", type=float, nargs="+")
    parser.add_argument("--stop-threshold", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--status-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    eval_dir = args.eval_dir
    if eval_dir is None:
        eval_dir = Path("results/eval_id_128px_v1" if args.eval_split == "id" else "results/eval_ood_128px_v1")
    eval_seeds = args.eval_seeds
    if eval_seeds is None:
        eval_seeds = [200, 201, 202] if args.eval_split == "id" else [100, 101, 102]

    tasks = enumerate_models(config, args.train_configs, args.budgets, args.seeds)
    shard = [task for index, task in enumerate(tasks) if index % args.num_workers == args.worker_index]
    status_records = []
    if args.status_path is not None:
        append_log(args.status_path, f"worker_index\t{args.worker_index}\teval_split\t{args.eval_split}\ttask_count\t{len(shard)}")

    for task in shard:
        train_config = task["train_config"]
        budget = task["budget"]
        seed = task["seed"]
        model_path = args.model_dir / f"{model_stem(args.model_family, train_config, budget, seed)}.pt"
        eval_config = eval_config_for(args.eval_split, train_config)
        metrics_path = (
            args.metrics_dir
            / args.eval_split
            / f"metrics__{args.model_family}__{train_config}__budget{budget:03d}__seed{seed:03d}__eval_{eval_config}.json"
        )
        if metrics_path.exists() and not args.overwrite:
            record = {"status": "skipped_existing", **task, "eval_split": args.eval_split, "metrics_path": str(metrics_path)}
            status_records.append(record)
            if args.status_path is not None:
                append_log(args.status_path, json.dumps(record, sort_keys=True))
            continue
        if not model_path.exists():
            record = {"status": "missing_model", **task, "model_path": str(model_path)}
            status_records.append(record)
            if args.status_path is not None:
                append_log(args.status_path, json.dumps(record, sort_keys=True))
            continue

        eval_dataset_paths = [
            eval_dir / f"{dataset_stem(eval_config, args.eval_budget, eval_seed)}.pkl"
            for eval_seed in eval_seeds
        ]
        try:
            summary = evaluate_policy(
                model_path=model_path,
                eval_dataset_paths=eval_dataset_paths,
                metrics_path=metrics_path,
                device_name=args.device,
                steps_per_episode=args.steps_per_episode,
                max_episodes_per_dataset=args.max_episodes_per_dataset,
                success_thresholds=args.success_thresholds,
                stop_threshold=args.stop_threshold,
            )
            record = {
                "status": "ok",
                **task,
                "eval_split": args.eval_split,
                "eval_config": eval_config,
                "metrics_path": str(metrics_path),
                **summary["metrics"],
            }
        except Exception as exc:
            record = {"status": "error", **task, "eval_split": args.eval_split, "eval_config": eval_config, "error": repr(exc)}
            status_records.append(record)
            if args.status_path is not None:
                append_log(args.status_path, json.dumps(record, sort_keys=True))
            raise
        status_records.append(record)
        if args.status_path is not None:
            append_log(args.status_path, json.dumps(record, sort_keys=True))

    print(json.dumps({"worker_index": args.worker_index, "eval_split": args.eval_split, "task_count": len(shard), "records": status_records}, indent=2))


if __name__ == "__main__":
    main()
