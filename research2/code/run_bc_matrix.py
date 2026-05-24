#!/usr/bin/env python3
"""Run a worker shard of the research2 BC training matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from preview_simulator import load_yaml
from train_bc import MODEL_FAMILY, dataset_stem, model_stem, train_one


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def enumerate_tasks(config: dict[str, Any], train_config_names: list[str] | None, budgets: list[int] | None, seeds: list[int] | None) -> list[dict[str, Any]]:
    available = {item["name"]: item for item in config["train_configs"]}
    names = train_config_names if train_config_names is not None else [item["name"] for item in config["train_configs"]]
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown train configs: {unknown}")
    resolved_budgets = budgets if budgets is not None else [int(item) for item in config["collection"]["budgets"]]
    resolved_seeds = seeds if seeds is not None else [int(item) for item in config["collection"]["seeds"]]
    tasks = []
    for name in names:
        for budget in resolved_budgets:
            for seed in resolved_seeds:
                tasks.append({"train_config": name, "budget": int(budget), "seed": int(seed)})
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_128px_v1.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("results/datasets_128px_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/bc_128px_v1"))
    parser.add_argument("--model-family", default=MODEL_FAMILY)
    parser.add_argument("--train-configs", nargs="+")
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-data-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--phase-balance", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--status-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    tasks = enumerate_tasks(config, args.train_configs, args.budgets, args.seeds)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    shard = [task for index, task in enumerate(tasks) if index % args.num_workers == args.worker_index]
    status_records = []
    if args.status_path is not None:
        append_log(args.status_path, f"worker_index\t{args.worker_index}\ttask_count\t{len(shard)}")

    for task in shard:
        train_config = task["train_config"]
        budget = task["budget"]
        seed = task["seed"]
        dataset_path = args.dataset_dir / f"{dataset_stem(train_config, budget, seed)}.pkl"
        stem = model_stem(args.model_family, train_config, budget, seed)
        model_path = args.output_dir / "models" / f"{stem}.pt"
        history_path = args.output_dir / "histories" / f"{stem}.json"
        if model_path.exists() and history_path.exists() and not args.overwrite:
            record = {"status": "skipped_existing", **task, "model_path": str(model_path), "history_path": str(history_path)}
            status_records.append(record)
            if args.status_path is not None:
                append_log(args.status_path, json.dumps(record, sort_keys=True))
            continue

        try:
            summary = train_one(
                dataset_path=dataset_path,
                output_dir=args.output_dir,
                model_family=args.model_family,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                backbone_lr=args.backbone_lr,
                weight_decay=args.weight_decay,
                num_workers=args.num_data_workers,
                train_seed=seed,
                device_name=args.device,
                max_samples=args.max_samples,
                val_fraction=args.val_fraction,
                phase_balance=args.phase_balance,
                checkpoint_every=args.checkpoint_every,
                overwrite=args.overwrite,
            )
            record = {
                "status": "ok",
                "model_family": args.model_family,
                **task,
                "model_path": summary["model_path"],
                "history_path": summary["history_path"],
                "final_train_loss": summary.get("final_train_loss"),
            }
        except Exception as exc:
            record = {"status": "error", **task, "error": repr(exc)}
            status_records.append(record)
            if args.status_path is not None:
                append_log(args.status_path, json.dumps(record, sort_keys=True))
            raise
        status_records.append(record)
        if args.status_path is not None:
            append_log(args.status_path, json.dumps(record, sort_keys=True))

    print(json.dumps({"worker_index": args.worker_index, "task_count": len(shard), "records": status_records}, indent=2))


if __name__ == "__main__":
    main()
