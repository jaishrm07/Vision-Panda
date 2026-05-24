#!/usr/bin/env python3
"""Combine per-seed closed-loop eval JSON files into one metrics JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_bc import normalize_success_thresholds, summarize_results
from train_bc import write_json


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def combine_metrics(inputs: list[Path], output: Path) -> dict[str, Any]:
    if not inputs:
        raise ValueError("Pass at least one input metrics file.")
    payloads = [read_json(path) for path in inputs]
    combined = dict(payloads[0])
    combined["combined_from"] = [str(path) for path in inputs]
    combined["eval_datasets"] = []
    combined["results"] = []
    for payload in payloads:
        combined["eval_datasets"].extend(payload.get("eval_datasets", []))
        combined["results"].extend(payload.get("results", []))
    threshold_items = payloads[0].get("success_thresholds", {})
    thresholds = [float(value) for _, value in threshold_items.items()] if isinstance(threshold_items, dict) else None
    combined["metrics"] = summarize_results(combined["results"], normalize_success_thresholds(thresholds))
    write_json(output, combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = combine_metrics(args.inputs, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "input_count": len(args.inputs),
                "rollout_count": summary["metrics"]["rollout_count"],
                "success_rate_at_1cm": summary["metrics"].get("success_rate_at_1cm"),
                "success_rate_at_5cm": summary["metrics"].get("success_rate_at_5cm"),
                "mean_best_distance": summary["metrics"].get("mean_best_distance"),
                "mean_final_distance": summary["metrics"].get("mean_final_distance"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
