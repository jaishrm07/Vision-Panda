#!/usr/bin/env python3
"""Generate best/worst rollout video case studies from existing eval metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CaseSpec:
    group: str
    task: str
    axis: str
    label: str
    metrics_path: str
    note: str


VISUAL_CASES = [
    CaseSpec(
        "visual_axes",
        "reach",
        "color",
        "reach_color_multi",
        "results/bc_128px_v1/metrics/ood/metrics__scratch_bc_128__color_multi__budget200__seed000__eval_color_ood_eval.json",
        "Scratch CNN, budget 200, color-diverse training, color OOD eval.",
    ),
    CaseSpec(
        "visual_axes",
        "reach",
        "spatial_distribution",
        "reach_spatial_wide",
        "results/bc_128px_v1/metrics/ood/metrics__scratch_bc_128__spatial_wide__budget200__seed000__eval_spatial_edge_ood_eval.json",
        "Scratch CNN, budget 200, spatial-wide training, spatial-edge OOD eval.",
    ),
    CaseSpec(
        "visual_axes",
        "reach",
        "camera_viewpoint",
        "reach_camera_multi_pose",
        "results/bc_128px_v1/metrics/ood/metrics__scratch_bc_128__camera_multi_pose__budget200__seed000__eval_camera_extreme_ood_eval.json",
        "Scratch CNN, budget 200, multi-camera training, extreme-camera OOD eval.",
    ),
    CaseSpec(
        "visual_axes",
        "reach",
        "lighting_direction_intensity",
        "reach_lighting_diverse",
        "results/bc_128px_v1/metrics/ood/metrics__scratch_bc_128__lighting_diverse__budget200__seed000__eval_lighting_extreme_ood_eval.json",
        "Scratch CNN, budget 200, lighting-diverse training, extreme-lighting OOD eval.",
    ),
    CaseSpec(
        "visual_axes",
        "avoid_reach",
        "color",
        "avoid_color_multi",
        "results/bc_128px_phase_precise_avoid/metrics/ood/metrics__scratch_bc_128__avoid_color_multi__budget200__seed000__eval_avoid_color_ood_eval.json",
        "Scratch CNN, phase-precise avoid training, budget 200, color-diverse training, color OOD eval.",
    ),
    CaseSpec(
        "visual_axes",
        "avoid_reach",
        "spatial_distribution",
        "avoid_spatial_wide",
        "results/bc_128px_phase_precise_avoid/metrics/ood/metrics__scratch_bc_128__avoid_spatial_wide__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
        "Scratch CNN, phase-precise avoid training, budget 200, spatial-wide training, spatial-edge OOD eval.",
    ),
    CaseSpec(
        "visual_axes",
        "avoid_reach",
        "camera_viewpoint",
        "avoid_camera_multi_pose",
        "results/bc_128px_phase_precise_avoid/metrics/ood/metrics__scratch_bc_128__avoid_camera_multi_pose__budget200__seed000__eval_avoid_camera_extreme_ood_eval.json",
        "Scratch CNN, phase-precise avoid training, budget 200, multi-camera training, extreme-camera OOD eval.",
    ),
    CaseSpec(
        "visual_axes",
        "avoid_reach",
        "lighting_direction_intensity",
        "avoid_lighting_diverse",
        "results/bc_128px_phase_precise_avoid/metrics/ood/metrics__scratch_bc_128__avoid_lighting_diverse__budget200__seed000__eval_avoid_lighting_extreme_ood_eval.json",
        "Scratch CNN, phase-precise avoid training, budget 200, lighting-diverse training, extreme-lighting OOD eval.",
    ),
]


IMPROVEMENT_CASES = [
    CaseSpec(
        "improvements",
        "avoid_reach",
        "spatial_distribution",
        "edge_balanced_visual_only_scratch",
        "results/edge_balanced_scratch_bc_128px_phase_precise_avoid/metrics/ood/metrics__scratch_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
        "Before diagnostic: edge-balanced obstacle-aware visual-only Scratch CNN.",
    ),
    CaseSpec(
        "improvements",
        "avoid_reach",
        "spatial_distribution",
        "phase_geometry_scratch",
        "results/structured_scratch_bc_128px_phase_geo_avoid/metrics/ood/metrics__scratch_structured_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
        "Main improvement: phase+geometry-conditioned visual Scratch CNN.",
    ),
    CaseSpec(
        "improvements",
        "avoid_reach",
        "spatial_distribution",
        "target_geometry_only_ablation",
        "results/ablation_scratch_target_only_bc_128px_phase_geo_avoid/metrics/ood/metrics__scratch_target_only_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
        "Ablation: target geometry only.",
    ),
    CaseSpec(
        "improvements",
        "avoid_reach",
        "spatial_distribution",
        "full_geometry_only_ablation",
        "results/ablation_scratch_geometry_only_bc_128px_phase_geo_avoid/metrics/ood/metrics__scratch_geometry_only_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
        "Ablation: full geometry only, no phase.",
    ),
    CaseSpec(
        "improvements",
        "avoid_reach",
        "spatial_distribution",
        "phase_only_ablation",
        "results/ablation_scratch_phase_only_bc_128px_phase_geo_avoid/metrics/ood/metrics__scratch_phase_only_bc_128__avoid_spatial_edge_balanced__budget200__seed000__eval_avoid_spatial_edge_ood_eval.json",
        "Ablation: phase only.",
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def cm(value_m: float) -> float:
    return 100.0 * float(value_m)


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("@", "at")
        .replace(":", "_")
    )


def pick_quantile_representatives(results: list[dict[str, Any]], per_case: int) -> list[dict[str, Any]]:
    ordered = sorted(results, key=lambda r: (float(r["final_distance"]), float(r["best_distance"])))
    if len(ordered) <= per_case:
        return ordered
    if per_case == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(i * (len(ordered) - 1) / (per_case - 1)) for i in range(per_case)]
    picked: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for idx in indices:
        item = ordered[int(idx)]
        key = (int(item["eval_seed"]), int(item["episode_index"]))
        if key not in seen:
            picked.append(item)
            seen.add(key)
    for item in ordered:
        if len(picked) >= per_case:
            break
        key = (int(item["eval_seed"]), int(item["episode_index"]))
        if key not in seen:
            picked.append(item)
            seen.add(key)
    return picked


def select_episodes(
    metrics: dict[str, Any],
    per_case: int,
    contrast_best_gap_cm: float,
    contrast_final_gap_cm: float,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    results = list(metrics.get("results", []))
    if not results:
        raise ValueError("metrics file has no rollout results")

    best = sorted(
        results,
        key=lambda r: (
            not bool(r.get("success_at_0p5cm", False)),
            not bool(r.get("success_at_1cm", False)),
            float(r["best_distance"]),
            float(r["final_distance"]),
        ),
    )[:per_case]
    worst = sorted(
        results,
        key=lambda r: (
            bool(r.get("success_at_5cm", False)),
            bool(r.get("success_at_2cm", False)),
            bool(r.get("success_at_1cm", False)),
            -float(r["final_distance"]),
            -float(r["best_distance"]),
        ),
    )[:per_case]

    best_gap_cm = 100.0 * (min(float(r["best_distance"]) for r in worst) - max(float(r["best_distance"]) for r in best))
    final_gap_cm = 100.0 * (min(float(r["final_distance"]) for r in worst) - max(float(r["final_distance"]) for r in best))
    success_contrast = (
        any(bool(r.get("success_at_1cm", False)) for r in best)
        and any(not bool(r.get("success_at_1cm", False)) for r in worst)
    ) or (
        any(bool(r.get("success_at_5cm", False)) for r in best)
        and any(not bool(r.get("success_at_5cm", False)) for r in worst)
    )

    if best_gap_cm < contrast_best_gap_cm and final_gap_cm < contrast_final_gap_cm and not success_contrast:
        representatives = pick_quantile_representatives(results, per_case=per_case)
        reason = (
            "No meaningful best/worst contrast in this metric file; "
            f"best-distance gap={best_gap_cm:.2f}cm, final-distance gap={final_gap_cm:.2f}cm."
        )
        return {"representative": representatives}, reason

    reason = f"Contrast selected; best-distance gap={best_gap_cm:.2f}cm, final-distance gap={final_gap_cm:.2f}cm."
    return {"best": best, "worst": worst}, reason


def build_jobs(
    cases: list[CaseSpec],
    per_case: int,
    output_root: Path,
    contrast_best_gap_cm: float,
    contrast_final_gap_cm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for case in cases:
        metrics_file = ROOT / case.metrics_path
        if not metrics_file.exists():
            raise FileNotFoundError(metrics_file)
        metrics = load_json(metrics_file)
        selections, selection_note = select_episodes(
            metrics,
            per_case=per_case,
            contrast_best_gap_cm=contrast_best_gap_cm,
            contrast_final_gap_cm=contrast_final_gap_cm,
        )
        model_path = str(metrics["model_path"])
        model_family = str(metrics["model_family"])
        for case_type, episodes in selections.items():
            for rank, episode in enumerate(episodes, start=1):
                eval_dataset = str(episode["eval_dataset_path"])
                eval_seed = int(episode["eval_seed"])
                episode_index = int(episode["episode_index"])
                best_cm = cm(float(episode["best_distance"]))
                final_cm = cm(float(episode["final_distance"]))
                success_1cm = bool(episode.get("success_at_1cm", False))
                success_5cm = bool(episode.get("success_at_5cm", False))
                out_dir = output_root / case.group / case.task / case.axis / case.label / case_type
                out_name = (
                    f"{rank:02d}__seed{eval_seed:03d}__ep{episode_index:03d}"
                    f"__best{best_cm:.1f}cm__final{final_cm:.1f}cm.mp4"
                )
                out_path = out_dir / (safe_name(Path(out_name).stem) + ".mp4")
                row = {
                    "group": case.group,
                    "task": case.task,
                    "axis": case.axis,
                    "case_label": case.label,
                    "case_type": case_type,
                    "rank": rank,
                    "note": case.note,
                    "selection_note": selection_note,
                    "model_family": model_family,
                    "model_path": model_path,
                    "metrics_path": case.metrics_path,
                    "eval_dataset_path": eval_dataset,
                    "eval_seed": eval_seed,
                    "episode_index": episode_index,
                    "best_distance_cm": f"{best_cm:.3f}",
                    "final_distance_cm": f"{final_cm:.3f}",
                    "success_at_1cm": success_1cm,
                    "success_at_5cm": success_5cm,
                    "video_path": str(out_path.relative_to(ROOT)),
                    "summary_path": str(out_path.with_suffix(".json").relative_to(ROOT)),
                }
                manifest_rows.append(row)
                jobs.append(
                    {
                        "model_path": model_path,
                        "model_family": model_family,
                        "eval_dataset": eval_dataset,
                        "episode_index": episode_index,
                        "output": str(out_path),
                        "manifest": row,
                    }
                )
    return jobs, manifest_rows


def run_job(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output = ROOT / job["output"]
    if output.exists() and output.with_suffix(".json").exists() and not args.overwrite:
        return {"status": "skipped", **job["manifest"]}

    cmd = [
        args.python_bin,
        "code/record_bc_rollout.py",
        "--model-path",
        job["model_path"],
        "--model-family",
        job["model_family"],
        "--eval-dataset",
        job["eval_dataset"],
        "--episode-index",
        str(job["episode_index"]),
        "--output",
        str(output),
        "--device",
        args.device,
        "--width",
        str(args.model_width),
        "--height",
        str(args.model_height),
        "--video-width",
        str(args.video_width),
        "--video-height",
        str(args.video_height),
        "--video-fov",
        str(args.video_fov),
        "--video-external-distance-scale",
        str(args.video_external_distance_scale),
        "--view",
        args.view,
        "--video-stride",
        str(args.video_stride),
        "--steps-per-episode",
        "400",
        "--fps",
        str(args.fps),
        "--encoder",
        args.encoder,
        "--crf",
        str(args.crf),
        "--stop-threshold",
        "0.001",
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("TORCH_NUM_THREADS", "2")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        return {
            "status": "failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-4000:],
            **job["manifest"],
        }
    return {"status": "ok", **job["manifest"]}


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_html_index(rows: list[dict[str, Any]], output_root: Path) -> None:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["group"], row["task"], row["axis"], row["case_label"], row["case_type"])
        groups.setdefault(key, []).append(row)

    sections = []
    for key in sorted(groups):
        group, task, axis, case_label, case_type = key
        videos = []
        for row in sorted(groups[key], key=lambda r: int(r["rank"])):
            rel = Path(row["video_path"]).relative_to(output_root.relative_to(ROOT))
            videos.append(
                f"""
<figure>
  <video controls preload="metadata" src="{rel.as_posix()}"></video>
  <figcaption>
    rank {row['rank']} | seed {row['eval_seed']} ep {row['episode_index']} |
    best {row['best_distance_cm']} cm | final {row['final_distance_cm']} cm |
    S@1cm {row['success_at_1cm']} | S@5cm {row['success_at_5cm']}
  </figcaption>
</figure>"""
            )
        sections.append(
            f"""
<section>
  <h2>{group} / {task} / {axis} / {case_label} / {case_type}</h2>
  <p>{groups[key][0]['note']}</p>
  <div class="grid">{''.join(videos)}</div>
</section>"""
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Case Studies</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; margin: 24px; color: #18202b; background: #f7f8fa; }}
    a {{ color: #156b75; }}
    h1 {{ font-size: 40px; margin-bottom: 6px; }}
    h2 {{ margin-top: 34px; border-top: 1px solid #d9dee7; padding-top: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 10px; }}
    video {{ width: 100%; border-radius: 6px; background: black; }}
    figcaption {{ font-size: 13px; color: #5e6877; margin-top: 8px; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 5px; }}
  </style>
</head>
<body>
  <h1>Video Case Studies</h1>
  <p>Best and worst closed-loop rollout videos selected from existing precision eval metrics. Manifest: <a href="manifest.csv">manifest.csv</a>.</p>
  {''.join(sections)}
</body>
</html>
"""
    (output_root / "index.html").write_text(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-case", type=int, default=3, help="Number of best and worst videos per case.")
    parser.add_argument("--contrast-best-gap-cm", type=float, default=0.25)
    parser.add_argument("--contrast-final-gap-cm", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--output-root", default="results/video_case_studies_1080p")
    parser.add_argument("--model-width", type=int, default=128)
    parser.add_argument("--model-height", type=int, default=128)
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--video-fov", type=float, default=85.0)
    parser.add_argument("--video-external-distance-scale", type=float, default=1.35)
    parser.add_argument("--view", choices=["external", "eef", "external_with_eef_inset", "side_by_side"], default="external")
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--encoder", choices=["auto", "ffmpeg", "opencv"], default="auto")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--only-plan", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root
    cases = VISUAL_CASES + IMPROVEMENT_CASES
    jobs, manifest_rows = build_jobs(
        cases=cases,
        per_case=args.per_case,
        output_root=output_root,
        contrast_best_gap_cm=float(args.contrast_best_gap_cm),
        contrast_final_gap_cm=float(args.contrast_final_gap_cm),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_rows, output_root / "manifest.csv")
    (output_root / "manifest.json").write_text(json.dumps(manifest_rows, indent=2))
    print(f"planned_jobs={len(jobs)} output_root={output_root.relative_to(ROOT)}")
    if args.only_plan:
        write_html_index(manifest_rows, output_root)
        return

    completed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_to_job = {
            executor.submit(run_job, job, args): job for job in jobs
        }
        for idx, future in enumerate(as_completed(future_to_job), start=1):
            result = future.result()
            status = result.pop("status")
            result["record_status"] = status
            completed_rows.append(result)
            print(
                f"[{idx:03d}/{len(jobs):03d}] {status}: "
                f"{result['group']}/{result['task']}/{result['axis']}/{result['case_label']}/{result['case_type']}/"
                f"{result['rank']} best={result['best_distance_cm']}cm final={result['final_distance_cm']}cm",
                flush=True,
            )
            if status == "failed":
                failures.append(result)

    write_manifest(completed_rows, output_root / "recording_manifest.csv")
    (output_root / "recording_manifest.json").write_text(json.dumps(completed_rows, indent=2))
    write_html_index(completed_rows, output_root)
    if failures:
        (output_root / "failures.json").write_text(json.dumps(failures, indent=2))
        raise SystemExit(f"{len(failures)} recording jobs failed; see {output_root / 'failures.json'}")
    print(f"completed_jobs={len(completed_rows)}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
