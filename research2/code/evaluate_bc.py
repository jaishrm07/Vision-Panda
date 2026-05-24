#!/usr/bin/env python3
"""Closed-loop evaluation for research2 BC policies on saved scene banks."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data
import torch
from tqdm import tqdm

from bc_model import build_policy, extract_action_prediction, load_checkpoint_payload, resolve_model_family
from collect_dataset import ACTION_MAGNITUDE, POSITION_GAIN, SUCCESS_THRESHOLDS, clip_action, expert_action, task_distance
from preview_simulator import (
    DEFAULT_JOINT_START_POSITIONS,
    PANDA_EE_LINK_INDEX,
    CameraConfig,
    LightingConfig,
    RenderConfig,
    SceneConfig,
    add_scene_objects,
    compute_eef_view,
    compute_external_view,
    render_rgb,
)
from train_bc import dataset_stem, select_device, write_json
from structured_features import phase_label_to_index, structured_state_array

DEFAULT_SUCCESS_THRESHOLDS = (0.005, 0.01, 0.02, 0.05)
DEFAULT_STOP_THRESHOLD = 0.001


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_torch_threads() -> None:
    requested = os.environ.get("TORCH_NUM_THREADS")
    if requested:
        torch.set_num_threads(max(1, int(requested)))


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def threshold_label(threshold_meters: float) -> str:
    centimeters = threshold_meters * 100.0
    return f"{centimeters:g}".replace(".", "p") + "cm"


def normalize_success_thresholds(thresholds: list[float] | None) -> list[tuple[str, float]]:
    resolved = DEFAULT_SUCCESS_THRESHOLDS if thresholds is None else tuple(float(value) for value in thresholds)
    if not resolved:
        raise ValueError("At least one success threshold is required.")
    return [(threshold_label(float(value)), float(value)) for value in resolved]


def update_threshold_crossings(
    distance: float,
    step: int,
    threshold_specs: list[tuple[str, float]],
    success_flags: dict[str, bool],
    first_steps: dict[str, int | None],
) -> None:
    for label, threshold in threshold_specs:
        if not success_flags[label] and distance <= threshold:
            success_flags[label] = True
            first_steps[label] = int(step)


def deserialize_lighting(payload: dict[str, Any] | None) -> LightingConfig | None:
    if payload is None or payload.get("mode") == "pybullet_default":
        return None
    return LightingConfig(
        light_direction=tuple(float(v) for v in payload["light_direction"]),
        ambient=float(payload.get("ambient", payload.get("light_ambient_coeff"))),
        diffuse=float(payload.get("diffuse", payload.get("light_diffuse_coeff"))),
        specular=float(payload.get("specular", payload.get("light_specular_coeff"))),
        light_color=tuple(float(v) for v in payload.get("light_color", (1.0, 1.0, 1.0))),
    )


def deserialize_scene(payload: dict[str, Any]) -> SceneConfig:
    camera_payload = payload["camera"]
    camera = CameraConfig(
        yaw=float(camera_payload["yaw"]),
        pitch=float(camera_payload["pitch"]),
        distance=float(camera_payload["distance"]),
        target_position=tuple(float(v) for v in camera_payload["target_position"]),
    )
    return SceneConfig(
        train_config=str(payload["train_config"]),
        task=str(payload["task"]),
        axis=str(payload["axis"]),
        variant=str(payload["variant"]),
        target_position=tuple(float(v) for v in payload["target_position"]),
        target_color_name=str(payload["target_color_name"]),
        target_rgba=tuple(float(v) for v in payload["target_rgba"]),
        camera=camera,
        lighting=deserialize_lighting(payload.get("lighting")),
        obstacle=payload.get("obstacle"),
    )


class ClosedLoopEvalEnv:
    def __init__(self, render_cfg: RenderConfig):
        self.render_cfg = render_cfg
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.panda: int | None = None
        self.cube: int | None = None

    def close(self) -> None:
        p.disconnect(physicsClientId=self.physics_client)

    def reset_world(self) -> None:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.625])
        p.loadURDF("table/table.urdf", basePosition=[0.5, 0.0, -0.625], useFixedBase=True)
        self.panda = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=True,
        )
        for joint_index, joint_position in enumerate(DEFAULT_JOINT_START_POSITIONS):
            p.resetJointState(self.panda, joint_index, joint_position)

    def reset_episode(self, scene: SceneConfig) -> None:
        self.reset_world()
        add_scene_objects(scene)
        cube_candidates = []
        for body_id in range(p.getNumBodies()):
            info = p.getBodyInfo(body_id)
            name = info[1].decode("utf-8") if info and info[1] else ""
            if "cube" in name.lower():
                cube_candidates.append(body_id)
        if not cube_candidates:
            raise RuntimeError("Could not locate target cube after scene reset.")
        self.cube = cube_candidates[-1]
        for _ in range(20):
            p.stepSimulation()

    def cube_position(self) -> np.ndarray:
        if self.cube is None:
            raise RuntimeError("Cube is not initialized.")
        position, _ = p.getBasePositionAndOrientation(self.cube)
        return np.asarray(position, dtype=np.float32)

    def ee_position(self) -> np.ndarray:
        if self.panda is None:
            raise RuntimeError("Panda is not initialized.")
        ee_values = p.getLinkState(self.panda, PANDA_EE_LINK_INDEX, computeForwardKinematics=True)
        return np.asarray(ee_values[4], dtype=np.float32)

    def robot_state(self, scene: SceneConfig) -> dict[str, Any]:
        if self.panda is None:
            raise RuntimeError("Panda is not initialized.")
        external_rgb = render_rgb(compute_external_view(scene.camera), self.render_cfg, scene.lighting)
        eef_rgb = render_rgb(compute_eef_view(self.panda), self.render_cfg, scene.lighting)
        return {
            "external_rgb": external_rgb,
            "eef_rgb": eef_rgb,
            "ee_position": self.ee_position().tolist(),
        }

    def step_to_pose(self, ee_position: np.ndarray, position_gain: float = POSITION_GAIN) -> None:
        if self.panda is None:
            raise RuntimeError("Panda is not initialized.")
        ee_quaternion = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
        target_positions = p.calculateInverseKinematics(self.panda, PANDA_EE_LINK_INDEX, list(ee_position), ee_quaternion)
        p.setJointMotorControlArray(
            self.panda,
            range(9),
            p.POSITION_CONTROL,
            targetPositions=target_positions[:9],
            positionGains=[position_gain] * 9,
        )
        p.stepSimulation()


def tensor_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).to(device)


def load_eval_payload(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "episodes" not in payload:
        raise ValueError(f"Expected research2 eval dataset payload with episodes: {path}")
    return payload


def summarize_results(results: list[dict[str, Any]], threshold_specs: list[tuple[str, float]]) -> dict[str, Any]:
    primary_label = threshold_specs[-1][0]
    successes = [bool(item.get(f"success_at_{primary_label}", item["success"])) for item in results]
    success_steps = [
        float(item[f"first_step_at_{primary_label}"])
        for item in results
        if item.get(f"success_at_{primary_label}", item["success"]) and item.get(f"first_step_at_{primary_label}") is not None
    ]
    summary = {
        "rollout_count": len(results),
        "success_count": int(sum(successes)),
        "success_rate": float(sum(successes) / len(results)) if results else 0.0,
        "primary_success_label": primary_label,
        "mean_best_distance": mean_or_none([float(item["best_distance"]) for item in results]),
        "mean_final_distance": mean_or_none([float(item["final_distance"]) for item in results]),
        "mean_steps": mean_or_none([float(item["steps"]) for item in results]),
        "mean_success_steps": mean_or_none(success_steps),
    }
    for label, _ in threshold_specs:
        threshold_successes = [bool(item.get(f"success_at_{label}", False)) for item in results]
        threshold_steps = [
            float(item[f"first_step_at_{label}"])
            for item in results
            if item.get(f"success_at_{label}", False) and item.get(f"first_step_at_{label}") is not None
        ]
        summary[f"success_count_at_{label}"] = int(sum(threshold_successes))
        summary[f"success_rate_at_{label}"] = float(sum(threshold_successes) / len(results)) if results else 0.0
        summary[f"mean_first_step_at_{label}"] = mean_or_none(threshold_steps)
    return summary


def evaluate_policy(
    model_path: Path,
    eval_dataset_paths: list[Path],
    metrics_path: Path | None = None,
    model_family: str | None = None,
    device_name: str = "auto",
    steps_per_episode: int = 400,
    max_episodes_per_dataset: int | None = None,
    success_threshold: float | None = None,
    success_thresholds: list[float] | None = None,
    stop_threshold: float = DEFAULT_STOP_THRESHOLD,
    position_gain: float = POSITION_GAIN,
) -> dict[str, Any]:
    configure_torch_threads()
    if success_thresholds is None and success_threshold is not None:
        success_thresholds = [float(success_threshold)]
    threshold_specs = normalize_success_thresholds(success_thresholds)
    primary_label, primary_threshold = threshold_specs[-1]
    stop_threshold = float(stop_threshold)

    device = select_device(device_name)
    checkpoint_payload = load_checkpoint_payload(model_path, map_location=device)
    resolved_model_family = resolve_model_family(model_family, checkpoint_payload)
    model = build_policy(resolved_model_family).to(device)
    model.load_state_dict(checkpoint_payload["state_dict"])
    model.eval()
    checkpoint_metadata = checkpoint_payload.get("metadata", {})

    render_cfg = RenderConfig(width=128, height=128)
    env = ClosedLoopEvalEnv(render_cfg)
    results: list[dict[str, Any]] = []
    eval_dataset_metadata: list[dict[str, Any]] = []
    try:
        for eval_dataset_path in eval_dataset_paths:
            payload = load_eval_payload(eval_dataset_path)
            episodes = list(payload["episodes"])
            if max_episodes_per_dataset is not None:
                episodes = episodes[: int(max_episodes_per_dataset)]
            eval_dataset_metadata.append(
                {
                    "eval_dataset_path": str(eval_dataset_path),
                    "eval_config": payload.get("train_config"),
                    "task": payload.get("task"),
                    "axis": payload.get("axis"),
                    "variant": payload.get("variant"),
                    "budget": payload.get("budget"),
                    "seed": payload.get("seed"),
                    "episodes_used": len(episodes),
                }
            )
            for episode in tqdm(episodes, desc=f"eval {payload.get('train_config')} seed{payload.get('seed')}", unit="episode"):
                scene_payload = episode["initial_scene"]
                scene = deserialize_scene(scene_payload)
                env.reset_episode(scene)
                success_flags = {label: False for label, _ in threshold_specs}
                first_steps = {label: None for label, _ in threshold_specs}
                initial_ee_position = env.ee_position()
                initial_cube_position = env.cube_position()
                initial_distance = task_distance(scene.task, initial_ee_position, initial_cube_position)
                best_distance = float(initial_distance)
                final_distance = float(initial_distance)
                steps_taken = 0
                stopped_early = False
                stop_reason = "horizon"
                update_threshold_crossings(initial_distance, 0, threshold_specs, success_flags, first_steps)

                if initial_distance <= stop_threshold:
                    stopped_early = True
                    stop_reason = "stop_threshold"

                controller_state: dict[str, Any] = {}
                for step_idx in range(int(steps_per_episode)):
                    if stopped_early:
                        break
                    robot_state = env.robot_state(scene)
                    ee_position = np.asarray(robot_state["ee_position"], dtype=np.float32)
                    cube_position = env.cube_position()

                    external = tensor_image(robot_state["external_rgb"], device)
                    eef = tensor_image(robot_state["eef_rgb"], device)
                    state = torch.tensor(ee_position, dtype=torch.float32, device=device).unsqueeze(0)
                    structured_state = None
                    if bool(getattr(model, "uses_structured_state", False)):
                        phase_index = -100
                        if scene.task == "avoid_reach":
                            expert_action(scene.task, ee_position, cube_position, scene.obstacle, controller_state)
                            phase_index = phase_label_to_index(controller_state.get("avoid_phase"))
                        structured_state = torch.tensor(
                            structured_state_array(
                                ee_position=ee_position,
                                target_position=cube_position,
                                obstacle=scene.obstacle,
                                phase_index=phase_index,
                            ),
                            dtype=torch.float32,
                            device=device,
                        ).unsqueeze(0)
                    with torch.no_grad():
                        if structured_state is not None:
                            model_output = model(external, eef, state, structured_state)
                        else:
                            model_output = model(external, eef, state)
                        action = extract_action_prediction(model_output).detach().cpu().numpy().squeeze()
                    action = clip_action(action, magnitude=ACTION_MAGNITUDE)
                    env.step_to_pose(ee_position + action, position_gain=position_gain)
                    steps_taken = step_idx + 1

                    stepped_ee_position = env.ee_position()
                    cube_position = env.cube_position()
                    distance = task_distance(scene.task, stepped_ee_position, cube_position)
                    best_distance = min(best_distance, distance)
                    final_distance = distance
                    update_threshold_crossings(distance, steps_taken, threshold_specs, success_flags, first_steps)
                    if distance <= stop_threshold:
                        stopped_early = True
                        stop_reason = "stop_threshold"
                        break

                success = bool(success_flags[primary_label])
                result = {
                    f"success_at_{label}": bool(success_flags[label])
                    for label, _ in threshold_specs
                }
                result.update(
                    {
                        f"first_step_at_{label}": first_steps[label]
                        for label, _ in threshold_specs
                    }
                )
                result.update(
                    {
                        "success_thresholds": {label: threshold for label, threshold in threshold_specs},
                        "primary_success_label": primary_label,
                        "success_threshold": primary_threshold,
                        "stop_threshold": stop_threshold,
                        "stopped_early": bool(stopped_early),
                        "stop_reason": stop_reason,
                    }
                )
                results.append(
                    {
                        "eval_dataset_path": str(eval_dataset_path),
                        "eval_config": payload.get("train_config"),
                        "eval_seed": payload.get("seed"),
                        "episode_index": episode.get("episode_index"),
                        "task": scene.task,
                        "axis": scene.axis,
                        "variant": scene.variant,
                        "success": bool(success),
                        "best_distance": float(best_distance),
                        "final_distance": float(final_distance),
                        "steps": int(steps_taken),
                        **result,
                    }
                )
    finally:
        env.close()

    summary = {
        "created_at": utc_now(),
        "model_path": str(model_path),
        "model_family": resolved_model_family,
        "checkpoint_metadata": checkpoint_metadata,
        "device": str(device),
        "steps_per_episode": int(steps_per_episode),
        "max_episodes_per_dataset": max_episodes_per_dataset,
        "success_thresholds": {label: threshold for label, threshold in threshold_specs},
        "primary_success_label": primary_label,
        "stop_threshold": stop_threshold,
        "position_gain": float(position_gain),
        "eval_datasets": eval_dataset_metadata,
        "metrics": summarize_results(results, threshold_specs),
        "results": results,
    }
    if metrics_path is not None:
        write_json(Path(metrics_path), summary)
    return summary


def resolve_eval_dataset_paths(args: argparse.Namespace) -> list[Path]:
    if args.eval_datasets:
        return [Path(path) for path in args.eval_datasets]
    if args.eval_dir is None or args.eval_config is None:
        raise ValueError("Pass --eval-datasets or pass --eval-dir and --eval-config.")
    seeds = args.eval_seeds if args.eval_seeds else [args.eval_seed]
    return [Path(args.eval_dir) / f"{dataset_stem(args.eval_config, args.eval_budget, seed)}.pkl" for seed in seeds]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-family")
    parser.add_argument("--eval-datasets", type=Path, nargs="+")
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--eval-config")
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=200)
    parser.add_argument("--eval-seeds", type=int, nargs="+")
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps-per-episode", type=int, default=400)
    parser.add_argument("--max-episodes-per-dataset", type=int)
    parser.add_argument("--success-threshold", type=float)
    parser.add_argument("--success-thresholds", type=float, nargs="+")
    parser.add_argument("--stop-threshold", type=float, default=DEFAULT_STOP_THRESHOLD)
    parser.add_argument("--position-gain", type=float, default=POSITION_GAIN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_policy(
        model_path=args.model_path,
        eval_dataset_paths=resolve_eval_dataset_paths(args),
        metrics_path=args.metrics_path,
        model_family=args.model_family,
        device_name=args.device,
        steps_per_episode=args.steps_per_episode,
        max_episodes_per_dataset=args.max_episodes_per_dataset,
        success_threshold=args.success_threshold,
        success_thresholds=args.success_thresholds,
        stop_threshold=args.stop_threshold,
        position_gain=args.position_gain,
    )
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
