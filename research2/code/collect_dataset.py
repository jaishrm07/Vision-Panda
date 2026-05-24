#!/usr/bin/env python3
"""Collect 128x128 behavior-cloning datasets for the visual-diversity matrix."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data
from tqdm import tqdm

from preview_simulator import (
    DEFAULT_JOINT_START_POSITIONS,
    PANDA_EE_LINK_INDEX,
    RenderConfig,
    add_scene_objects,
    build_scene,
    compute_eef_view,
    compute_external_view,
    load_yaml,
    render_rgb,
    serialize_scene,
)


SUCCESS_THRESHOLDS = {
    "reach": 0.05,
    "avoid_reach": 0.05,
}
ACTION_MAGNITUDE = 1.0
POSITION_GAIN = 0.03
AVOID_HOVER_HEIGHT = 0.16
AVOID_XY_TOLERANCE = 0.025
AVOID_Z_TOLERANCE = 0.02
AVOID_WALL_CLEARANCE = 0.07
AVOID_POST_WALL_OFFSET = 0.04
AVOID_CUBE_SIDE_OFFSET = 0.03
AVOID_TARGET_HEIGHT_OFFSET = 0.12
AVOID_PHASE_NAMES = ("side_align", "cube_hover", "final_descent")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_budget_tag(budget: int) -> str:
    return f"budget{int(budget):03d}"


def format_seed_tag(seed: int) -> str:
    return f"seed{int(seed):03d}"


def dataset_stem(train_config_name: str, budget: int, seed: int) -> str:
    return "__".join(["dataset", train_config_name, format_budget_tag(budget), format_seed_tag(seed)])


def clip_action(action: np.ndarray, magnitude: float = ACTION_MAGNITUDE) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    norm = float(np.linalg.norm(action))
    if norm > magnitude and norm > 0.0:
        action = action * (magnitude / norm)
    return action


def avoid_target_position(cube_position: np.ndarray) -> np.ndarray:
    target = np.asarray(cube_position, dtype=np.float32).copy()
    target[2] += AVOID_TARGET_HEIGHT_OFFSET
    return target


def task_distance(task_name: str, ee_position: np.ndarray, cube_position: np.ndarray) -> float:
    if task_name == "reach":
        target = cube_position
    elif task_name == "avoid_reach":
        target = avoid_target_position(cube_position)
    else:
        raise ValueError(f"Unsupported task: {task_name}")
    return float(np.linalg.norm(np.asarray(ee_position, dtype=np.float32) - target))


def expert_action(
    task_name: str,
    ee_position: np.ndarray,
    cube_position: np.ndarray,
    obstacle: dict[str, Any] | None,
    controller_state: dict[str, Any],
) -> np.ndarray:
    if task_name == "reach" or obstacle is None:
        return clip_action(cube_position - ee_position)
    if task_name == "avoid_reach":
        return avoid_reach_action(ee_position, cube_position, obstacle, controller_state)
    raise ValueError(f"Unsupported task: {task_name}")


def avoid_reach_action(
    ee_position: np.ndarray,
    cube_position: np.ndarray,
    obstacle: dict[str, Any],
    controller_state: dict[str, Any],
) -> np.ndarray:
    obstacle_center = np.asarray(obstacle["center"], dtype=np.float32)
    obstacle_half_extents = np.asarray(obstacle["half_extents"], dtype=np.float32)

    hover_target = np.array([ee_position[0], ee_position[1], AVOID_HOVER_HEIGHT], dtype=np.float32)
    if ee_position[2] < AVOID_HOVER_HEIGHT - AVOID_Z_TOLERANCE:
        controller_state["avoid_stage"] = -1
        controller_state["avoid_phase"] = "initial_hover"
        controller_state["avoid_waypoint"] = hover_target.astype(float).tolist()
        controller_state["avoid_route_sign"] = None
        controller_state["avoid_route_y"] = None
        return clip_action(hover_target - ee_position)

    route_sign = 1.0 if cube_position[1] >= obstacle_center[1] else -1.0
    if abs(float(cube_position[1] - obstacle_center[1])) < 0.035:
        route_sign = 1.0
    route_y = obstacle_center[1] + route_sign * (obstacle_half_extents[1] + AVOID_WALL_CLEARANCE)

    obstacle_right_x = obstacle_center[0] + obstacle_half_extents[0]
    side_align = np.array(
        [
            max(cube_position[0] + AVOID_CUBE_SIDE_OFFSET, obstacle_right_x + AVOID_POST_WALL_OFFSET),
            route_y,
            AVOID_HOVER_HEIGHT,
        ],
        dtype=np.float32,
    )
    cube_hover = np.array([cube_position[0], cube_position[1], AVOID_HOVER_HEIGHT], dtype=np.float32)
    final_target = avoid_target_position(cube_position)

    waypoints = (side_align, cube_hover, final_target)
    stage = int(controller_state.get("avoid_stage", 0))
    stage = max(0, min(stage, len(waypoints) - 1))
    while stage < len(waypoints) - 1:
        waypoint = waypoints[stage]
        xy_error = float(np.linalg.norm(ee_position[:2] - waypoint[:2]))
        z_error = abs(float(ee_position[2] - waypoint[2]))
        if xy_error > AVOID_XY_TOLERANCE or z_error > AVOID_Z_TOLERANCE:
            break
        stage += 1
    controller_state["avoid_stage"] = stage
    controller_state["avoid_phase"] = AVOID_PHASE_NAMES[stage]
    controller_state["avoid_waypoint"] = waypoints[stage].astype(float).tolist()
    controller_state["avoid_route_sign"] = float(route_sign)
    controller_state["avoid_route_y"] = float(route_y)

    return clip_action(waypoints[stage] - ee_position)


class DatasetEnv:
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

    def reset_episode(self, scene) -> None:
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

    def robot_state(self, scene) -> dict[str, Any]:
        if self.panda is None:
            raise RuntimeError("Panda is not initialized.")
        joint_values = p.getJointStates(self.panda, range(11))
        ee_values = p.getLinkState(self.panda, PANDA_EE_LINK_INDEX, computeForwardKinematics=True)
        external_rgb = render_rgb(compute_external_view(scene.camera), self.render_cfg, scene.lighting)
        eef_rgb = render_rgb(compute_eef_view(self.panda), self.render_cfg, scene.lighting)
        return {
            "external_rgb": external_rgb,
            "eef_rgb": eef_rgb,
            "robot_state": {
                "ee_position": list(ee_values[4]),
                "ee_quaternion": list(ee_values[5]),
                "joint_positions": [float(item[0]) for item in joint_values],
                "joint_velocities": [float(item[1]) for item in joint_values],
                "joint_torques": [float(item[3]) for item in joint_values],
            },
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


def train_config_by_name(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    by_name = {item["name"]: item for item in config["train_configs"]}
    for item in config.get("eval_configs", []):
        if item["name"] in by_name:
            raise ValueError(f"Duplicate config name: {item['name']}")
        by_name[item["name"]] = item
    return by_name


def selected_train_configs(config: dict[str, Any], names: list[str] | None) -> list[dict[str, str]]:
    by_name = train_config_by_name(config)
    if names is None:
        return list(config["train_configs"])
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown train configs: {unknown}")
    return [by_name[name] for name in names]


def quality_checks(samples: list[dict[str, Any]], height: int, width: int) -> dict[str, bool]:
    if not samples:
        return {
            "external_rgb_shape_ok": False,
            "eef_rgb_shape_ok": False,
            "external_rgb_nonblank": False,
            "eef_rgb_nonblank": False,
            "metadata_complete": False,
            "sample_count_nonzero": False,
        }
    external_shapes = [sample["external_rgb"].shape for sample in samples]
    eef_shapes = [sample["eef_rgb"].shape for sample in samples]
    return {
        "external_rgb_shape_ok": all(shape == (height, width, 3) for shape in external_shapes),
        "eef_rgb_shape_ok": all(shape == (height, width, 3) for shape in eef_shapes),
        "external_rgb_nonblank": all(float(sample["external_rgb"].std()) > 1.0 for sample in samples),
        "eef_rgb_nonblank": all(float(sample["eef_rgb"].std()) > 1.0 for sample in samples),
        "metadata_complete": True,
        "sample_count_nonzero": len(samples) > 0,
    }


def collect_one_dataset(
    config: dict[str, Any],
    train_config: dict[str, str],
    budget: int,
    seed: int,
    max_steps_per_demo: int,
    output_dir: Path,
    success_threshold: float | None = None,
    episode_start: int = 0,
    episode_end: int | None = None,
    output_suffix: str = "",
) -> dict[str, Any]:
    height = int(config["collection"]["resolution"]["height"])
    width = int(config["collection"]["resolution"]["width"])
    render_cfg = RenderConfig(width=width, height=height)
    episode_start = int(episode_start)
    episode_end = int(episode_end) if episode_end is not None else int(budget)
    if episode_start < 0 or episode_end <= episode_start or episode_end > int(budget):
        raise ValueError(f"Invalid episode range [{episode_start}, {episode_end}) for budget {budget}.")
    stem = dataset_stem(train_config["name"], budget, seed) + str(output_suffix)
    sample_path = output_dir / f"{stem}.pkl"
    metadata_path = output_dir / f"{stem}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    active_success_threshold = float(success_threshold) if success_threshold is not None else float(SUCCESS_THRESHOLDS[train_config["task"]])

    samples: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    env = DatasetEnv(render_cfg)
    try:
        episode_range = range(episode_start, episode_end)
        for episode_index in tqdm(episode_range, desc=f"collect {train_config['name']} {format_budget_tag(budget)} {format_seed_tag(seed)}", unit="demo"):
            scene = build_scene(config, train_config, seed=seed, sample_index=episode_index)
            env.reset_episode(scene)
            scene_payload = serialize_scene(scene)
            final_distance = None
            termination_reason = "step_cap"
            steps_collected = 0
            controller_state: dict[str, Any] = {}

            for step_index in range(int(max_steps_per_demo)):
                state_payload = env.robot_state(scene)
                ee_position = np.asarray(state_payload["robot_state"]["ee_position"], dtype=np.float32)
                cube_position = env.cube_position()
                distance = task_distance(train_config["task"], ee_position, cube_position)
                success = distance <= active_success_threshold
                if step_index > 0 and success:
                    final_distance = distance
                    termination_reason = "success"
                    break

                action = expert_action(train_config["task"], ee_position, cube_position, scene.obstacle, controller_state)
                expert_action_payload = {
                    "type": "delta_position",
                    "delta_position": action.astype(float).tolist(),
                }
                sample_payload = {
                    "episode_index": episode_index,
                    "step_index": step_index,
                    "external_rgb": state_payload["external_rgb"],
                    "eef_rgb": state_payload["eef_rgb"],
                    "robot_state": state_payload["robot_state"],
                    "expert_action": expert_action_payload,
                    "task_distance": distance,
                    "success": bool(success),
                    "scene": scene_payload,
                }
                if train_config["task"] == "avoid_reach":
                    phase_payload = {
                        "avoid_stage": int(controller_state.get("avoid_stage", -1)),
                        "avoid_phase": str(controller_state.get("avoid_phase", "unknown")),
                        "avoid_waypoint": controller_state.get("avoid_waypoint"),
                        "avoid_route_sign": controller_state.get("avoid_route_sign"),
                        "avoid_route_y": controller_state.get("avoid_route_y"),
                    }
                    sample_payload.update(phase_payload)
                    expert_action_payload.update(phase_payload)
                samples.append(sample_payload)
                env.step_to_pose(ee_position + action)
                steps_collected += 1
                final_distance = distance

            if final_distance is None:
                state_payload = env.robot_state(scene)
                final_distance = task_distance(
                    train_config["task"],
                    np.asarray(state_payload["robot_state"]["ee_position"], dtype=np.float32),
                    env.cube_position(),
                )
                if final_distance <= active_success_threshold:
                    termination_reason = "success"

            episodes.append(
                {
                    "episode_index": episode_index,
                    "num_steps": steps_collected,
                    "success": termination_reason == "success",
                    "final_task_distance": final_distance,
                    "initial_scene": scene_payload,
                    "final_scene": scene_payload,
                    "termination_reason": termination_reason,
                }
            )
    finally:
        env.close()

    dataset_payload = {
        "version": config["version"],
        "train_config": train_config["name"],
        "task": train_config["task"],
        "axis": train_config["axis"],
        "variant": train_config["variant"],
        "budget": int(budget),
        "seed": int(seed),
        "episode_start": episode_start,
        "episode_end": episode_end,
        "output_suffix": str(output_suffix),
        "resolution": {"height": height, "width": width},
        "samples": samples,
        "episodes": episodes,
    }
    with sample_path.open("wb") as handle:
        pickle.dump(dataset_payload, handle)

    checks = quality_checks(samples, height=height, width=width)
    success_count = int(sum(1 for item in episodes if item["success"]))
    metadata = {
        "version": config["version"],
        "created_at": utc_now(),
        "workspace": config["workspace"],
        "python_executable": config["python"],
        "train_config": train_config["name"],
        "task": train_config["task"],
        "axis": train_config["axis"],
        "variant": train_config["variant"],
        "budget": int(budget),
        "seed": int(seed),
        "episode_start": episode_start,
        "episode_end": episode_end,
        "output_suffix": str(output_suffix),
        "max_steps_per_demo": int(max_steps_per_demo),
        "success_threshold": active_success_threshold,
        "resolution": {"height": height, "width": width},
        "num_samples": len(samples),
        "num_episodes": len(episodes),
        "success_count": success_count,
        "success_rate": float(success_count / len(episodes)) if episodes else 0.0,
        "sample_file": str(sample_path),
        "config_file": "configs/dataset_128px_v1.yaml",
        "scene_distribution": {
            "visual_axes": config["visual_axes"],
            "train_config": train_config,
        },
        "quality_checks": checks,
        "episodes": episodes,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def collect_matrix(
    config_path: Path,
    output_dir: Path,
    train_config_names: list[str] | None,
    budgets: list[int] | None,
    seeds: list[int] | None,
    max_steps_per_demo: int | None,
    summary_path: Path | None,
    success_threshold: float | None = None,
    episode_start: int = 0,
    episode_end: int | None = None,
    output_suffix: str = "",
) -> dict[str, Any]:
    config = load_yaml(config_path)
    configs = selected_train_configs(config, train_config_names)
    budgets = budgets if budgets is not None else list(config["collection"]["budgets"])
    seeds = seeds if seeds is not None else list(config["collection"]["seeds"])
    max_steps = int(max_steps_per_demo if max_steps_per_demo is not None else config["collection"]["max_steps_per_demo"])

    records = []
    for train_config in configs:
        for budget in budgets:
            for seed in seeds:
                metadata = collect_one_dataset(
                    config=config,
                    train_config=train_config,
                    budget=int(budget),
                    seed=int(seed),
                    max_steps_per_demo=max_steps,
                    output_dir=output_dir,
                    success_threshold=success_threshold,
                    episode_start=episode_start,
                    episode_end=episode_end,
                    output_suffix=output_suffix,
                )
                records.append(
                    {
                        "train_config": metadata["train_config"],
                        "task": metadata["task"],
                        "axis": metadata["axis"],
                        "variant": metadata["variant"],
                        "budget": metadata["budget"],
                        "seed": metadata["seed"],
                        "num_samples": metadata["num_samples"],
                        "success_rate": metadata["success_rate"],
                        "sample_file": metadata["sample_file"],
                    }
                )

    summary = {
        "version": config["version"],
        "created_at": utc_now(),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "train_configs": [item["name"] for item in configs],
        "budgets": [int(item) for item in budgets],
        "seeds": [int(item) for item in seeds],
        "max_steps_per_demo": max_steps,
        "success_threshold": success_threshold,
        "episode_start": int(episode_start),
        "episode_end": episode_end,
        "output_suffix": str(output_suffix),
        "expected_dataset_count": len(configs) * len(budgets) * len(seeds),
        "completed_dataset_count": len(records),
        "records": records,
    }
    if summary_path is None:
        summary_path = output_dir / "collection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_128px_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/datasets_128px_v1"))
    parser.add_argument("--train-configs", nargs="+")
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-steps-per-demo", type=int)
    parser.add_argument("--success-threshold", type=float)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--summary-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect_matrix(
        config_path=args.config,
        output_dir=args.output_dir,
        train_config_names=args.train_configs,
        budgets=args.budgets,
        seeds=args.seeds,
        max_steps_per_demo=args.max_steps_per_demo,
        summary_path=args.summary_path,
        success_threshold=args.success_threshold,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        output_suffix=args.output_suffix,
    )
    print(json.dumps({"completed_dataset_count": summary["completed_dataset_count"], "output_dir": summary["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
