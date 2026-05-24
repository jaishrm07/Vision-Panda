#!/usr/bin/env python3
"""Generate minimal 128x128 simulator previews for the visual-diversity dataset."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data
import yaml


PANDA_EE_LINK_INDEX = 11
DEFAULT_JOINT_START_POSITIONS = [
    0.0,
    0.0,
    0.0,
    -np.pi / 2.0,
    0.0,
    np.pi / 2.0,
    np.pi / 4.0,
    0.0,
    0.0,
    0.04,
    0.04,
]
EEF_CAMERA_DISTANCE = 0.2
EEF_CAMERA_OFFSET_POSITION = [0.05, 0.0, 0.0]
EEF_CAMERA_OFFSET_EULER = [0.0, -np.pi / 2.0, 0.0]


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int
    fov: float = 60.0
    near: float = 0.01
    far: float = 3.0


@dataclass(frozen=True)
class CameraConfig:
    yaw: float
    pitch: float
    distance: float
    target_position: tuple[float, float, float]


@dataclass(frozen=True)
class LightingConfig:
    light_direction: tuple[float, float, float]
    ambient: float
    diffuse: float
    specular: float
    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class SceneConfig:
    train_config: str
    task: str
    axis: str
    variant: str
    target_position: tuple[float, float, float]
    target_color_name: str
    target_rgba: tuple[float, float, float, float]
    camera: CameraConfig
    lighting: LightingConfig | None
    obstacle: dict[str, Any] | None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def as_tuple(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def setup_world() -> int:
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.625])
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0.0, -0.625], useFixedBase=True)
    panda = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[0.0, 0.0, 0.0],
        useFixedBase=True,
    )
    reset_panda_pose(panda)
    return panda


def reset_panda_pose(panda: int) -> None:
    for joint_index, joint_position in enumerate(DEFAULT_JOINT_START_POSITIONS):
        p.resetJointState(panda, joint_index, joint_position)


def create_box(
    half_extents: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    position: tuple[float, float, float],
    mass: float = 0.0,
) -> int:
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def add_scene_objects(scene: SceneConfig) -> None:
    cube = p.loadURDF(
        "cube_small.urdf",
        basePosition=list(scene.target_position),
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        useFixedBase=True,
    )
    p.changeVisualShape(cube, -1, rgbaColor=list(scene.target_rgba))
    if scene.obstacle is not None:
        create_box(
            half_extents=tuple(scene.obstacle["half_extents"]),
            rgba=tuple(scene.obstacle["rgba"]),
            position=tuple(scene.obstacle["center"]),
            mass=0.0,
        )


def compute_external_view(camera: CameraConfig) -> list[float]:
    return p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(camera.target_position),
        distance=camera.distance,
        yaw=camera.yaw,
        pitch=camera.pitch,
        roll=0.0,
        upAxisIndex=2,
    )


def compute_eef_view(panda: int) -> list[float]:
    link_state = p.getLinkState(panda, PANDA_EE_LINK_INDEX, computeForwardKinematics=True)
    ee_position = link_state[4]
    ee_quaternion = link_state[5]
    camera_offset_quaternion = p.getQuaternionFromEuler(EEF_CAMERA_OFFSET_EULER)
    camera_position, camera_orientation = p.multiplyTransforms(
        ee_position,
        ee_quaternion,
        EEF_CAMERA_OFFSET_POSITION,
        camera_offset_quaternion,
    )
    rotation = np.array(p.getMatrixFromQuaternion(camera_orientation)).reshape(3, 3)
    camera_forward = rotation @ np.array([1.0, 0.0, 0.0])
    camera_up = rotation @ np.array([0.0, 0.0, 1.0])
    camera_target = np.asarray(camera_position) + EEF_CAMERA_DISTANCE * camera_forward
    return p.computeViewMatrix(
        cameraEyePosition=list(camera_position),
        cameraTargetPosition=camera_target.tolist(),
        cameraUpVector=camera_up.tolist(),
    )


def render_rgb(
    view_matrix: list[float],
    render: RenderConfig,
    lighting: LightingConfig | None,
) -> np.ndarray:
    projection = p.computeProjectionMatrixFOV(
        fov=render.fov,
        aspect=float(render.width) / float(render.height),
        nearVal=render.near,
        farVal=render.far,
    )
    render_kwargs: dict[str, Any] = {}
    if lighting is not None:
        render_kwargs = {
            "lightDirection": list(lighting.light_direction),
            "lightColor": list(lighting.light_color),
            "lightAmbientCoeff": lighting.ambient,
            "lightDiffuseCoeff": lighting.diffuse,
            "lightSpecularCoeff": lighting.specular,
        }
    _, _, rgba, _, _ = p.getCameraImage(
        width=render.width,
        height=render.height,
        viewMatrix=view_matrix,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_NO_SEGMENTATION_MASK,
        **render_kwargs,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(render.height, render.width, 4)[:, :, :3]


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def sample_range(rng: random.Random, values: list[float]) -> float:
    lo, hi = values
    return rng.uniform(float(lo), float(hi))


def choose_color(axis_cfg: dict[str, Any], variant: str, rng: random.Random) -> tuple[str, tuple[float, float, float, float]]:
    if variant == "fixed_train":
        colors = axis_cfg["fixed_train"]["colors"]
    elif variant == "diverse_train":
        colors = axis_cfg["diverse_train"]["colors"]
    elif variant == "ood_eval":
        colors = axis_cfg["ood_eval"]["colors"]
    else:
        raise ValueError(f"Unsupported color variant: {variant}")
    name = rng.choice(sorted(colors))
    return name, tuple(colors[name])


def choose_edge_balanced_target_position(axis_cfg: dict[str, Any], rng: random.Random, sample_index: int) -> tuple[float, float, float]:
    cfg = axis_cfg["edge_balanced_train"]
    x_min, x_max = [float(item) for item in cfg["x_range"]]
    y_min, y_max = [float(item) for item in cfg["y_range"]]
    interior_x_min, interior_x_max = [float(item) for item in cfg["interior_x_range"]]
    interior_y_min, interior_y_max = [float(item) for item in cfg["interior_y_range"]]
    edge_width = float(cfg["edge_width"])
    bucket_cycle = list(cfg["bucket_cycle"])
    bucket = bucket_cycle[int(sample_index) % len(bucket_cycle)]

    left_x = (x_min, min(x_min + edge_width, x_max))
    right_x = (max(x_max - edge_width, x_min), x_max)
    lower_y = (y_min, min(y_min + edge_width, y_max))
    upper_y = (max(y_max - edge_width, y_min), y_max)

    if bucket == "left_edge":
        x_range, y_range = left_x, (interior_y_min, interior_y_max)
    elif bucket == "right_edge":
        x_range, y_range = right_x, (interior_y_min, interior_y_max)
    elif bucket == "lower_edge":
        x_range, y_range = (interior_x_min, interior_x_max), lower_y
    elif bucket == "upper_edge":
        x_range, y_range = (interior_x_min, interior_x_max), upper_y
    elif bucket == "lower_left_corner":
        x_range, y_range = left_x, lower_y
    elif bucket == "lower_right_corner":
        x_range, y_range = right_x, lower_y
    elif bucket == "upper_left_corner":
        x_range, y_range = left_x, upper_y
    elif bucket == "upper_right_corner":
        x_range, y_range = right_x, upper_y
    elif bucket == "interior":
        x_range, y_range = (interior_x_min, interior_x_max), (interior_y_min, interior_y_max)
    else:
        raise ValueError(f"Unsupported edge-balanced spatial bucket: {bucket}")
    return (sample_range(rng, x_range), sample_range(rng, y_range), 0.025)


def choose_target_position(axis_cfg: dict[str, Any], variant: str, rng: random.Random, sample_index: int = 0) -> tuple[float, float, float]:
    if variant == "narrow_train":
        ranges = axis_cfg["narrow_train"]
    elif variant == "wide_train":
        ranges = axis_cfg["wide_train"]
    elif variant == "edge_balanced_train":
        return choose_edge_balanced_target_position(axis_cfg, rng, sample_index=sample_index)
    elif variant == "edge_ood_eval":
        ranges = axis_cfg["edge_ood_eval"]
    else:
        raise ValueError(f"Unsupported spatial variant: {variant}")
    return (sample_range(rng, ranges["x_range"]), sample_range(rng, ranges["y_range"]), 0.025)


def choose_camera(axis_cfg: dict[str, Any], variant: str, rng: random.Random) -> CameraConfig:
    if variant == "diverse_train":
        cfg = axis_cfg["diverse_train"]
        yaw = sample_range(rng, cfg["yaw_range"])
        pitch = sample_range(rng, cfg["pitch_range"])
        distance = sample_range(rng, cfg["distance_range"])
        target_position = tuple(cfg["target_position"])
    elif variant == "extreme_ood_eval":
        cfg = axis_cfg["extreme_ood_eval"]
        yaw = sample_range(rng, rng.choice(cfg["yaw_ranges"]))
        pitch = sample_range(rng, cfg["pitch_range"])
        distance = sample_range(rng, cfg["distance_range"])
        target_position = tuple(cfg["target_position"])
    elif variant == "fixed_train":
        cfg = axis_cfg["fixed_train"]
        yaw = float(cfg["yaw"])
        pitch = float(cfg["pitch"])
        distance = float(cfg["distance"])
        target_position = tuple(cfg["target_position"])
    else:
        raise ValueError(f"Unsupported camera variant: {variant}")
    return CameraConfig(yaw=yaw, pitch=pitch, distance=distance, target_position=target_position)


def choose_lighting(axis_cfg: dict[str, Any], variant: str, rng: random.Random) -> LightingConfig:
    if variant == "diverse_train":
        cfg = axis_cfg["diverse_train"]
        direction_name = rng.choice(sorted(cfg["direction_presets"]))
        intensity_name = rng.choice(sorted(cfg["intensity_presets"]))
        direction = tuple(cfg["direction_presets"][direction_name])
        intensity = cfg["intensity_presets"][intensity_name]
    elif variant == "extreme_ood_eval":
        cfg = axis_cfg["extreme_ood_eval"]
        direction_name = rng.choice(sorted(cfg["directions"]))
        intensity_name = rng.choice(sorted(cfg["intensity_presets"]))
        direction = tuple(cfg["directions"][direction_name])
        intensity = cfg["intensity_presets"][intensity_name]
    elif variant == "fixed_train":
        cfg = axis_cfg["fixed_train"]
        direction = tuple(cfg["light_direction"])
        intensity = cfg
    else:
        raise ValueError(f"Unsupported lighting variant: {variant}")
    return LightingConfig(
        light_direction=direction,
        ambient=float(intensity["ambient"]),
        diffuse=float(intensity["diffuse"]),
        specular=float(intensity["specular"]),
    )


def fixed_obstacle() -> dict[str, Any]:
    return {
        "shape": "rectangular_wall",
        "center": [0.56, 0.0, 0.075],
        "half_extents": [0.018, 0.065, 0.075],
        "rgba": [0.45, 0.45, 0.48, 1.0],
    }


def build_scene(config: dict[str, Any], train_config: dict[str, str], seed: int, sample_index: int) -> SceneConfig:
    rng = random.Random(seed * 1009 + sample_index * 9176 + sum(ord(c) for c in train_config["name"]))
    axes = config["visual_axes"]
    task = train_config["task"]
    axis = train_config["axis"]
    variant = train_config["variant"]

    target_position = (0.50, 0.0, 0.025)
    target_color_name = "red"
    target_rgba = (1.0, 0.0, 0.0, 1.0)
    camera = choose_camera(axes["camera_location_viewpoint"], "fixed_train", rng)
    lighting = None

    if axis == "color":
        target_color_name, target_rgba = choose_color(axes["color"], variant, rng)
    elif axis == "spatial_distribution":
        target_position = choose_target_position(axes["spatial_distribution"], variant, rng, sample_index=sample_index)
    elif axis == "camera_location_viewpoint":
        camera = choose_camera(axes["camera_location_viewpoint"], variant, rng)
    elif axis == "lighting_direction_intensity":
        lighting = choose_lighting(axes["lighting_direction_intensity"], variant, rng)
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    obstacle = fixed_obstacle() if task == "avoid_reach" else None
    return SceneConfig(
        train_config=train_config["name"],
        task=task,
        axis=axis,
        variant=variant,
        target_position=target_position,
        target_color_name=target_color_name,
        target_rgba=target_rgba,
        camera=camera,
        lighting=lighting,
        obstacle=obstacle,
    )


def serialize_scene(scene: SceneConfig) -> dict[str, Any]:
    return {
        "train_config": scene.train_config,
        "task": scene.task,
        "axis": scene.axis,
        "variant": scene.variant,
        "target_position": list(scene.target_position),
        "target_color_name": scene.target_color_name,
        "target_rgba": list(scene.target_rgba),
        "camera": {
            "yaw": scene.camera.yaw,
            "pitch": scene.camera.pitch,
            "distance": scene.camera.distance,
            "target_position": list(scene.camera.target_position),
        },
        "lighting": serialize_lighting(scene.lighting),
        "obstacle": scene.obstacle,
    }


def serialize_lighting(lighting: LightingConfig | None) -> dict[str, Any]:
    if lighting is None:
        return {"mode": "pybullet_default"}
    return {
        "mode": "explicit",
        "light_direction": list(lighting.light_direction),
        "light_color": list(lighting.light_color),
        "light_ambient_coeff": lighting.ambient,
        "light_diffuse_coeff": lighting.diffuse,
        "light_specular_coeff": lighting.specular,
    }


def generate_previews(config_path: Path, output_dir: Path, seed: int, samples_per_config: int) -> None:
    config = load_yaml(config_path)
    render_cfg = RenderConfig(
        width=int(config["collection"]["resolution"]["width"]),
        height=int(config["collection"]["resolution"]["height"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []

    for train_config in config["train_configs"]:
        for sample_index in range(samples_per_config):
            scene = build_scene(config, train_config, seed=seed, sample_index=sample_index)
            panda = setup_world()
            try:
                add_scene_objects(scene)
                for _ in range(20):
                    p.stepSimulation()
                external_rgb = render_rgb(compute_external_view(scene.camera), render_cfg, scene.lighting)
                eef_rgb = render_rgb(compute_eef_view(panda), render_cfg, scene.lighting)
            finally:
                p.disconnect()

            prefix = f"preview__{scene.train_config}__seed{seed:03d}__sample{sample_index:03d}"
            external_path = output_dir / f"{prefix}__external.png"
            eef_path = output_dir / f"{prefix}__eef.png"
            save_png(external_path, external_rgb)
            save_png(eef_path, eef_rgb)
            index.append(
                {
                    "train_config": scene.train_config,
                    "seed": seed,
                    "sample_index": sample_index,
                    "axis": scene.axis,
                    "variant": scene.variant,
                    "task": scene.task,
                    "external_preview_path": str(external_path),
                    "eef_preview_path": str(eef_path),
                    "external_pixel_std": float(external_rgb.std()),
                    "eef_pixel_std": float(eef_rgb.std()),
                    "scene": serialize_scene(scene),
                }
            )

    index_path = output_dir / "preview_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset_128px_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/previews_128px_v1"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-config", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_previews(
        config_path=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
        samples_per_config=args.samples_per_config,
    )


if __name__ == "__main__":
    main()
