#!/usr/bin/env python3
"""Record a closed-loop BC rollout from a saved eval episode."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from bc_model import build_policy, extract_action_prediction, load_checkpoint_payload, resolve_model_family
from collect_dataset import ACTION_MAGNITUDE, POSITION_GAIN, clip_action, expert_action, task_distance
from evaluate_bc import (
    ClosedLoopEvalEnv,
    DEFAULT_STOP_THRESHOLD,
    deserialize_scene,
    load_eval_payload,
    normalize_success_thresholds,
    select_device,
    tensor_image,
    update_threshold_crossings,
)
from preview_simulator import CameraConfig, RenderConfig, compute_eef_view, compute_external_view, render_rgb
from structured_features import phase_label_to_index, structured_state_array
from train_bc import write_json


def draw_overlay(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    frame = frame_rgb.copy()
    overlay = frame.copy()
    font_scale = max(0.45, min(1.0, frame.shape[0] / 1080.0 * 0.9))
    thickness = max(1, int(round(font_scale * 2.0)))
    line_height = int(round(26 * font_scale))
    top_pad = int(round(18 * font_scale))
    height = top_pad + line_height * len(lines)
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], height), (0, 0, 0), thickness=-1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    for line_idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (int(round(12 * font_scale)), top_pad + line_height * line_idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return frame


def add_eef_inset(external_rgb: np.ndarray, eef_rgb: np.ndarray) -> np.ndarray:
    frame = external_rgb.copy()
    height, width = frame.shape[:2]
    margin = max(8, int(round(width * 0.018)))
    inset_width = min(int(round(width * 0.32)), 320)
    inset_height = max(1, int(round(inset_width * eef_rgb.shape[0] / eef_rgb.shape[1])))
    inset = cv2.resize(eef_rgb, (inset_width, inset_height), interpolation=cv2.INTER_AREA)

    border = max(2, int(round(width * 0.004)))
    x0 = width - inset_width - margin - 2 * border
    y0 = height - inset_height - margin - 2 * border
    x1 = width - margin
    y1 = height - margin
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1)
    frame[y0 + border : y0 + border + inset_height, x0 + border : x0 + border + inset_width] = inset

    label_h = max(18, int(round(inset_height * 0.16)))
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0 + border, y0 + border),
        (x0 + border + inset_width, y0 + border + label_h),
        (0, 0, 0),
        thickness=-1,
    )
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.putText(
        frame,
        "wrist camera",
        (x0 + border + 8, y0 + border + label_h - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def combine_views(external_rgb: np.ndarray, eef_rgb: np.ndarray, view: str) -> np.ndarray:
    if view == "external":
        return external_rgb
    if view == "eef":
        return eef_rgb
    if view == "external_with_eef_inset":
        return add_eef_inset(external_rgb, eef_rgb)
    if view == "side_by_side":
        return np.concatenate([external_rgb, eef_rgb], axis=1)
    raise ValueError(f"Unsupported view: {view}")


class StreamingVideoWriter:
    def __init__(self, path: Path, first_frame_rgb: np.ndarray, fps: float, encoder: str = "auto", crf: int = 18):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.proc: subprocess.Popen[bytes] | None = None
        self.cv_writer: cv2.VideoWriter | None = None
        self.stderr = b""
        height, width = first_frame_rgb.shape[:2]
        ffmpeg = shutil.which("ffmpeg")
        if encoder in {"auto", "ffmpeg"} and ffmpeg:
            cmd = [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(float(fps)),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(int(crf)),
                "-pix_fmt",
                "yuv420p",
                str(path),
            ]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.encoder_name = "ffmpeg-libx264"
            return

        if encoder == "ffmpeg":
            raise RuntimeError("ffmpeg requested but not found on PATH")
        self.cv_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not self.cv_writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")
        self.encoder_name = "opencv-mp4v"

    def write(self, frame_rgb: np.ndarray) -> None:
        if self.proc is not None:
            if self.proc.stdin is None:
                raise RuntimeError("ffmpeg stdin is closed")
            self.proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
            return
        if self.cv_writer is None:
            raise RuntimeError("Video writer is not initialized")
        self.cv_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> str:
        if self.proc is not None:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                self.proc.stdin.close()
            self.stderr = self.proc.stderr.read() if self.proc.stderr is not None else b""
            self.proc.wait()
            if self.proc.returncode != 0:
                raise RuntimeError(self.stderr.decode("utf-8", errors="replace")[-4000:])
            return self.encoder_name
        if self.cv_writer is not None:
            self.cv_writer.release()
        return self.encoder_name


def scaled_external_camera(scene_camera: CameraConfig, distance_scale: float) -> CameraConfig:
    return CameraConfig(
        yaw=scene_camera.yaw,
        pitch=scene_camera.pitch,
        distance=float(scene_camera.distance) * float(distance_scale),
        target_position=scene_camera.target_position,
    )


def render_video_views(
    env: ClosedLoopEvalEnv,
    scene: Any,
    render_cfg: RenderConfig,
    external_distance_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if env.panda is None:
        raise RuntimeError("Panda is not initialized.")
    video_camera = scaled_external_camera(scene.camera, external_distance_scale)
    external_rgb = render_rgb(compute_external_view(video_camera), render_cfg, scene.lighting)
    eef_rgb = render_rgb(compute_eef_view(env.panda), render_cfg, scene.lighting)
    return external_rgb, eef_rgb


def select_episode(payload: dict[str, Any], episode_index: int) -> dict[str, Any]:
    for fallback_idx, episode in enumerate(payload["episodes"]):
        if int(episode.get("episode_index", fallback_idx)) == int(episode_index):
            return episode
    raise ValueError(f"Episode index {episode_index} not found.")


def record_rollout(
    model_path: Path,
    eval_dataset: Path,
    episode_index: int,
    output: Path,
    model_family: str | None = None,
    device_name: str = "auto",
    width: int = 128,
    height: int = 128,
    video_width: int | None = None,
    video_height: int | None = None,
    video_fov: float = 60.0,
    video_external_distance_scale: float = 1.0,
    fps: float = 20.0,
    view: str = "external",
    video_stride: int = 1,
    encoder: str = "auto",
    crf: int = 18,
    steps_per_episode: int = 400,
    success_thresholds: list[float] | None = None,
    stop_threshold: float = DEFAULT_STOP_THRESHOLD,
    position_gain: float = POSITION_GAIN,
) -> dict[str, Any]:
    threshold_specs = normalize_success_thresholds(success_thresholds)
    device = select_device(device_name)
    checkpoint_payload = load_checkpoint_payload(model_path, map_location=device)
    resolved_model_family = resolve_model_family(model_family, checkpoint_payload)
    model = build_policy(resolved_model_family).to(device)
    model.load_state_dict(checkpoint_payload["state_dict"])
    model.eval()

    payload = load_eval_payload(eval_dataset)
    episode = select_episode(payload, episode_index)
    scene = deserialize_scene(episode["initial_scene"])
    render_cfg = RenderConfig(width=int(width), height=int(height))
    video_render_cfg = RenderConfig(
        width=int(video_width if video_width is not None else width),
        height=int(video_height if video_height is not None else height),
        fov=float(video_fov),
    )
    env = ClosedLoopEvalEnv(render_cfg)

    video_writer: StreamingVideoWriter | None = None
    frames_written = 0
    video_encoder = "not_written"
    success_flags = {label: False for label, _ in threshold_specs}
    first_steps = {label: None for label, _ in threshold_specs}
    best_distance = float("inf")
    final_distance = float("inf")
    steps_taken = 0
    stopped_early = False
    stop_reason = "horizon"
    phase_label = "none"
    try:
        env.reset_episode(scene)
        initial_distance = task_distance(scene.task, env.ee_position(), env.cube_position())
        best_distance = float(initial_distance)
        final_distance = float(initial_distance)
        update_threshold_crossings(initial_distance, 0, threshold_specs, success_flags, first_steps)
        if initial_distance <= stop_threshold:
            stopped_early = True
            stop_reason = "stop_threshold"

        controller_state: dict[str, Any] = {}
        for step_idx in range(int(steps_per_episode)):
            robot_state = env.robot_state(scene)
            ee_position = np.asarray(robot_state["ee_position"], dtype=np.float32)
            cube_position = env.cube_position()
            distance = float(task_distance(scene.task, ee_position, cube_position))
            best_distance = min(best_distance, distance)
            final_distance = distance
            update_threshold_crossings(distance, step_idx, threshold_specs, success_flags, first_steps)

            should_record_video = (
                step_idx % max(1, int(video_stride)) == 0
                or stopped_early
                or distance <= stop_threshold
                or step_idx == int(steps_per_episode) - 1
            )
            if should_record_video:
                video_external, video_eef = render_video_views(
                    env,
                    scene,
                    video_render_cfg,
                    external_distance_scale=video_external_distance_scale,
                )
                frame = combine_views(video_external, video_eef, view)
                frame = draw_overlay(
                    frame,
                    [
                        f"{resolved_model_family} | seed {payload.get('seed')} ep {episode_index}",
                        f"step {step_idx:03d} | dist {100.0 * distance:.2f} cm | best {100.0 * best_distance:.2f} cm | phase {phase_label}",
                    ],
                )
                if video_writer is None:
                    video_writer = StreamingVideoWriter(output, frame, fps=fps, encoder=encoder, crf=crf)
                    video_encoder = video_writer.encoder_name
                video_writer.write(frame)
                frames_written += 1

            if stopped_early or distance <= stop_threshold:
                stopped_early = True
                stop_reason = "stop_threshold"
                break

            external = tensor_image(robot_state["external_rgb"], device)
            eef = tensor_image(robot_state["eef_rgb"], device)
            state = torch.tensor(ee_position, dtype=torch.float32, device=device).unsqueeze(0)
            structured_state = None
            if bool(getattr(model, "uses_structured_state", False)):
                phase_index = -100
                if scene.task == "avoid_reach":
                    expert_action(scene.task, ee_position, cube_position, scene.obstacle, controller_state)
                    phase_label = str(controller_state.get("avoid_phase", "none"))
                    phase_index = phase_label_to_index(phase_label)
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
    finally:
        if video_writer is not None:
            video_encoder = video_writer.close()
        env.close()

    summary = {
        "model_path": str(model_path),
        "model_family": resolved_model_family,
        "eval_dataset": str(eval_dataset),
        "episode_index": int(episode_index),
        "output": str(output),
        "view": view,
        "model_width": int(width),
        "model_height": int(height),
        "video_width": int(video_render_cfg.width),
        "video_height": int(video_render_cfg.height),
        "video_fov": float(video_render_cfg.fov),
        "video_external_distance_scale": float(video_external_distance_scale),
        "video_stride": int(video_stride),
        "video_encoder": video_encoder,
        "frames": int(frames_written),
        "fps": float(fps),
        "steps_taken": int(steps_taken),
        "best_distance": float(best_distance),
        "final_distance": float(final_distance),
        "success_thresholds": {label: threshold for label, threshold in threshold_specs},
        "success": {label: bool(success_flags[label]) for label, _ in threshold_specs},
        "first_steps": {label: first_steps[label] for label, _ in threshold_specs},
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
    }
    write_json(output.with_suffix(".json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-family")
    parser.add_argument("--eval-dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--video-width", type=int)
    parser.add_argument("--video-height", type=int)
    parser.add_argument("--video-fov", type=float, default=60.0)
    parser.add_argument("--video-external-distance-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--view", choices=["external", "eef", "external_with_eef_inset", "side_by_side"], default="external")
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--encoder", choices=["auto", "ffmpeg", "opencv"], default="auto")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--steps-per-episode", type=int, default=400)
    parser.add_argument("--success-thresholds", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--stop-threshold", type=float, default=DEFAULT_STOP_THRESHOLD)
    parser.add_argument("--position-gain", type=float, default=POSITION_GAIN)
    args = parser.parse_args()
    summary = record_rollout(
        model_path=args.model_path,
        model_family=args.model_family,
        eval_dataset=args.eval_dataset,
        episode_index=args.episode_index,
        output=args.output,
        device_name=args.device,
        width=args.width,
        height=args.height,
        video_width=args.video_width,
        video_height=args.video_height,
        video_fov=args.video_fov,
        video_external_distance_scale=args.video_external_distance_scale,
        fps=args.fps,
        view=args.view,
        video_stride=args.video_stride,
        encoder=args.encoder,
        crf=args.crf,
        steps_per_episode=args.steps_per_episode,
        success_thresholds=args.success_thresholds,
        stop_threshold=args.stop_threshold,
        position_gain=args.position_gain,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
