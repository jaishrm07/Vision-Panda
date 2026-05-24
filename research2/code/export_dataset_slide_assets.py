#!/usr/bin/env python3
"""Export dataset-diversity slide assets from real obstacle-aware datasets."""

from __future__ import annotations

import gc
import html
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/jaisharma/HW8/research2")
DATASET_DIR = ROOT / "results/datasets_128px_v1_phase_precise_avoid"
OUT_DIR = ROOT / "analysis/dataset_slides"
RAW_DIR = OUT_DIR / "raw_images"

CONFIGS = {
    "color": {
        "label": "Color Diversity",
        "config": "avoid_color_multi",
        "view": "eef_rgb",
        "caption": "Target cube color changes while spatial layout, camera, and default lighting stay fixed.",
    },
    "camera": {
        "label": "Camera View Diversity",
        "config": "avoid_camera_multi_pose",
        "view": "external_rgb",
        "caption": "External camera yaw, pitch, and distance change across demonstrations.",
    },
    "spatial": {
        "label": "Spatial Diversity",
        "config": "avoid_spatial_wide",
        "view": "external_rgb",
        "caption": "Target position changes across the tabletop while color, camera, and default lighting stay fixed.",
    },
    "lighting": {
        "label": "Lighting Diversity",
        "config": "avoid_lighting_diverse",
        "view": "external_rgb",
        "caption": "Light direction and intensity coefficients change while scene geometry stays fixed.",
    },
}

AXIS_ORDER = ["color", "camera", "spatial", "lighting"]
RAW_RECORDS: list[dict[str, Any]] = []

COLORS = {
    "bg": (246, 248, 251),
    "panel": (255, 255, 255),
    "ink": (18, 23, 33),
    "muted": (91, 103, 118),
    "line": (209, 218, 230),
    "accent": (31, 111, 139),
    "color": (47, 111, 173),
    "camera": (38, 132, 90),
    "spatial": (199, 67, 67),
    "lighting": (198, 106, 43),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = font(42, True)
FONT_SUBTITLE = font(23)
FONT_SECTION = font(25, True)
FONT_LABEL = font(18, True)
FONT_SMALL = font(15)
FONT_TINY = font(13)


def load_dataset(config: str) -> dict[str, Any]:
    path = DATASET_DIR / f"dataset__{config}__budget200__seed000.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def group_episodes(samples: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        episodes[int(sample["episode_index"])].append(sample)
    for items in episodes.values():
        items.sort(key=lambda s: int(s["step_index"]))
    return dict(episodes)


def scene_for(items: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(items[0]["scene"])


def frame_from_sample(sample: dict[str, Any], view: str) -> Image.Image:
    arr = sample[view]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_raw_frame(
    image: Image.Image,
    section: str,
    axis: str,
    sample: dict[str, Any],
    view: str,
    descriptor_text: str,
    frame_index: int | None = None,
) -> str:
    ep = int(sample["episode_index"])
    step = int(sample["step_index"])
    frame_part = "" if frame_index is None else f"__frame{frame_index:02d}"
    filename = f"{axis}__ep{ep:03d}{frame_part}__step{step:03d}__{view}.png"
    out_dir = RAW_DIR / section / axis
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    image.save(out_path)
    rel_path = out_path.relative_to(OUT_DIR).as_posix()
    RAW_RECORDS.append(
        {
            "section": section,
            "axis": axis,
            "episode_index": ep,
            "step_index": step,
            "view": view,
            "descriptor": descriptor_text,
            "path": rel_path,
        }
    )
    return rel_path


def pick_frame(items: list[dict[str, Any]], frac: float = 0.55) -> dict[str, Any]:
    index = max(0, min(len(items) - 1, int(round(frac * (len(items) - 1)))))
    return items[index]


def color_visibility_score(sample: dict[str, Any], view: str) -> float:
    scene = sample["scene"]
    rgba = scene.get("target_rgba", [1.0, 0.0, 0.0, 1.0])
    target = np.array(rgba[:3], dtype=np.float32) * 255.0
    arr = sample[view].astype(np.float32)
    dist = np.linalg.norm(arr - target[None, None, :], axis=2)
    return float(np.maximum(0.0, 95.0 - dist).sum())


def pick_overview_frame(axis: str, items: list[dict[str, Any]], view: str) -> dict[str, Any]:
    if axis == "color":
        return max(items, key=lambda sample: color_visibility_score(sample, view))
    return pick_frame(items, 0.55)


def evenly_spaced(items: list[dict[str, Any]], n: int = 8) -> list[dict[str, Any]]:
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int).tolist()
    return [items[i] for i in idx]


def descriptor(axis: str, scene: dict[str, Any]) -> str:
    if axis == "color":
        return str(scene.get("target_color_name", "unknown"))
    if axis == "camera":
        camera = scene["camera"]
        return f"yaw {camera['yaw']:.0f}, pitch {camera['pitch']:.0f}, dist {camera['distance']:.2f}"
    if axis == "spatial":
        target = scene["target_position"]
        return f"target x={target[0]:.2f}, y={target[1]:.2f}"
    lighting = scene["lighting"]
    direction = lighting.get("light_direction", [0, 0, -1])
    ambient = lighting.get("light_ambient_coeff", 0)
    diffuse = lighting.get("light_diffuse_coeff", 0)
    return f"dir [{direction[0]:.1f},{direction[1]:.1f},{direction[2]:.1f}], a={ambient:.2f}, d={diffuse:.2f}"


def feature(axis: str, scene: dict[str, Any]) -> tuple[float, ...]:
    if axis == "camera":
        camera = scene["camera"]
        return (float(camera["yaw"]), float(camera["pitch"]), float(camera["distance"]) * 50.0)
    if axis == "spatial":
        target = scene["target_position"]
        return (float(target[0]) * 100.0, float(target[1]) * 100.0)
    if axis == "lighting":
        lighting = scene["lighting"]
        direction = lighting.get("light_direction", [0.0, 0.0, -1.0])
        return tuple(float(v) for v in direction) + (
            float(lighting.get("light_ambient_coeff", 0.0)) * 4.0,
            float(lighting.get("light_diffuse_coeff", 0.0)) * 4.0,
            float(lighting.get("light_specular_coeff", 0.0)) * 4.0,
        )
    rgba = scene.get("target_rgba", [1.0, 0.0, 0.0, 1.0])
    return tuple(float(v) for v in rgba[:3])


def distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def farthest_episodes(axis: str, episodes: dict[int, list[dict[str, Any]]], k: int = 4) -> list[int]:
    scenes = {ep: scene_for(items) for ep, items in episodes.items()}
    if axis == "color":
        selected = []
        seen = set()
        for ep, scene in sorted(scenes.items()):
            name = scene.get("target_color_name", "")
            if name not in seen:
                selected.append(ep)
                seen.add(name)
            if len(selected) >= k:
                return selected
        return selected

    feats = {ep: feature(axis, scene) for ep, scene in scenes.items()}
    center = tuple(sum(vals[i] for vals in feats.values()) / len(feats) for i in range(len(next(iter(feats.values())))))
    first = max(feats, key=lambda ep: distance(feats[ep], center))
    selected = [first]
    while len(selected) < k:
        remaining = [ep for ep in feats if ep not in selected]
        selected.append(max(remaining, key=lambda ep: min(distance(feats[ep], feats[s]) for s in selected)))
    return selected


def representative_episode(axis: str, selected: list[int], episodes: dict[int, list[dict[str, Any]]]) -> int:
    if axis == "color":
        for ep in selected:
            if scene_for(episodes[ep]).get("target_color_name") == "blue":
                return ep
    return selected[0]


def resize_frame(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.Resampling.BICUBIC)


def draw_text_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font_obj: ImageFont.ImageFont, fill: tuple[int, int, int], line_spacing: int = 4) -> None:
    x0, y0, x1, _ = box
    max_width = x1 - x0
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        test = word if not cur else f"{cur} {word}"
        if draw.textbbox((0, 0), test, font=font_obj)[2] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    y = y0
    for line in lines:
        draw.text((x0, y), line, font=font_obj, fill=fill)
        y += draw.textbbox((0, 0), line, font=font_obj)[3] + line_spacing


def paste_image_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    xy: tuple[int, int],
    size: tuple[int, int],
    title: str,
    subtitle: str,
    accent: tuple[int, int, int],
) -> None:
    x, y = xy
    w, h = size
    draw.rounded_rectangle((x - 8, y - 8, x + w + 8, y + h + 60), radius=14, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    draw.rounded_rectangle((x - 8, y - 8, x + w + 8, y - 1), radius=8, fill=accent)
    canvas.paste(resize_frame(img, (w, h)), (x, y))
    draw.rectangle((x, y, x + w, y + h), outline=COLORS["line"], width=2)
    draw.text((x, y + h + 9), title, font=FONT_LABEL, fill=COLORS["ink"])
    draw.text((x, y + h + 34), subtitle, font=FONT_TINY, fill=COLORS["muted"])


def make_overview(all_data: dict[str, dict[str, Any]], selected: dict[str, list[int]]) -> None:
    canvas = Image.new("RGB", (1920, 1080), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)
    draw.text((62, 42), "Dataset visual diversity: real obstacle-aware demonstration frames", font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((64, 96), "Each row varies one visual axis while the scripted obstacle-aware task stays the same.", font=FONT_SUBTITLE, fill=COLORS["muted"])

    x_label = 64
    x0 = 385
    y0 = 165
    row_h = 220
    img_w = 250
    img_h = 142
    gap = 36

    for ri, axis in enumerate(AXIS_ORDER):
        spec = CONFIGS[axis]
        y = y0 + ri * row_h
        accent = COLORS[axis]
        draw.rounded_rectangle((x_label, y - 4, x_label + 285, y + 174), radius=14, fill=COLORS["panel"], outline=COLORS["line"], width=2)
        draw.rectangle((x_label, y - 4, x_label + 8, y + 174), fill=accent)
        draw_text_box(draw, (x_label + 22, y + 16, x_label + 265, y + 78), spec["label"], FONT_SECTION, accent)
        draw_text_box(draw, (x_label + 22, y + 82, x_label + 265, y + 150), spec["caption"], FONT_SMALL, COLORS["muted"])

        episodes = all_data[axis]["episodes"]
        view = spec["view"]
        for ci, ep in enumerate(selected[axis]):
            items = episodes[ep]
            sample = pick_overview_frame(axis, items, view)
            scene = scene_for(items)
            img = frame_from_sample(sample, view)
            desc = descriptor(axis, scene)
            save_raw_frame(img, "overview", axis, sample, view, desc)
            title = f"Episode {ep:03d}"
            subtitle = desc
            paste_image_card(canvas, draw, img, (x0 + ci * (img_w + gap), y), (img_w, img_h), title, subtitle, accent)

    canvas.save(OUT_DIR / "01_axis_diversity_overview.png")


def make_trajectory(all_data: dict[str, dict[str, Any]], rep_eps: dict[str, int], view: str, filename: str, title: str, subtitle: str) -> None:
    canvas = Image.new("RGB", (1920, 1080), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)
    draw.text((62, 42), title, font=FONT_TITLE, fill=COLORS["ink"])
    draw.text((64, 96), subtitle, font=FONT_SUBTITLE, fill=COLORS["muted"])

    left = 335
    top = 160
    row_h = 215
    frame = 156
    gap = 19
    n = 8

    for ri, axis in enumerate(AXIS_ORDER):
        spec = CONFIGS[axis]
        episodes = all_data[axis]["episodes"]
        ep = rep_eps[axis]
        items = episodes[ep]
        scene = scene_for(items)
        frames = evenly_spaced(items, n=n)
        y = top + ri * row_h
        accent = COLORS[axis]

        draw.rounded_rectangle((58, y - 8, 306, y + frame + 48), radius=14, fill=COLORS["panel"], outline=COLORS["line"], width=2)
        draw.rectangle((58, y - 8, 66, y + frame + 48), fill=accent)
        draw_text_box(draw, (80, y + 16, 288, y + 78), spec["label"], FONT_SECTION, accent)
        draw_text_box(draw, (80, y + 82, 288, y + 150), descriptor(axis, scene), FONT_SMALL, COLORS["muted"])
        draw.text((80, y + 162), f"episode {ep:03d}", font=FONT_TINY, fill=COLORS["muted"])

        for i, sample in enumerate(frames):
            x = left + i * (frame + gap)
            raw_img = frame_from_sample(sample, view)
            save_raw_frame(raw_img, f"trajectories_{view}", axis, sample, view, descriptor(axis, scene), frame_index=i)
            img = resize_frame(raw_img, (frame, frame))
            draw.rounded_rectangle((x - 6, y - 6, x + frame + 6, y + frame + 31), radius=12, fill=COLORS["panel"], outline=COLORS["line"], width=2)
            canvas.paste(img, (x, y))
            draw.rectangle((x, y, x + frame, y + frame), outline=COLORS["line"], width=2)
            step = int(sample["step_index"])
            label = "start" if i == 0 else ("end" if i == len(frames) - 1 else f"step {step}")
            draw.text((x + 8, y + frame + 8), label, font=FONT_TINY, fill=COLORS["muted"])

    canvas.save(OUT_DIR / filename)


def write_index(manifest: dict[str, Any]) -> None:
    body = "\n".join(
        f'<figure><img src="{html.escape(path)}" alt="{html.escape(path)}"><figcaption>{html.escape(path)}</figcaption></figure>'
        for path in [
            "01_axis_diversity_overview.png",
            "02_obstacle_trajectories_external.png",
            "03_obstacle_trajectories_wrist.png",
        ]
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dataset Slide Assets</title>
<style>
body {{ margin: 24px; font-family: Arial, sans-serif; background: #f6f8fb; color: #121721; }}
figure {{ margin: 0 0 28px; padding: 16px; background: white; border: 1px solid #dde3ec; border-radius: 8px; }}
img {{ width: 100%; height: auto; display: block; }}
figcaption {{ margin-top: 10px; color: #606a78; }}
pre {{ white-space: pre-wrap; background: white; border: 1px solid #dde3ec; padding: 14px; border-radius: 8px; }}
</style>
</head>
<body>
<h1>Dataset Slide Assets</h1>
<p>Generated from real budget-200 obstacle-aware 128px datasets on role-lab.</p>
{body}
<h2>Manifest</h2>
<pre>{html.escape(json.dumps(manifest, indent=2))}</pre>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DIR.exists():
        for path in sorted(RAW_DIR.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RECORDS.clear()
    all_data: dict[str, dict[str, Any]] = {}
    selected: dict[str, list[int]] = {}
    rep_eps: dict[str, int] = {}
    manifest: dict[str, Any] = {
        "source_dataset_dir": str(DATASET_DIR),
        "budget": 200,
        "seed": 0,
        "task": "avoid_reach",
        "axes": {},
    }

    for axis in AXIS_ORDER:
        spec = CONFIGS[axis]
        payload = load_dataset(spec["config"])
        episodes = group_episodes(payload["samples"])
        chosen = farthest_episodes(axis, episodes, k=4)
        rep = representative_episode(axis, chosen, episodes)
        all_data[axis] = {"payload": payload, "episodes": episodes}
        selected[axis] = chosen
        rep_eps[axis] = rep
        manifest["axes"][axis] = {
            "label": spec["label"],
            "train_config": spec["config"],
            "overview_view": spec["view"],
            "selected_overview_episodes": chosen,
            "trajectory_episode": rep,
            "trajectory_descriptor": descriptor(axis, scene_for(episodes[rep])),
            "selected_descriptors": [descriptor(axis, scene_for(episodes[ep])) for ep in chosen],
            "sample_count": len(payload["samples"]),
            "episode_count": len(episodes),
        }

    make_overview(all_data, selected)
    make_trajectory(
        all_data,
        rep_eps,
        "external_rgb",
        "02_obstacle_trajectories_external.png",
        "Obstacle-aware trajectories across visual axes: external camera",
        "Each row is one real demonstration trajectory from the budget-200 obstacle-aware dataset.",
    )
    make_trajectory(
        all_data,
        rep_eps,
        "eef_rgb",
        "03_obstacle_trajectories_wrist.png",
        "Obstacle-aware trajectories across visual axes: wrist camera",
        "Same selected demonstrations, shown from the end-effector camera used by the policy.",
    )

    (OUT_DIR / "dataset_slide_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT_DIR / "raw_image_manifest.json").write_text(json.dumps({"records": RAW_RECORDS}, indent=2), encoding="utf-8")
    write_index(manifest)
    print(json.dumps({"output_dir": str(OUT_DIR), "assets": sorted(p.name for p in OUT_DIR.iterdir())}, indent=2))

    all_data.clear()
    gc.collect()


if __name__ == "__main__":
    main()
