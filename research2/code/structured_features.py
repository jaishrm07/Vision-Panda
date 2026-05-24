#!/usr/bin/env python3
"""Structured phase and geometry features for research2 BC policies."""

from __future__ import annotations

from typing import Any

import numpy as np

from bc_model import AVOID_PHASE_LABELS


PHASE_TO_INDEX = {label: index for index, label in enumerate(AVOID_PHASE_LABELS)}
STRUCTURED_STATE_DIM = 18


def phase_label_to_index(phase_label: str | None) -> int:
    if phase_label is None:
        return -100
    return int(PHASE_TO_INDEX.get(str(phase_label), -100))


def phase_one_hot(phase_index: int) -> np.ndarray:
    values = np.zeros(len(AVOID_PHASE_LABELS), dtype=np.float32)
    if 0 <= int(phase_index) < len(AVOID_PHASE_LABELS):
        values[int(phase_index)] = 1.0
    return values


def vector3(values: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if values is None:
        return np.asarray(default, dtype=np.float32)
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size < 3:
        padded = np.asarray(default, dtype=np.float32).copy()
        padded[: array.size] = array
        return padded
    return array[:3].astype(np.float32, copy=False)


def structured_state_array(
    ee_position: Any,
    target_position: Any,
    obstacle: dict[str, Any] | None,
    phase_index: int,
) -> np.ndarray:
    """Return phase one-hot plus compact obstacle-aware geometry features."""
    ee = vector3(ee_position)
    target = vector3(target_position)
    obstacle = obstacle or {}
    obstacle_center = vector3(obstacle.get("center"))
    obstacle_half_extents = vector3(obstacle.get("half_extents"))
    features = np.concatenate(
        (
            phase_one_hot(int(phase_index)),
            target,
            obstacle_center,
            obstacle_half_extents,
            target - ee,
            target - obstacle_center,
        )
    ).astype(np.float32)
    if features.shape != (STRUCTURED_STATE_DIM,):
        raise ValueError(f"Structured feature shape mismatch: {features.shape}")
    return features
