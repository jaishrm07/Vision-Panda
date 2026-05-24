#!/usr/bin/env python3
"""Train a scratch 128px behavior-cloning policy on one research2 dataset."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

from bc_model import AVOID_PHASE_LABELS, MODEL_FAMILY, build_policy, extract_action_prediction, list_model_families, save_policy_checkpoint
from structured_features import structured_state_array


PHASE_TO_INDEX = {label: index for index, label in enumerate(AVOID_PHASE_LABELS)}
DEFAULT_SPATIAL_X_RANGE = (0.34, 0.68)
DEFAULT_SPATIAL_Y_RANGE = (-0.24, 0.24)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_budget_tag(budget: int) -> str:
    return f"budget{int(budget):03d}"


def format_seed_tag(seed: int) -> str:
    return f"seed{int(seed):03d}"


def dataset_stem(config_name: str, budget: int, seed: int) -> str:
    return "__".join(["dataset", config_name, format_budget_tag(budget), format_seed_tag(seed)])


def model_stem(model_family: str, config_name: str, budget: int, seed: int) -> str:
    return "__".join(["model", model_family, config_name, format_budget_tag(budget), format_seed_tag(seed)])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_threads() -> None:
    requested = os.environ.get("TORCH_NUM_THREADS")
    if requested:
        torch.set_num_threads(max(1, int(requested)))


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


class Research2BCDataset(Dataset):
    def __init__(self, dataset_path: Path, max_samples: int | None = None):
        self.dataset_path = Path(dataset_path)
        with self.dataset_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or "samples" not in payload:
            raise ValueError(f"Expected research2 dict dataset with samples: {self.dataset_path}")

        samples = payload["samples"]
        if max_samples is not None:
            samples = samples[: int(max_samples)]
        if not samples:
            raise ValueError(f"Dataset is empty: {self.dataset_path}")

        train_config = str(payload.get("train_config"))
        self.payload_metadata = {
            "version": payload.get("version"),
            "train_config": payload.get("train_config"),
            "task": payload.get("task"),
            "axis": payload.get("axis"),
            "variant": payload.get("variant"),
            "budget": payload.get("budget"),
            "seed": payload.get("seed"),
            "resolution": payload.get("resolution"),
            "num_samples_loaded": len(samples),
            "num_episodes": len(payload.get("episodes", [])),
            "phase_annotation": payload.get("phase_annotation"),
        }
        self.dataset_paths = [self.dataset_path]
        self.samples = samples
        self.phase_labels = [self.phase_label_for_sample(sample) for sample in self.samples]
        self.phase_label_indices = [PHASE_TO_INDEX.get(label, -100) for label in self.phase_labels]
        self.phase_counts = dict(sorted(Counter(self.phase_labels).items()))
        self.spatial_xy = [self.spatial_xy_for_sample(sample) for sample in self.samples]
        self.source_labels = [train_config for _ in self.samples]
        self.source_counts = dict(sorted(Counter(self.source_labels).items()))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def phase_label_for_sample(sample: dict[str, Any]) -> str:
        if sample.get("avoid_phase") is not None:
            return str(sample["avoid_phase"])
        expert_action = sample.get("expert_action", {})
        if expert_action.get("avoid_phase") is not None:
            return str(expert_action["avoid_phase"])
        return "unannotated"

    @staticmethod
    def spatial_xy_for_sample(sample: dict[str, Any]) -> tuple[float, float] | None:
        scene = sample.get("scene", {})
        target_position = scene.get("target_position")
        if target_position is None or len(target_position) < 2:
            return None
        return float(target_position[0]), float(target_position[1])

    @staticmethod
    def structured_state_for_sample(sample: dict[str, Any], phase_label_index: int) -> np.ndarray:
        scene = sample.get("scene", {})
        robot_state = sample.get("robot_state", {})
        return structured_state_array(
            ee_position=robot_state.get("ee_position"),
            target_position=scene.get("target_position"),
            obstacle=scene.get("obstacle"),
            phase_index=int(phase_label_index),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        phase_label_index = self.phase_label_indices[idx]
        external = torch.from_numpy(np.ascontiguousarray(sample["external_rgb"])).permute(2, 0, 1)
        eef = torch.from_numpy(np.ascontiguousarray(sample["eef_rgb"])).permute(2, 0, 1)
        ee_position = torch.tensor(sample["robot_state"]["ee_position"], dtype=torch.float32)
        action = torch.tensor(sample["expert_action"]["delta_position"], dtype=torch.float32)
        return {
            "external_rgb": external,
            "eef_rgb": eef,
            "ee_position": ee_position,
            "action": action,
            "phase_label_index": torch.tensor(phase_label_index, dtype=torch.long),
            "structured_state": torch.from_numpy(self.structured_state_for_sample(sample, phase_label_index)),
        }


class CombinedResearch2BCDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list[Path],
        combined_train_config: str,
        combined_budget: int,
        combined_seed: int,
        max_samples_per_dataset: int | None = None,
        max_samples: int | None = None,
    ):
        if not dataset_paths:
            raise ValueError("CombinedResearch2BCDataset requires at least one dataset path.")
        self.dataset_paths = [Path(path) for path in dataset_paths]
        self.samples: list[dict[str, Any]] = []
        self.source_labels: list[str] = []
        source_metadata: list[dict[str, Any]] = []
        for dataset_path in self.dataset_paths:
            with dataset_path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict) or "samples" not in payload:
                raise ValueError(f"Expected research2 dict dataset with samples: {dataset_path}")
            source_samples = list(payload["samples"])
            if max_samples_per_dataset is not None:
                source_samples = source_samples[: int(max_samples_per_dataset)]
            if not source_samples:
                raise ValueError(f"Source dataset is empty after truncation: {dataset_path}")
            source_label = str(payload.get("train_config"))
            self.samples.extend(source_samples)
            self.source_labels.extend([source_label for _ in source_samples])
            source_metadata.append(
                {
                    "dataset_path": str(dataset_path),
                    "train_config": payload.get("train_config"),
                    "task": payload.get("task"),
                    "axis": payload.get("axis"),
                    "variant": payload.get("variant"),
                    "budget": payload.get("budget"),
                    "seed": payload.get("seed"),
                    "resolution": payload.get("resolution"),
                    "num_samples_loaded": len(source_samples),
                    "num_episodes": len(payload.get("episodes", [])),
                    "phase_annotation": payload.get("phase_annotation"),
                }
            )

        if max_samples is not None:
            keep = int(max_samples)
            self.samples = self.samples[:keep]
            self.source_labels = self.source_labels[:keep]
        if not self.samples:
            raise ValueError("Combined dataset is empty.")

        tasks = sorted({str(item.get("task")) for item in source_metadata})
        axes = sorted({str(item.get("axis")) for item in source_metadata})
        self.payload_metadata = {
            "version": "combined_existing_datasets",
            "train_config": combined_train_config,
            "task": tasks[0] if len(tasks) == 1 else "mixed",
            "axis": axes[0] if len(axes) == 1 else "mixed",
            "variant": "spatial_balanced_existing",
            "budget": int(combined_budget),
            "seed": int(combined_seed),
            "resolution": source_metadata[0].get("resolution"),
            "num_samples_loaded": len(self.samples),
            "num_episodes": sum(int(item.get("num_episodes", 0)) for item in source_metadata),
            "phase_annotation": {
                "method": "combined_existing_datasets_with_spatial_phase_balanced_sampling",
                "source_dataset_count": len(source_metadata),
            },
            "source_datasets": source_metadata,
        }
        self.phase_labels = [Research2BCDataset.phase_label_for_sample(sample) for sample in self.samples]
        self.phase_label_indices = [PHASE_TO_INDEX.get(label, -100) for label in self.phase_labels]
        self.phase_counts = dict(sorted(Counter(self.phase_labels).items()))
        self.spatial_xy = [Research2BCDataset.spatial_xy_for_sample(sample) for sample in self.samples]
        self.source_counts = dict(sorted(Counter(self.source_labels).items()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        phase_label_index = self.phase_label_indices[idx]
        external = torch.from_numpy(np.ascontiguousarray(sample["external_rgb"])).permute(2, 0, 1)
        eef = torch.from_numpy(np.ascontiguousarray(sample["eef_rgb"])).permute(2, 0, 1)
        ee_position = torch.tensor(sample["robot_state"]["ee_position"], dtype=torch.float32)
        action = torch.tensor(sample["expert_action"]["delta_position"], dtype=torch.float32)
        return {
            "external_rgb": external,
            "eef_rgb": eef,
            "ee_position": ee_position,
            "action": action,
            "phase_label_index": torch.tensor(phase_label_index, dtype=torch.long),
            "structured_state": torch.from_numpy(Research2BCDataset.structured_state_for_sample(sample, phase_label_index)),
        }


def split_indices(num_items: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(num_items))
    if val_fraction <= 0.0 or num_items < 2:
        return indices, []
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(round(num_items * val_fraction))
    val_count = max(1, min(num_items - 1, val_count))
    return indices[val_count:], indices[:val_count]


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator if shuffle else None,
    )


def build_phase_balanced_sampler(dataset: Dataset, train_indices: list[int]) -> tuple[WeightedRandomSampler | None, dict[str, Any]]:
    labels = [dataset.phase_labels[index] for index in train_indices]
    counts = Counter(labels)
    summary: dict[str, Any] = {
        "enabled": False,
        "train_phase_counts": dict(sorted(counts.items())),
        "reason": None,
    }
    if not labels:
        summary["reason"] = "empty_train_indices"
        return None, summary
    if set(counts) == {"unannotated"}:
        summary["reason"] = "no_phase_annotations"
        return None, summary
    if len(counts) < 2:
        summary["reason"] = "only_one_phase_present"
        return None, summary

    weights = [1.0 / float(counts[label]) for label in labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    summary.update(
        {
            "enabled": True,
            "reason": "inverse_frequency_weighted_sampler",
            "num_samples_per_epoch": len(weights),
            "phase_weight_by_label": {label: 1.0 / float(count) for label, count in sorted(counts.items())},
        }
    )
    return sampler, summary


def spatial_bin_label(
    xy: tuple[float, float] | None,
    bins_x: int,
    bins_y: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> str:
    if xy is None:
        return "unknown_xy"
    x, y = xy
    x_min, x_max = x_range
    y_min, y_max = y_range
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Spatial ranges must have max > min.")
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    x_bin = int(np.floor(x_norm * bins_x))
    y_bin = int(np.floor(y_norm * bins_y))
    x_bin = max(0, min(int(bins_x) - 1, x_bin))
    y_bin = max(0, min(int(bins_y) - 1, y_bin))
    return f"x{x_bin}_y{y_bin}"


def build_spatial_phase_balanced_sampler(
    dataset: Dataset,
    train_indices: list[int],
    bins_x: int,
    bins_y: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[WeightedRandomSampler | None, dict[str, Any]]:
    if not train_indices:
        return None, {"enabled": False, "reason": "empty_train_indices"}
    if not hasattr(dataset, "spatial_xy") or not hasattr(dataset, "phase_labels"):
        return None, {"enabled": False, "reason": "dataset_missing_spatial_or_phase_labels"}

    joint_labels = []
    spatial_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_labels = getattr(dataset, "source_labels", ["unknown_source"] * len(dataset))
    for index in train_indices:
        phase_label = str(dataset.phase_labels[index])
        spatial_label = spatial_bin_label(dataset.spatial_xy[index], bins_x=bins_x, bins_y=bins_y, x_range=x_range, y_range=y_range)
        source_label = str(source_labels[index])
        joint_label = f"{phase_label}|{spatial_label}"
        joint_labels.append(joint_label)
        phase_counts[phase_label] += 1
        spatial_counts[spatial_label] += 1
        source_counts[source_label] += 1

    joint_counts = Counter(joint_labels)
    if len(joint_counts) < 2:
        return None, {
            "enabled": False,
            "reason": "only_one_joint_phase_spatial_bin_present",
            "train_joint_counts": dict(sorted(joint_counts.items())),
            "train_phase_counts": dict(sorted(phase_counts.items())),
            "train_spatial_bin_counts": dict(sorted(spatial_counts.items())),
            "train_source_counts": dict(sorted(source_counts.items())),
        }

    weights = [1.0 / float(joint_counts[label]) for label in joint_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler, {
        "enabled": True,
        "reason": "inverse_frequency_weighted_sampler_over_phase_and_spatial_bin",
        "num_samples_per_epoch": len(weights),
        "bins_x": int(bins_x),
        "bins_y": int(bins_y),
        "x_range": [float(x_range[0]), float(x_range[1])],
        "y_range": [float(y_range[0]), float(y_range[1])],
        "train_joint_counts": dict(sorted(joint_counts.items())),
        "train_phase_counts": dict(sorted(phase_counts.items())),
        "train_spatial_bin_counts": dict(sorted(spatial_counts.items())),
        "train_source_counts": dict(sorted(source_counts.items())),
        "joint_weight_by_label": {label: 1.0 / float(count) for label, count in sorted(joint_counts.items())},
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    for batch in loader:
        external = batch["external_rgb"].to(device, non_blocking=True)
        eef = batch["eef_rgb"].to(device, non_blocking=True)
        state = batch["ee_position"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        structured_state = batch.get("structured_state")
        if structured_state is not None:
            structured_state = structured_state.to(device, non_blocking=True)
        phase_label_index = batch.get("phase_label_index")
        if phase_label_index is not None:
            phase_label_index = phase_label_index.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if bool(getattr(model, "uses_structured_state", False)):
                if structured_state is None:
                    raise ValueError("Structured model requires structured_state in the training batch.")
                prediction = model(external, eef, state, structured_state)
            else:
                prediction = model(external, eef, state)
            action_prediction = extract_action_prediction(prediction)
            loss = model.mse_loss(action_prediction, action)
            if isinstance(prediction, dict) and "phase_logits" in prediction and phase_label_index is not None:
                valid_phase_count = int((phase_label_index >= 0).sum().detach().cpu().item())
                if valid_phase_count > 0:
                    phase_loss = model.phase_loss(prediction["phase_logits"], phase_label_index)
                    loss = loss + float(getattr(model, "phase_loss_weight", 0.001)) * phase_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr: float | None = None,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    parameter_group_builder = getattr(model, "optimizer_parameter_groups", None)
    if callable(parameter_group_builder):
        raw_groups = parameter_group_builder(lr=lr, weight_decay=weight_decay, backbone_lr=backbone_lr)
        parameter_groups = []
        optimizer_summary = []
        for index, raw_group in enumerate(raw_groups):
            params = [parameter for parameter in raw_group["params"] if parameter.requires_grad]
            if not params:
                continue
            group = dict(raw_group)
            group["params"] = params
            parameter_groups.append(group)
            optimizer_summary.append(
                {
                    "index": index,
                    "name": str(raw_group.get("name", f"group_{index}")),
                    "lr": float(raw_group.get("lr", lr)),
                    "weight_decay": float(raw_group.get("weight_decay", weight_decay)),
                    "num_parameters": int(sum(parameter.numel() for parameter in params)),
                }
            )
        if not parameter_groups:
            raise ValueError("No trainable parameters found in model optimizer parameter groups.")
        return torch.optim.Adam(parameter_groups), optimizer_summary

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found in model.")
    optimizer_summary = [
        {
            "index": 0,
            "name": "default",
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "num_parameters": int(sum(parameter.numel() for parameter in params)),
        }
    ]
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay), optimizer_summary


def train_one(
    dataset_path: Path | None,
    output_dir: Path,
    model_family: str = MODEL_FAMILY,
    dataset_paths: list[Path] | None = None,
    combined_train_config: str | None = None,
    combined_budget: int | None = None,
    combined_seed: int | None = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    backbone_lr: float | None = None,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    train_seed: int = 0,
    device_name: str = "auto",
    max_samples: int | None = None,
    max_samples_per_dataset: int | None = None,
    val_fraction: float = 0.0,
    phase_balance: bool = False,
    spatial_phase_balance: bool = False,
    spatial_bins_x: int = 4,
    spatial_bins_y: int = 4,
    spatial_x_range: tuple[float, float] = DEFAULT_SPATIAL_X_RANGE,
    spatial_y_range: tuple[float, float] = DEFAULT_SPATIAL_Y_RANGE,
    checkpoint_every: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    configure_torch_threads()
    set_seed(train_seed)
    output_dir = Path(output_dir)
    if dataset_paths is not None and len(dataset_paths) > 0:
        if combined_train_config is None:
            combined_train_config = "combined_existing"
        if combined_budget is None:
            combined_budget = len(dataset_paths)
        if combined_seed is None:
            combined_seed = train_seed
        dataset = CombinedResearch2BCDataset(
            dataset_paths=[Path(path) for path in dataset_paths],
            combined_train_config=combined_train_config,
            combined_budget=combined_budget,
            combined_seed=combined_seed,
            max_samples_per_dataset=max_samples_per_dataset,
            max_samples=max_samples,
        )
        dataset_path_for_summary: str | list[str] = [str(path) for path in dataset.dataset_paths]
    else:
        if dataset_path is None:
            raise ValueError("Pass dataset_path or dataset_paths.")
        dataset_path = Path(dataset_path)
        dataset = Research2BCDataset(dataset_path, max_samples=max_samples)
        dataset_path_for_summary = str(dataset_path)
    metadata = dataset.payload_metadata
    config_name = str(metadata["train_config"])
    budget = int(metadata["budget"])
    data_seed = int(metadata["seed"])
    stem = model_stem(model_family, config_name, budget, data_seed)
    model_path = output_dir / "models" / f"{stem}.pt"
    history_path = output_dir / "histories" / f"{stem}.json"
    if history_path.exists() and model_path.exists() and not overwrite:
        return json.loads(history_path.read_text(encoding="utf-8"))

    device = select_device(device_name)
    model = build_policy(model_family).to(device)
    parameter_counts = count_parameters(model)
    optimizer, optimizer_summary = build_optimizer(model, lr=lr, weight_decay=weight_decay, backbone_lr=backbone_lr)

    train_indices, val_indices = split_indices(len(dataset), val_fraction=val_fraction, seed=train_seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    phase_balance_sampler = None
    phase_balance_summary: dict[str, Any] = {
        "requested": bool(phase_balance),
        "enabled": False,
        "train_phase_counts": {},
        "reason": "not_requested",
    }
    if phase_balance:
        phase_balance_sampler, phase_balance_summary = build_phase_balanced_sampler(dataset, train_indices)
        phase_balance_summary["requested"] = True
    spatial_phase_balance_summary: dict[str, Any] = {
        "requested": bool(spatial_phase_balance),
        "enabled": False,
        "reason": "not_requested",
    }
    if spatial_phase_balance:
        phase_balance_sampler, spatial_phase_balance_summary = build_spatial_phase_balanced_sampler(
            dataset,
            train_indices,
            bins_x=int(spatial_bins_x),
            bins_y=int(spatial_bins_y),
            x_range=spatial_x_range,
            y_range=spatial_y_range,
        )
        spatial_phase_balance_summary["requested"] = True
        phase_balance_summary["suppressed_by_spatial_phase_balance"] = bool(phase_balance)
    train_loader = make_loader(
        train_dataset,
        batch_size,
        shuffle=phase_balance_sampler is None,
        num_workers=num_workers,
        seed=train_seed,
        device=device,
        sampler=phase_balance_sampler,
    )
    val_loader = (
        make_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, seed=train_seed, device=device)
        if val_dataset is not None
        else None
    )

    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "model_family": model_family,
        "dataset_path": dataset_path_for_summary,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "output_dir": str(output_dir),
        "dataset_metadata": metadata,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "backbone_lr": None if backbone_lr is None else float(backbone_lr),
        "weight_decay": float(weight_decay),
        "optimizer": {"type": "Adam", "parameter_groups": optimizer_summary},
        "num_workers": int(num_workers),
        "train_seed": int(train_seed),
        "device": str(device),
        "max_samples": max_samples,
        "max_samples_per_dataset": max_samples_per_dataset,
        "val_fraction": float(val_fraction),
        "num_train_samples": len(train_indices),
        "num_val_samples": len(val_indices),
        "parameter_counts": parameter_counts,
        "dataset_phase_counts": dataset.phase_counts,
        "dataset_source_counts": getattr(dataset, "source_counts", {}),
        "phase_labels": list(AVOID_PHASE_LABELS),
        "phase_balance": phase_balance_summary,
        "spatial_phase_balance": spatial_phase_balance_summary,
        "history": [],
    }

    for epoch_idx in tqdm(range(int(epochs)), desc=f"train {config_name} {format_budget_tag(budget)} {format_seed_tag(data_seed)}", unit="epoch"):
        train_loss = run_epoch(model, train_loader, device=device, optimizer=optimizer)
        epoch_record: dict[str, Any] = {
            "epoch": epoch_idx + 1,
            "train_loss": train_loss,
        }
        if val_loader is not None:
            with torch.no_grad():
                epoch_record["val_loss"] = run_epoch(model, val_loader, device=device, optimizer=None)
        summary["history"].append(epoch_record)
        summary["final_train_loss"] = train_loss
        if "val_loss" in epoch_record:
            summary["final_val_loss"] = epoch_record["val_loss"]
        write_json(history_path, summary)

        if checkpoint_every and (epoch_idx + 1) % int(checkpoint_every) == 0:
            checkpoint_path = output_dir / "checkpoints" / f"{stem}__epoch{epoch_idx + 1:03d}.pt"
            save_policy_checkpoint(
                checkpoint_path,
                model,
                model_family,
                metadata={**summary, "checkpoint_epoch": epoch_idx + 1},
            )

    save_policy_checkpoint(model_path, model, model_family, metadata=summary)
    summary["completed_at"] = utc_now()
    write_json(history_path, summary)
    return summary


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path is not None:
        return Path(args.dataset_path)
    if args.train_config is None or args.budget is None or args.seed is None:
        raise ValueError("Pass --dataset-path or pass --train-config, --budget, and --seed.")
    return Path(args.dataset_dir) / f"{dataset_stem(args.train_config, args.budget, args.seed)}.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-paths", type=Path, nargs="+")
    parser.add_argument("--dataset-dir", type=Path, default=Path("results/datasets_128px_v1"))
    parser.add_argument("--train-config")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--combined-train-config", default="combined_existing")
    parser.add_argument("--combined-budget", type=int)
    parser.add_argument("--combined-seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/bc_128px_v1"))
    parser.add_argument("--model-family", choices=list_model_families(), default=MODEL_FAMILY)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-samples-per-dataset", type=int)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--phase-balance", action="store_true")
    parser.add_argument("--spatial-phase-balance", action="store_true")
    parser.add_argument("--spatial-bins-x", type=int, default=4)
    parser.add_argument("--spatial-bins-y", type=int, default=4)
    parser.add_argument("--spatial-x-range", type=float, nargs=2, default=DEFAULT_SPATIAL_X_RANGE)
    parser.add_argument("--spatial-y-range", type=float, nargs=2, default=DEFAULT_SPATIAL_Y_RANGE)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_one(
        dataset_path=None if args.dataset_paths else resolve_dataset_path(args),
        output_dir=args.output_dir,
        model_family=args.model_family,
        dataset_paths=args.dataset_paths,
        combined_train_config=args.combined_train_config,
        combined_budget=args.combined_budget,
        combined_seed=args.combined_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        train_seed=args.train_seed,
        device_name=args.device,
        max_samples=args.max_samples,
        max_samples_per_dataset=args.max_samples_per_dataset,
        val_fraction=args.val_fraction,
        phase_balance=args.phase_balance,
        spatial_phase_balance=args.spatial_phase_balance,
        spatial_bins_x=args.spatial_bins_x,
        spatial_bins_y=args.spatial_bins_y,
        spatial_x_range=tuple(args.spatial_x_range),
        spatial_y_range=tuple(args.spatial_y_range),
        checkpoint_every=args.checkpoint_every,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: summary.get(key) for key in ("model_path", "history_path", "final_train_loss", "final_val_loss")}, indent=2))


if __name__ == "__main__":
    main()
