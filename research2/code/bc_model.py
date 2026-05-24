#!/usr/bin/env python3
"""128px behavior-cloning policies for research2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


MODEL_FAMILY = "scratch_bc_128"
COORDCONV_MODEL_FAMILY = "scratch_coordconv_bc_128"
SPATIAL_SOFTMAX_MODEL_FAMILY = "scratch_spatial_softmax_bc_128"
FROZEN_RESNET18_MODEL_FAMILY = "frozen_resnet18_bc_128"
PARTIAL_RESNET18_MODEL_FAMILY = "partial_resnet18_bc_128"
PARTIAL_RESNET18_L4_BNFREEZE_MODEL_FAMILY = "partial_resnet18_l4_bnfreeze_bc_128"
PARTIAL_RESNET18_SEP_L4_PHASEAUX_MODEL_FAMILY = "partial_resnet18_sep_l4_phaseaux_bc_128"
STRUCTURED_SCRATCH_MODEL_FAMILY = "scratch_structured_bc_128"
STRUCTURED_FROZEN_RESNET18_MODEL_FAMILY = "frozen_resnet18_structured_bc_128"
STRUCTURED_PARTIAL_RESNET18_MODEL_FAMILY = "partial_resnet18_structured_bc_128"
SCRATCH_PHASE_ONLY_MODEL_FAMILY = "scratch_phase_only_bc_128"
SCRATCH_TARGET_ONLY_MODEL_FAMILY = "scratch_target_only_bc_128"
SCRATCH_GEOMETRY_ONLY_MODEL_FAMILY = "scratch_geometry_only_bc_128"
STRUCTURED_STATE_DIM = 18
SUPPORTED_MODEL_FAMILIES = (
    MODEL_FAMILY,
    COORDCONV_MODEL_FAMILY,
    SPATIAL_SOFTMAX_MODEL_FAMILY,
    FROZEN_RESNET18_MODEL_FAMILY,
    PARTIAL_RESNET18_MODEL_FAMILY,
    PARTIAL_RESNET18_L4_BNFREEZE_MODEL_FAMILY,
    PARTIAL_RESNET18_SEP_L4_PHASEAUX_MODEL_FAMILY,
    STRUCTURED_SCRATCH_MODEL_FAMILY,
    STRUCTURED_FROZEN_RESNET18_MODEL_FAMILY,
    STRUCTURED_PARTIAL_RESNET18_MODEL_FAMILY,
    SCRATCH_PHASE_ONLY_MODEL_FAMILY,
    SCRATCH_TARGET_ONLY_MODEL_FAMILY,
    SCRATCH_GEOMETRY_ONLY_MODEL_FAMILY,
)
AVOID_PHASE_LABELS = ("side_align", "cube_hover", "final_descent")


def list_model_families() -> tuple[str, ...]:
    return SUPPORTED_MODEL_FAMILIES


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def freeze_batchnorm_layers(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            freeze_module(child)


def resolve_model_family(explicit_model_family: str | None = None, checkpoint_payload: dict[str, Any] | None = None) -> str:
    if explicit_model_family is not None:
        return explicit_model_family
    if isinstance(checkpoint_payload, dict) and checkpoint_payload.get("model_family"):
        return str(checkpoint_payload["model_family"])
    return MODEL_FAMILY


def structured_feature_mask(mode: str, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros(STRUCTURED_STATE_DIM, dtype=torch.float32, device=device)
    if mode == "phase":
        mask[0:3] = 1.0
    elif mode == "target":
        mask[3:6] = 1.0
        mask[12:15] = 1.0
    elif mode == "geometry":
        mask[3:18] = 1.0
    elif mode in ("full", "phase_geometry"):
        mask[:] = 1.0
    else:
        raise ValueError(f"Unsupported structured feature mode: {mode}")
    return mask


class Scratch128Encoder(nn.Module):
    def __init__(self, emb_dim: int = 64, input_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, emb_dim),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float() / 255.0
        return self.proj(self.conv(image))


class CoordConvScratch128Encoder(Scratch128Encoder):
    def __init__(self, emb_dim: int = 64):
        super().__init__(emb_dim=emb_dim, input_channels=5)

    @staticmethod
    def coordinate_channels(image: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = image.shape
        y = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype).view(1, 1, height, 1)
        x = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype).view(1, 1, 1, width)
        y = y.expand(batch_size, 1, height, width)
        x = x.expand(batch_size, 1, height, width)
        return torch.cat((x, y), dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float() / 255.0
        image = torch.cat((image, self.coordinate_channels(image)), dim=1)
        return self.proj(self.conv(image))


class ScratchBC128Policy(nn.Module):
    encoder_cls = Scratch128Encoder

    def __init__(self, state_dim: int = 3, action_dim: int = 3, emb_dim: int = 64):
        super().__init__()
        self.external_encoder = self.encoder_cls(emb_dim=emb_dim)
        self.eef_encoder = self.encoder_cls(emb_dim=emb_dim)
        self.policy = nn.Sequential(
            nn.Linear(state_dim + 2 * emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, external_rgb: torch.Tensor, eef_rgb: torch.Tensor, ee_position: torch.Tensor) -> torch.Tensor:
        z_external = self.external_encoder(external_rgb)
        z_eef = self.eef_encoder(eef_rgb)
        features = torch.cat((z_external, z_eef, ee_position.float()), dim=-1)
        return self.policy(features)


class CoordConvScratchBC128Policy(ScratchBC128Policy):
    encoder_cls = CoordConvScratch128Encoder


class StructuredScratchBC128Policy(nn.Module):
    encoder_cls = Scratch128Encoder
    uses_structured_state = True

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 3,
        emb_dim: int = 64,
        structured_state_dim: int = STRUCTURED_STATE_DIM,
        structured_feature_mode: str = "full",
    ):
        super().__init__()
        self.structured_state_dim = int(structured_state_dim)
        self.structured_feature_mode = str(structured_feature_mode)
        self.register_buffer(
            "structured_state_mask",
            structured_feature_mask(self.structured_feature_mode),
            persistent=False,
        )
        self.external_encoder = self.encoder_cls(emb_dim=emb_dim)
        self.eef_encoder = self.encoder_cls(emb_dim=emb_dim)
        self.policy = nn.Sequential(
            nn.Linear(state_dim + self.structured_state_dim + 2 * emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.mse_loss = nn.MSELoss()

    def empty_structured_state(self, ee_position: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (ee_position.shape[0], self.structured_state_dim),
            dtype=ee_position.dtype,
            device=ee_position.device,
        )

    def forward(
        self,
        external_rgb: torch.Tensor,
        eef_rgb: torch.Tensor,
        ee_position: torch.Tensor,
        structured_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_external = self.external_encoder(external_rgb)
        z_eef = self.eef_encoder(eef_rgb)
        if structured_state is None:
            structured_state = self.empty_structured_state(ee_position)
        structured_state = structured_state.float() * self.structured_state_mask.to(structured_state.device)
        features = torch.cat((z_external, z_eef, ee_position.float(), structured_state.float()), dim=-1)
        return self.policy(features)


class SpatialSoftmax(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, height * width), persistent=False)
        self.register_buffer("pos_y", pos_y.reshape(1, 1, height * width), persistent=False)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature_map.shape
        attention = feature_map.reshape(batch_size, channels, height * width)
        attention = torch.softmax(attention, dim=-1)
        expected_x = torch.sum(attention * self.pos_x, dim=-1)
        expected_y = torch.sum(attention * self.pos_y, dim=-1)
        return torch.cat((expected_x, expected_y), dim=-1)


class SpatialSoftmaxScratch128Encoder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.spatial_softmax = SpatialSoftmax(height=6, width=6)
        self.proj = nn.Sequential(
            nn.Linear(256, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float() / 255.0
        return self.proj(self.spatial_softmax(self.conv(image)))


class SpatialSoftmaxScratchBC128Policy(nn.Module):
    def __init__(self, state_dim: int = 3, action_dim: int = 3, emb_dim: int = 128):
        super().__init__()
        self.external_encoder = SpatialSoftmaxScratch128Encoder(emb_dim=emb_dim)
        self.eef_encoder = SpatialSoftmaxScratch128Encoder(emb_dim=emb_dim)
        self.policy = nn.Sequential(
            nn.Linear(state_dim + 2 * emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, external_rgb: torch.Tensor, eef_rgb: torch.Tensor, ee_position: torch.Tensor) -> torch.Tensor:
        z_external = self.external_encoder(external_rgb)
        z_eef = self.eef_encoder(eef_rgb)
        features = torch.cat((z_external, z_eef, ee_position.float()), dim=-1)
        return self.policy(features)


class FrozenResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:
            raise RuntimeError("frozen_resnet18_bc_128 requires torchvision.") from exc

        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.features.eval()
        mean = torch.tensor(weights.transforms().mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(weights.transforms().std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def train(self, mode: bool = True) -> "FrozenResNet18Backbone":
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float() / 255.0
        image = (image - self.mean) / self.std
        with torch.no_grad():
            return self.features(image).flatten(start_dim=1)


class PartialResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:
            raise RuntimeError("partial_resnet18_bc_128 requires torchvision.") from exc

        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = resnet18(weights=weights)
        self.frozen_features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        self.trainable_features = nn.Sequential(
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        for parameter in self.frozen_features.parameters():
            parameter.requires_grad_(False)
        self.frozen_features.eval()
        mean = torch.tensor(weights.transforms().mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(weights.transforms().std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def train(self, mode: bool = True) -> "PartialResNet18Backbone":
        super().train(mode)
        self.frozen_features.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float() / 255.0
        image = (image - self.mean) / self.std
        with torch.no_grad():
            image = self.frozen_features(image)
        return self.trainable_features(image).flatten(start_dim=1)


class PartialResNet18Layer4BNFrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:
            raise RuntimeError("partial_resnet18_l4_bnfreeze_bc_128 requires torchvision.") from exc

        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = resnet18(weights=weights)
        self.frozen_features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.trainable_features = nn.Sequential(
            resnet.layer4,
            resnet.avgpool,
        )
        freeze_module(self.frozen_features)
        freeze_batchnorm_layers(self)
        self.frozen_features.eval()
        mean = torch.tensor(weights.transforms().mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(weights.transforms().std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def train(self, mode: bool = True) -> "PartialResNet18Layer4BNFrozenBackbone":
        super().train(mode)
        self.frozen_features.eval()
        freeze_batchnorm_layers(self)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float() / 255.0
        image = (image - self.mean) / self.std
        with torch.no_grad():
            image = self.frozen_features(image)
        return self.trainable_features(image).flatten(start_dim=1)


class FrozenResNet18BC128Policy(nn.Module):
    def __init__(self, state_dim: int = 3, action_dim: int = 3, emb_dim: int = 128):
        super().__init__()
        self.backbone = FrozenResNet18Backbone()
        self.external_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.eef_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.policy = nn.Sequential(
            nn.Linear(state_dim + 2 * emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, external_rgb: torch.Tensor, eef_rgb: torch.Tensor, ee_position: torch.Tensor) -> torch.Tensor:
        z_external = self.external_proj(self.backbone(external_rgb))
        z_eef = self.eef_proj(self.backbone(eef_rgb))
        features = torch.cat((z_external, z_eef, ee_position.float()), dim=-1)
        return self.policy(features)


class StructuredFrozenResNet18BC128Policy(nn.Module):
    uses_structured_state = True

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 3,
        emb_dim: int = 128,
        structured_state_dim: int = STRUCTURED_STATE_DIM,
    ):
        super().__init__()
        self.structured_state_dim = int(structured_state_dim)
        self.backbone = FrozenResNet18Backbone()
        self.external_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.eef_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.policy = nn.Sequential(
            nn.Linear(state_dim + self.structured_state_dim + 2 * emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.mse_loss = nn.MSELoss()

    def empty_structured_state(self, ee_position: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (ee_position.shape[0], self.structured_state_dim),
            dtype=ee_position.dtype,
            device=ee_position.device,
        )

    def forward(
        self,
        external_rgb: torch.Tensor,
        eef_rgb: torch.Tensor,
        ee_position: torch.Tensor,
        structured_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_external = self.external_proj(self.backbone(external_rgb))
        z_eef = self.eef_proj(self.backbone(eef_rgb))
        if structured_state is None:
            structured_state = self.empty_structured_state(ee_position)
        features = torch.cat((z_external, z_eef, ee_position.float(), structured_state.float()), dim=-1)
        return self.policy(features)


class PartialResNet18BC128Policy(nn.Module):
    def __init__(self, state_dim: int = 3, action_dim: int = 3, emb_dim: int = 128):
        super().__init__()
        self.backbone = PartialResNet18Backbone()
        self.external_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.eef_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.policy = nn.Sequential(
            nn.Linear(state_dim + 2 * emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, external_rgb: torch.Tensor, eef_rgb: torch.Tensor, ee_position: torch.Tensor) -> torch.Tensor:
        z_external = self.external_proj(self.backbone(external_rgb))
        z_eef = self.eef_proj(self.backbone(eef_rgb))
        features = torch.cat((z_external, z_eef, ee_position.float()), dim=-1)
        return self.policy(features)


class StructuredPartialResNet18BC128Policy(nn.Module):
    uses_structured_state = True

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 3,
        emb_dim: int = 128,
        structured_state_dim: int = STRUCTURED_STATE_DIM,
    ):
        super().__init__()
        self.structured_state_dim = int(structured_state_dim)
        self.backbone = PartialResNet18Backbone()
        self.external_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.eef_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.policy = nn.Sequential(
            nn.Linear(state_dim + self.structured_state_dim + 2 * emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.mse_loss = nn.MSELoss()

    def empty_structured_state(self, ee_position: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (ee_position.shape[0], self.structured_state_dim),
            dtype=ee_position.dtype,
            device=ee_position.device,
        )

    def forward(
        self,
        external_rgb: torch.Tensor,
        eef_rgb: torch.Tensor,
        ee_position: torch.Tensor,
        structured_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_external = self.external_proj(self.backbone(external_rgb))
        z_eef = self.eef_proj(self.backbone(eef_rgb))
        if structured_state is None:
            structured_state = self.empty_structured_state(ee_position)
        features = torch.cat((z_external, z_eef, ee_position.float(), structured_state.float()), dim=-1)
        return self.policy(features)


class PartialResNet18Layer4BNFrozenBC128Policy(nn.Module):
    def __init__(self, state_dim: int = 3, action_dim: int = 3, emb_dim: int = 128):
        super().__init__()
        self.backbone = PartialResNet18Layer4BNFrozenBackbone()
        self.external_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.eef_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.policy = nn.Sequential(
            nn.Linear(state_dim + 2 * emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.mse_loss = nn.MSELoss()

    def optimizer_parameter_groups(
        self,
        lr: float,
        weight_decay: float,
        backbone_lr: float | None = None,
    ) -> list[dict[str, object]]:
        if backbone_lr is None:
            backbone_lr = lr * 0.03
        head_parameters = [
            parameter
            for module in (self.external_proj, self.eef_proj, self.policy)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        backbone_parameters = [
            parameter
            for parameter in self.backbone.trainable_features.parameters()
            if parameter.requires_grad
        ]
        return [
            {
                "name": "head",
                "params": head_parameters,
                "lr": lr,
                "weight_decay": weight_decay,
            },
            {
                "name": "layer4_backbone",
                "params": backbone_parameters,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            },
        ]

    def forward(self, external_rgb: torch.Tensor, eef_rgb: torch.Tensor, ee_position: torch.Tensor) -> torch.Tensor:
        z_external = self.external_proj(self.backbone(external_rgb))
        z_eef = self.eef_proj(self.backbone(eef_rgb))
        features = torch.cat((z_external, z_eef, ee_position.float()), dim=-1)
        return self.policy(features)


class PartialResNet18SeparateLayer4PhaseAuxBC128Policy(nn.Module):
    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 3,
        emb_dim: int = 128,
        phase_loss_weight: float = 0.001,
    ):
        super().__init__()
        self.phase_loss_weight = float(phase_loss_weight)
        self.phase_labels = AVOID_PHASE_LABELS
        self.external_backbone = PartialResNet18Layer4BNFrozenBackbone()
        self.eef_backbone = PartialResNet18Layer4BNFrozenBackbone()
        self.external_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        self.eef_proj = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )
        feature_dim = state_dim + 2 * emb_dim
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.phase_labels)),
        )
        self.mse_loss = nn.MSELoss()
        self.phase_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def optimizer_parameter_groups(
        self,
        lr: float,
        weight_decay: float,
        backbone_lr: float | None = None,
    ) -> list[dict[str, object]]:
        if backbone_lr is None:
            backbone_lr = lr * 0.03
        head_parameters = [
            parameter
            for module in (self.external_proj, self.eef_proj, self.policy, self.phase_head)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        external_backbone_parameters = [
            parameter
            for parameter in self.external_backbone.trainable_features.parameters()
            if parameter.requires_grad
        ]
        eef_backbone_parameters = [
            parameter
            for parameter in self.eef_backbone.trainable_features.parameters()
            if parameter.requires_grad
        ]
        return [
            {
                "name": "head",
                "params": head_parameters,
                "lr": lr,
                "weight_decay": weight_decay,
            },
            {
                "name": "external_layer4_backbone",
                "params": external_backbone_parameters,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            },
            {
                "name": "eef_layer4_backbone",
                "params": eef_backbone_parameters,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            },
        ]

    def forward(self, external_rgb: torch.Tensor, eef_rgb: torch.Tensor, ee_position: torch.Tensor) -> dict[str, torch.Tensor]:
        z_external = self.external_proj(self.external_backbone(external_rgb))
        z_eef = self.eef_proj(self.eef_backbone(eef_rgb))
        features = torch.cat((z_external, z_eef, ee_position.float()), dim=-1)
        return {
            "action": self.policy(features),
            "phase_logits": self.phase_head(features),
        }


def build_policy(model_family: str = MODEL_FAMILY, state_dim: int = 3, action_dim: int = 3) -> nn.Module:
    if model_family == MODEL_FAMILY:
        return ScratchBC128Policy(state_dim=state_dim, action_dim=action_dim)
    if model_family == COORDCONV_MODEL_FAMILY:
        return CoordConvScratchBC128Policy(state_dim=state_dim, action_dim=action_dim)
    if model_family == SPATIAL_SOFTMAX_MODEL_FAMILY:
        return SpatialSoftmaxScratchBC128Policy(state_dim=state_dim, action_dim=action_dim)
    if model_family == FROZEN_RESNET18_MODEL_FAMILY:
        return FrozenResNet18BC128Policy(state_dim=state_dim, action_dim=action_dim)
    if model_family == PARTIAL_RESNET18_MODEL_FAMILY:
        return PartialResNet18BC128Policy(state_dim=state_dim, action_dim=action_dim)
    if model_family == PARTIAL_RESNET18_L4_BNFREEZE_MODEL_FAMILY:
        return PartialResNet18Layer4BNFrozenBC128Policy(state_dim=state_dim, action_dim=action_dim)
    if model_family == PARTIAL_RESNET18_SEP_L4_PHASEAUX_MODEL_FAMILY:
        return PartialResNet18SeparateLayer4PhaseAuxBC128Policy(state_dim=state_dim, action_dim=action_dim)
    if model_family == STRUCTURED_SCRATCH_MODEL_FAMILY:
        return StructuredScratchBC128Policy(state_dim=state_dim, action_dim=action_dim)
    if model_family == STRUCTURED_FROZEN_RESNET18_MODEL_FAMILY:
        return StructuredFrozenResNet18BC128Policy(state_dim=state_dim, action_dim=action_dim)
    if model_family == STRUCTURED_PARTIAL_RESNET18_MODEL_FAMILY:
        return StructuredPartialResNet18BC128Policy(state_dim=state_dim, action_dim=action_dim)
    if model_family == SCRATCH_PHASE_ONLY_MODEL_FAMILY:
        return StructuredScratchBC128Policy(
            state_dim=state_dim,
            action_dim=action_dim,
            structured_feature_mode="phase",
        )
    if model_family == SCRATCH_TARGET_ONLY_MODEL_FAMILY:
        return StructuredScratchBC128Policy(
            state_dim=state_dim,
            action_dim=action_dim,
            structured_feature_mode="target",
        )
    if model_family == SCRATCH_GEOMETRY_ONLY_MODEL_FAMILY:
        return StructuredScratchBC128Policy(
            state_dim=state_dim,
            action_dim=action_dim,
            structured_feature_mode="geometry",
        )
    raise ValueError(f"Unsupported model family for research2: {model_family}")


def extract_action_prediction(model_output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(model_output, dict):
        return model_output["action"]
    return model_output


def save_policy_checkpoint(path: str | Path, model: nn.Module, model_family: str, metadata: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "model_family": model_family,
        "state_dict": model.state_dict(),
    }
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint_payload(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {
        "model_family": MODEL_FAMILY,
        "state_dict": checkpoint,
        "metadata": {},
    }
