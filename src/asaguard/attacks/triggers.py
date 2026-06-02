"""Visual trigger utilities for attack training and evaluation.

These functions transform images only. They never change labels and they do
not implement any training-time attack.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Literal

import torch
import torch.nn.functional as F

from asaguard.data.datasets import DATASET_NORMALIZATION


Bounds = tuple[tuple[float, ...], tuple[float, ...]]


@dataclass(frozen=True)
class TriggerSpec:
    trigger_type: str = "patch"
    trigger_size: int = 4
    trigger_value: float = 1.0
    trigger_alpha: float = 0.2
    trigger_location: str | tuple[int, int] = "bottom_right"
    input_mean: tuple[float, ...] | None = None
    input_std: tuple[float, ...] | None = None
    normalized_input_bounds: Bounds = (
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    )


def _raw_bounds(channels: int = 3) -> Bounds:
    return (
        tuple(0.0 for _ in range(int(channels))),
        tuple(1.0 for _ in range(int(channels))),
    )


def _bounds_from_normalization(
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Bounds:
    lower = tuple((0.0 - float(mu)) / float(sigma) for mu, sigma in zip(mean, std))
    upper = tuple((1.0 - float(mu)) / float(sigma) for mu, sigma in zip(mean, std))
    return lower, upper


def build_trigger_from_config(
    attack_cfg: dict,
    data_cfg: dict,
) -> TriggerSpec:
    """Build one trigger spec shared by train-time poisoning and ASR evaluation."""
    dataset_name = str(data_cfg.get("name", "")).lower()
    normalization = DATASET_NORMALIZATION.get(dataset_name)
    if normalization is None:
        input_mean = None
        input_std = None
        bounds = _raw_bounds()
    else:
        input_mean = tuple(float(value) for value in normalization[0])
        input_std = tuple(float(value) for value in normalization[1])
        bounds = _bounds_from_normalization(input_mean, input_std)

    return TriggerSpec(
        trigger_type=str(attack_cfg.get("trigger_type", "patch")).lower(),
        trigger_size=int(attack_cfg.get("trigger_size", 4)),
        # Configured trigger_value is a raw pixel value; convert it to the
        # model input space when Normalize(mean, std) is part of preprocessing.
        trigger_value=float(attack_cfg.get("trigger_value", 1.0)),
        trigger_alpha=float(attack_cfg.get("trigger_alpha", 0.2)),
        trigger_location=str(attack_cfg.get("trigger_location", "bottom_right")),
        input_mean=input_mean,
        input_std=input_std,
        normalized_input_bounds=bounds,
    )


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError("Expected image tensor with shape CxHxW or batch NxCxHxW")


def _restore_shape(images: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return images.squeeze(0) if squeezed else images


def _coerce_bounds(
    images: torch.Tensor,
    spec: TriggerSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _ = _as_batch(images)
    channels = int(batch.shape[1])
    lower_values, upper_values = spec.normalized_input_bounds
    if len(lower_values) != channels or len(upper_values) != channels:
        raise ValueError("Trigger input bounds channel count must match images")
    lower = torch.tensor(lower_values, device=images.device, dtype=images.dtype).view(1, channels, 1, 1)
    upper = torch.tensor(upper_values, device=images.device, dtype=images.dtype).view(1, channels, 1, 1)
    return lower, upper


def _clamp_to_spec(images: torch.Tensor, spec: TriggerSpec) -> torch.Tensor:
    lower, upper = _coerce_bounds(images, spec)
    return torch.maximum(torch.minimum(images, upper), lower)


def _trigger_value_tensor(
    images: torch.Tensor,
    spec: TriggerSpec,
) -> torch.Tensor:
    batch, _ = _as_batch(images)
    channels = int(batch.shape[1])
    value = torch.full((channels,), float(spec.trigger_value), device=images.device, dtype=images.dtype)
    if spec.input_mean is not None and spec.input_std is not None:
        mean = torch.tensor(spec.input_mean, device=images.device, dtype=images.dtype)
        std = torch.tensor(spec.input_std, device=images.device, dtype=images.dtype)
        value = (value - mean) / std
    lower, upper = _coerce_bounds(images, spec)
    value = value.view(1, channels, 1, 1)
    return torch.maximum(torch.minimum(value, upper), lower)


def _legacy_spec(
    *,
    trigger_type: str,
    size: int = 4,
    value: float = 1.0,
    alpha: float = 0.15,
    location: str | tuple[int, int] = "bottom_right",
) -> TriggerSpec:
    return TriggerSpec(
        trigger_type=trigger_type,
        trigger_size=int(size),
        trigger_value=float(value),
        trigger_alpha=float(alpha),
        trigger_location=location,
        normalized_input_bounds=_raw_bounds(),
    )


def patch_trigger(
    images: torch.Tensor,
    *,
    size: int = 4,
    value: float = 1.0,
    location: Literal["bottom_right", "top_left"] | tuple[int, int] = "bottom_right",
    spec: TriggerSpec | None = None,
) -> torch.Tensor:
    """Apply a square patch trigger to an image or image batch."""
    if spec is None:
        spec = _legacy_spec(trigger_type="patch", size=size, value=value, location=location)
    batch, squeezed = _as_batch(images)
    output = batch.clone()
    _, _, height, width = output.shape
    size = max(1, min(int(spec.trigger_size), height, width))
    location = spec.trigger_location

    if location == "bottom_right":
        top = height - size
        left = width - size
    elif location == "top_left":
        top = 0
        left = 0
    else:
        top = max(0, min(int(location[0]), height - size))
        left = max(0, min(int(location[1]), width - size))

    output[:, :, top : top + size, left : left + size] = _trigger_value_tensor(output, spec)
    return _restore_shape(_clamp_to_spec(output, spec), squeezed)


def blend_trigger(
    images: torch.Tensor,
    *,
    alpha: float = 0.15,
    pattern_value: float = 1.0,
    spec: TriggerSpec | None = None,
) -> torch.Tensor:
    """Apply a low-alpha full-image blend trigger."""
    if spec is None:
        spec = _legacy_spec(trigger_type="blend", value=pattern_value, alpha=alpha)
    batch, squeezed = _as_batch(images)
    alpha = max(0.0, min(float(spec.trigger_alpha), 1.0))
    pattern = _trigger_value_tensor(batch, spec).expand_as(batch)
    output = (1.0 - alpha) * batch + alpha * pattern
    return _restore_shape(_clamp_to_spec(output, spec), squeezed)


def frequency_trigger(
    images: torch.Tensor,
    *,
    amplitude: float = 0.08,
    frequency: float = 6.0,
    phase: float = 0.0,
    spec: TriggerSpec | None = None,
) -> torch.Tensor:
    """Apply a sine-wave frequency trigger."""
    if spec is None:
        spec = _legacy_spec(trigger_type="frequency", value=amplitude)
    batch, squeezed = _as_batch(images)
    _, _, height, width = batch.shape
    xs = torch.linspace(0.0, 2.0 * pi, width, device=batch.device, dtype=batch.dtype)
    ys = torch.linspace(0.0, 2.0 * pi, height, device=batch.device, dtype=batch.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    sine = torch.sin(float(frequency) * (grid_x + grid_y) + float(phase))
    sine = sine.view(1, 1, height, width)
    output = batch + float(spec.trigger_value) * sine
    return _restore_shape(_clamp_to_spec(output, spec), squeezed)


def warping_trigger(
    images: torch.Tensor,
    *,
    strength: float = 0.12,
    frequency: float = 2.0,
    spec: TriggerSpec | None = None,
) -> torch.Tensor:
    """Apply a lightweight geometric warping trigger."""
    if spec is None:
        spec = _legacy_spec(trigger_type="warping", value=strength)
    batch, squeezed = _as_batch(images)
    num_images, _, height, width = batch.shape
    ys = torch.linspace(-1.0, 1.0, height, device=batch.device, dtype=batch.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=batch.device, dtype=batch.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    offset_x = float(spec.trigger_value) * torch.sin(float(frequency) * pi * grid_y)
    offset_y = float(spec.trigger_value) * torch.sin(float(frequency) * pi * grid_x)
    grid = torch.stack((grid_x + offset_x, grid_y + offset_y), dim=-1)
    grid = grid.unsqueeze(0).repeat(num_images, 1, 1, 1)
    output = F.grid_sample(
        batch,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )
    return _restore_shape(_clamp_to_spec(output, spec), squeezed)


def apply_configured_trigger(images: torch.Tensor, spec: TriggerSpec) -> torch.Tensor:
    trigger_type = spec.trigger_type.lower()
    if trigger_type == "patch":
        return patch_trigger(images, spec=spec)
    if trigger_type in {"blend", "low_alpha", "low-alpha"}:
        return blend_trigger(images, spec=spec)
    if trigger_type in {"frequency", "sine"}:
        return frequency_trigger(images, spec=spec)
    if trigger_type in {"warping", "warp"}:
        return warping_trigger(images, spec=spec)
    raise ValueError(f"Unsupported trigger_type: {spec.trigger_type}")


def apply_patch_trigger(images: torch.Tensor, **kwargs) -> torch.Tensor:
    return patch_trigger(images, **kwargs)


def apply_blend_trigger(images: torch.Tensor, **kwargs) -> torch.Tensor:
    return blend_trigger(images, **kwargs)


def apply_frequency_trigger(images: torch.Tensor, **kwargs) -> torch.Tensor:
    return frequency_trigger(images, **kwargs)


def apply_warping_trigger(images: torch.Tensor, **kwargs) -> torch.Tensor:
    return warping_trigger(images, **kwargs)
