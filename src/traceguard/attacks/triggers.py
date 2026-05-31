"""Visual trigger utilities for evaluation.

These functions transform images only. They never change labels and they do
not implement any training-time attack.
"""

from __future__ import annotations

from math import pi
from typing import Literal

import torch
import torch.nn.functional as F


ImageRange = tuple[float, float]


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError("Expected image tensor with shape CxHxW or batch NxCxHxW")


def _restore_shape(images: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return images.squeeze(0) if squeezed else images


def _clamp(images: torch.Tensor, value_range: ImageRange) -> torch.Tensor:
    value_min, value_max = value_range
    return images.clamp(min=value_min, max=value_max)


def patch_trigger(
    images: torch.Tensor,
    *,
    size: int = 4,
    value: float = 1.0,
    location: Literal["bottom_right", "top_left"] | tuple[int, int] = "bottom_right",
    value_range: ImageRange = (0.0, 1.0),
) -> torch.Tensor:
    """Apply a square patch trigger to an image or image batch."""
    batch, squeezed = _as_batch(images)
    output = batch.clone()
    _, _, height, width = output.shape
    size = max(1, min(int(size), height, width))

    if location == "bottom_right":
        top = height - size
        left = width - size
    elif location == "top_left":
        top = 0
        left = 0
    else:
        top = max(0, min(int(location[0]), height - size))
        left = max(0, min(int(location[1]), width - size))

    output[:, :, top : top + size, left : left + size] = value
    return _restore_shape(_clamp(output, value_range), squeezed)


def blend_trigger(
    images: torch.Tensor,
    *,
    alpha: float = 0.15,
    pattern_value: float = 1.0,
    value_range: ImageRange = (0.0, 1.0),
) -> torch.Tensor:
    """Apply a low-alpha full-image blend trigger."""
    batch, squeezed = _as_batch(images)
    alpha = max(0.0, min(float(alpha), 1.0))
    pattern = torch.full_like(batch, float(pattern_value))
    output = (1.0 - alpha) * batch + alpha * pattern
    return _restore_shape(_clamp(output, value_range), squeezed)


def frequency_trigger(
    images: torch.Tensor,
    *,
    amplitude: float = 0.08,
    frequency: float = 6.0,
    phase: float = 0.0,
    value_range: ImageRange = (0.0, 1.0),
) -> torch.Tensor:
    """Apply a sine-wave frequency trigger."""
    batch, squeezed = _as_batch(images)
    _, _, height, width = batch.shape
    xs = torch.linspace(0.0, 2.0 * pi, width, device=batch.device, dtype=batch.dtype)
    ys = torch.linspace(0.0, 2.0 * pi, height, device=batch.device, dtype=batch.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    sine = torch.sin(float(frequency) * (grid_x + grid_y) + float(phase))
    sine = sine.view(1, 1, height, width)
    output = batch + float(amplitude) * sine
    return _restore_shape(_clamp(output, value_range), squeezed)


def warping_trigger(
    images: torch.Tensor,
    *,
    strength: float = 0.12,
    frequency: float = 2.0,
    value_range: ImageRange = (0.0, 1.0),
) -> torch.Tensor:
    """Apply a lightweight geometric warping trigger."""
    batch, squeezed = _as_batch(images)
    num_images, _, height, width = batch.shape
    ys = torch.linspace(-1.0, 1.0, height, device=batch.device, dtype=batch.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=batch.device, dtype=batch.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    offset_x = float(strength) * torch.sin(float(frequency) * pi * grid_y)
    offset_y = float(strength) * torch.sin(float(frequency) * pi * grid_x)
    grid = torch.stack((grid_x + offset_x, grid_y + offset_y), dim=-1)
    grid = grid.unsqueeze(0).repeat(num_images, 1, 1, 1)
    output = F.grid_sample(
        batch,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )
    return _restore_shape(_clamp(output, value_range), squeezed)


def apply_patch_trigger(images: torch.Tensor, **kwargs) -> torch.Tensor:
    return patch_trigger(images, **kwargs)


def apply_blend_trigger(images: torch.Tensor, **kwargs) -> torch.Tensor:
    return blend_trigger(images, **kwargs)


def apply_frequency_trigger(images: torch.Tensor, **kwargs) -> torch.Tensor:
    return frequency_trigger(images, **kwargs)


def apply_warping_trigger(images: torch.Tensor, **kwargs) -> torch.Tensor:
    return warping_trigger(images, **kwargs)
