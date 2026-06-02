"""Simplified FLIP-style client hardening baseline.

This is a faithful simplified baseline inspired by FLIP's reverse-trigger
hardening idea. It is not a claim of exact official FLIP reproduction.
Benign clients reverse-engineer a small trigger against the current global
model, then train on hardening samples whose labels remain the original clean
labels. Malicious clients are not forced to run this defense.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from asaguard.data.datasets import get_normalized_input_bounds


InputBounds = tuple[torch.Tensor, torch.Tensor]


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError("Expected image tensor with shape CxHxW or BxCxHxW")


def _restore_shape(images: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return images.squeeze(0) if squeezed else images


def _default_input_bounds(
    *,
    channels: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> InputBounds:
    lower = torch.zeros(int(channels), 1, 1, device=device, dtype=dtype)
    upper = torch.ones(int(channels), 1, 1, device=device, dtype=dtype)
    return lower, upper


def _coerce_input_bounds(
    input_bounds: InputBounds | None,
    *,
    channels: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> InputBounds:
    if input_bounds is None:
        return _default_input_bounds(channels=channels, device=device, dtype=dtype)

    lower, upper = input_bounds
    lower = lower.to(device=device, dtype=dtype)
    upper = upper.to(device=device, dtype=dtype)
    if lower.ndim == 1:
        lower = lower.view(-1, 1, 1)
    if upper.ndim == 1:
        upper = upper.view(-1, 1, 1)
    if int(lower.shape[0]) != int(channels) or int(upper.shape[0]) != int(channels):
        raise ValueError("FLIP input bounds channel count must match images")
    return lower, upper


def _clamp_to_input_bounds(tensor: torch.Tensor, input_bounds: InputBounds) -> torch.Tensor:
    lower, upper = input_bounds
    return torch.maximum(torch.minimum(tensor, upper), lower)


def initialize_reverse_trigger(
    *,
    channels: int = 3,
    trigger_size: int = 4,
    init_value: float = 0.5,
    device: torch.device | str = "cpu",
    input_bounds: InputBounds | None = None,
) -> torch.Tensor:
    """Initialize a learnable reverse-engineered trigger patch."""
    lower, upper = _coerce_input_bounds(
        input_bounds,
        channels=channels,
        device=device,
        dtype=torch.float32,
    )
    value = lower + float(init_value) * (upper - lower)
    return _clamp_to_input_bounds(
        value.expand(int(channels), int(trigger_size), int(trigger_size)).clone(),
        (lower, upper),
    )


def apply_reverse_trigger(
    images: torch.Tensor,
    trigger_pattern: torch.Tensor,
    *,
    trigger_alpha: float = 0.2,
    location: str = "bottom_right",
    input_bounds: InputBounds | None = None,
) -> torch.Tensor:
    """Blend a reverse-engineered patch into an image or image batch."""
    batch, squeezed = _as_batch(images)
    output = batch.clone()
    _, channels, height, width = output.shape
    bounds = _coerce_input_bounds(
        input_bounds,
        channels=channels,
        device=output.device,
        dtype=output.dtype,
    )
    pattern = trigger_pattern.to(device=output.device, dtype=output.dtype)
    if pattern.ndim != 3:
        raise ValueError("trigger_pattern must have shape CxHxW")
    if pattern.shape[0] != channels:
        raise ValueError("trigger_pattern channel count must match images")

    patch_h = min(int(pattern.shape[1]), height)
    patch_w = min(int(pattern.shape[2]), width)
    if location == "bottom_right":
        top = height - patch_h
        left = width - patch_w
    elif location == "top_left":
        top = 0
        left = 0
    else:
        raise ValueError(f"Unsupported reverse trigger location: {location}")

    alpha = max(0.0, min(float(trigger_alpha), 1.0))
    patch = pattern[:, :patch_h, :patch_w]
    region = output[:, :, top : top + patch_h, left : left + patch_w]
    output[:, :, top : top + patch_h, left : left + patch_w] = (
        (1.0 - alpha) * region + alpha * patch
    )
    return _restore_shape(_clamp_to_input_bounds(output, bounds), squeezed)


def optimize_reverse_trigger(
    *,
    global_model: nn.Module,
    dataloader: DataLoader,
    target_label: int,
    trigger_size: int = 4,
    trigger_alpha: float = 0.2,
    reverse_steps: int = 20,
    reverse_lr: float = 0.05,
    device: torch.device | str = "cpu",
    input_bounds: InputBounds | None = None,
) -> torch.Tensor:
    """Reverse-optimize a trigger that induces target-label predictions."""
    device = torch.device(device)
    # Use a private model copy for reverse-trigger optimization. Freezing the
    # server's live global model would make later client training losses lose
    # their parameter gradient graph.
    model = deepcopy(global_model).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trigger = initialize_reverse_trigger(
        channels=3,
        trigger_size=trigger_size,
        device=device,
        input_bounds=input_bounds,
    ).requires_grad_(True)
    bounds = _coerce_input_bounds(
        input_bounds,
        channels=int(trigger.shape[0]),
        device=device,
        dtype=trigger.dtype,
    )
    optimizer = torch.optim.Adam([trigger], lr=float(reverse_lr))
    criterion = nn.CrossEntropyLoss()
    batches = list(dataloader)
    if not batches:
        return trigger.detach().cpu()

    target_label = int(target_label)
    for step_idx in range(max(1, int(reverse_steps))):
        images, labels = batches[step_idx % len(batches)]
        mask = labels != target_label
        if not bool(mask.any()):
            continue
        images = images[mask].to(device)
        targets = torch.full(
            (images.shape[0],),
            target_label,
            dtype=torch.long,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        triggered = apply_reverse_trigger(
            images,
            trigger,
            trigger_alpha=trigger_alpha,
            input_bounds=bounds,
        )
        loss = criterion(model(triggered), targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            trigger.copy_(_clamp_to_input_bounds(trigger, bounds))

    return _clamp_to_input_bounds(trigger.detach(), bounds).cpu()


@dataclass
class FLIPClientHardening:
    """Batch-level hardening used only by benign clients."""

    trigger_pattern: torch.Tensor
    input_bounds: InputBounds | None = None
    trigger_alpha: float = 0.2
    hardening_ratio: float = 0.5

    def make_hardening_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = max(0.0, min(float(self.hardening_ratio), 1.0))
        count = int(round(images.shape[0] * ratio))
        if count <= 0:
            return images[:0], labels[:0]
        hard_images = apply_reverse_trigger(
            images[:count],
            self.trigger_pattern,
            trigger_alpha=self.trigger_alpha,
            input_bounds=self.input_bounds,
        )
        hard_labels = labels[:count].clone()
        return hard_images, hard_labels


@dataclass
class FLIPDefense:
    """Factory for simplified FLIP-style benign-client hardening."""

    target_label: int = 0
    trigger_size: int = 4
    trigger_alpha: float = 0.2
    reverse_steps: int = 20
    reverse_lr: float = 0.05
    hardening_ratio: float = 0.5
    input_bounds: InputBounds | None = None

    @classmethod
    def from_config(cls, config: dict) -> "FLIPDefense":
        flip_cfg = config.get("flip", {})
        # FLIP optimizes triggers in the model input space; after Normalize,
        # the valid pixel box is [(0-mean)/std, (1-mean)/std].
        input_bounds = get_normalized_input_bounds(config)
        return cls(
            target_label=int(flip_cfg.get("target_label", 0)),
            trigger_size=int(flip_cfg.get("trigger_size", 4)),
            trigger_alpha=float(flip_cfg.get("trigger_alpha", 0.2)),
            reverse_steps=int(flip_cfg.get("reverse_steps", 20)),
            reverse_lr=float(flip_cfg.get("reverse_lr", 0.05)),
            hardening_ratio=float(flip_cfg.get("hardening_ratio", 0.5)),
            input_bounds=input_bounds,
        )

    def build_client_hardening(
        self,
        *,
        global_model: nn.Module,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        device: torch.device | str,
    ) -> FLIPClientHardening:
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
        )
        trigger = optimize_reverse_trigger(
            global_model=global_model,
            dataloader=loader,
            target_label=self.target_label,
            trigger_size=self.trigger_size,
            trigger_alpha=self.trigger_alpha,
            reverse_steps=self.reverse_steps,
            reverse_lr=self.reverse_lr,
            device=device,
            input_bounds=self.input_bounds,
        )
        return FLIPClientHardening(
            trigger_pattern=trigger,
            input_bounds=self.input_bounds,
            trigger_alpha=self.trigger_alpha,
            hardening_ratio=self.hardening_ratio,
        )
