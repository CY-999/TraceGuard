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


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError("Expected image tensor with shape CxHxW or BxCxHxW")


def _restore_shape(images: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return images.squeeze(0) if squeezed else images


def initialize_reverse_trigger(
    *,
    channels: int = 3,
    trigger_size: int = 4,
    init_value: float = 0.5,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Initialize a learnable reverse-engineered trigger patch."""
    return torch.full(
        (int(channels), int(trigger_size), int(trigger_size)),
        float(init_value),
        dtype=torch.float32,
        device=device,
    ).clamp(0.0, 1.0)


def apply_reverse_trigger(
    images: torch.Tensor,
    trigger_pattern: torch.Tensor,
    *,
    trigger_alpha: float = 0.2,
    location: str = "bottom_right",
    value_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Blend a reverse-engineered patch into an image or image batch."""
    batch, squeezed = _as_batch(images)
    output = batch.clone()
    _, channels, height, width = output.shape
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
    return _restore_shape(output.clamp(*value_range), squeezed)


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
    ).requires_grad_(True)
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
        )
        loss = criterion(model(triggered), targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            trigger.clamp_(0.0, 1.0)

    return trigger.detach().cpu().clamp(0.0, 1.0)


@dataclass
class FLIPClientHardening:
    """Batch-level hardening used only by benign clients."""

    trigger_pattern: torch.Tensor
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

    @classmethod
    def from_config(cls, config: dict) -> "FLIPDefense":
        defense_cfg = config.get("defense", {})
        return cls(
            target_label=int(defense_cfg.get("target_label", 0)),
            trigger_size=int(defense_cfg.get("trigger_size", 4)),
            trigger_alpha=float(defense_cfg.get("trigger_alpha", 0.2)),
            reverse_steps=int(defense_cfg.get("reverse_steps", 20)),
            reverse_lr=float(defense_cfg.get("reverse_lr", 0.05)),
            hardening_ratio=float(defense_cfg.get("hardening_ratio", 0.5)),
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
        )
        return FLIPClientHardening(
            trigger_pattern=trigger,
            trigger_alpha=self.trigger_alpha,
            hardening_ratio=self.hardening_ratio,
        )
