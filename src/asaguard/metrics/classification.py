"""Classification metrics."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def clean_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    if total == 0:
        return 0.0
    return correct / total


@torch.no_grad()
def attack_success_rate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    trigger_fn,
    target_label: int,
    device: torch.device | str,
) -> float:
    """Evaluate targeted ASR on non-target-class samples.

    The trigger function transforms images only. Labels are not modified in the
    dataset; this evaluator filters out samples whose original label is already
    the target class, applies the trigger, and measures prediction to target.
    """
    model.eval()
    target_hits = 0
    total = 0
    target_label = int(target_label)

    for images, labels in dataloader:
        mask = labels != target_label
        if not bool(mask.any()):
            continue

        images = images[mask].to(device)
        triggered = trigger_fn(images)
        predictions = model(triggered).argmax(dim=1)
        target_tensor = torch.full_like(predictions, target_label)
        target_hits += int((predictions == target_tensor).sum().item())
        total += int(predictions.numel())

    if total == 0:
        return 0.0
    return target_hits / total
