"""A3FL-style adaptive trigger baseline.

This is a lightweight local benchmark implementation. The attacker optimizes a
learnable trigger against the current global model and malicious local data,
then uses that trigger to poison local samples. It does not access any
server-secret TRACEGuard probe instances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


def initialize_adaptive_trigger(
    *,
    channels: int = 3,
    trigger_size: int = 4,
    init_value: float = 0.5,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Initialize a learnable adaptive trigger patch."""
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        trigger = torch.rand(
            channels,
            int(trigger_size),
            int(trigger_size),
            generator=generator,
        )
    else:
        trigger = torch.full(
            (channels, int(trigger_size), int(trigger_size)),
            float(init_value),
        )
    return trigger.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def apply_adaptive_trigger(
    images: torch.Tensor,
    trigger_pattern: torch.Tensor,
    *,
    trigger_alpha: float = 0.2,
    location: str = "bottom_right",
    value_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Blend an adaptive trigger patch into an image or image batch."""
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
        raise ValueError(f"Unsupported adaptive trigger location: {location}")

    alpha = max(0.0, min(float(trigger_alpha), 1.0))
    patch = pattern[:, :patch_h, :patch_w]
    region = output[:, :, top : top + patch_h, left : left + patch_w]
    output[:, :, top : top + patch_h, left : left + patch_w] = (
        (1.0 - alpha) * region + alpha * patch
    )
    return _restore_shape(output.clamp(*value_range), squeezed)


def optimize_adaptive_trigger(
    *,
    global_model: nn.Module,
    dataloader: DataLoader,
    target_label: int,
    trigger_size: int = 4,
    trigger_alpha: float = 0.2,
    trigger_lr: float = 0.05,
    adaptive_steps: int = 20,
    dynamics_perturb_steps: int = 1,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    """Optimize a trigger to induce target-label predictions.

    The lightweight global-dynamics approximation is input perturbation during
    trigger optimization: each step includes extra losses on slightly perturbed
    triggered images, encouraging the trigger to survive small training-time
    changes.
    """
    device = torch.device(device)
    model = global_model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trigger = initialize_adaptive_trigger(
        channels=3,
        trigger_size=trigger_size,
        seed=seed,
        device=device,
    ).requires_grad_(True)
    optimizer = torch.optim.Adam([trigger], lr=float(trigger_lr))
    criterion = nn.CrossEntropyLoss()
    batches = list(dataloader)
    if not batches:
        return trigger.detach().cpu().clamp(0.0, 1.0)

    rng = np.random.default_rng(seed)
    target_label = int(target_label)
    steps = max(1, int(adaptive_steps))
    perturb_steps = max(0, int(dynamics_perturb_steps))

    for _ in range(steps):
        images, labels = batches[int(rng.integers(0, len(batches)))]
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

        triggered = apply_adaptive_trigger(
            images,
            trigger,
            trigger_alpha=trigger_alpha,
        )
        loss = criterion(model(triggered), targets)

        for _ in range(perturb_steps):
            noise = torch.randn_like(triggered) * 0.01
            perturbed = (triggered + noise).clamp(0.0, 1.0)
            loss = loss + criterion(model(perturbed), targets)

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            trigger.clamp_(0.0, 1.0)

    return trigger.detach().cpu().clamp(0.0, 1.0)


class A3FLPoisonedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        target_label: int,
        poison_ratio: float,
        trigger_pattern: torch.Tensor,
        trigger_alpha: float,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.target_label = int(target_label)
        self.trigger_pattern = trigger_pattern.detach().cpu().clone()
        self.trigger_alpha = float(trigger_alpha)

        poison_ratio = max(0.0, min(float(poison_ratio), 1.0))
        candidate_indices = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if int(label) != self.target_label:
                candidate_indices.append(idx)

        num_poison = min(
            len(candidate_indices),
            int(round(len(dataset) * poison_ratio)),
        )
        rng = np.random.default_rng(seed)
        selected = rng.choice(candidate_indices, size=num_poison, replace=False) if num_poison else []
        self.poison_indices = set(int(idx) for idx in selected)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        if idx in self.poison_indices:
            image = apply_adaptive_trigger(
                image,
                self.trigger_pattern,
                trigger_alpha=self.trigger_alpha,
            )
            label = self.target_label
        return image, label


@dataclass
class A3FLAttack:
    target_label: int = 0
    poison_ratio: float = 0.1
    num_malicious: int = 0
    trigger_size: int = 4
    trigger_alpha: float = 0.2
    trigger_lr: float = 0.05
    adaptive_steps: int = 20
    dynamics_perturb_steps: int = 1
    seed: int = 42
    trigger_pattern: torch.Tensor | None = None

    @classmethod
    def from_config(cls, config: dict) -> "A3FLAttack":
        attack_cfg = config.get("attack", {})
        return cls(
            target_label=int(attack_cfg.get("target_label", 0)),
            poison_ratio=float(attack_cfg.get("poison_ratio", 0.1)),
            num_malicious=int(attack_cfg.get("num_malicious", 0)),
            trigger_size=int(attack_cfg.get("trigger_size", 4)),
            trigger_alpha=float(attack_cfg.get("trigger_alpha", 0.2)),
            trigger_lr=float(attack_cfg.get("trigger_lr", 0.05)),
            adaptive_steps=int(attack_cfg.get("adaptive_steps", 20)),
            dynamics_perturb_steps=int(attack_cfg.get("dynamics_perturb_steps", 1)),
            seed=int(config.get("training", {}).get("seed", 42)),
        )

    def malicious_client_ids(self, total_clients: int) -> set[int]:
        count = max(0, min(int(self.num_malicious), int(total_clients)))
        return set(range(count))

    def poison_dataset_with_model(
        self,
        dataset: Dataset,
        *,
        client_id: int,
        global_model: nn.Module,
        device: torch.device | str,
        batch_size: int,
        num_workers: int,
    ) -> Dataset:
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
        )
        self.trigger_pattern = optimize_adaptive_trigger(
            global_model=global_model,
            dataloader=loader,
            target_label=self.target_label,
            trigger_size=self.trigger_size,
            trigger_alpha=self.trigger_alpha,
            trigger_lr=self.trigger_lr,
            adaptive_steps=self.adaptive_steps,
            dynamics_perturb_steps=self.dynamics_perturb_steps,
            device=device,
            seed=self.seed + int(client_id),
        )
        return A3FLPoisonedDataset(
            dataset,
            target_label=self.target_label,
            poison_ratio=self.poison_ratio,
            trigger_pattern=self.trigger_pattern,
            trigger_alpha=self.trigger_alpha,
            seed=self.seed + int(client_id),
        )

    def trigger_fn(self, images: torch.Tensor) -> torch.Tensor:
        pattern = self.trigger_pattern
        if pattern is None:
            pattern = initialize_adaptive_trigger(
                channels=images.shape[-3],
                trigger_size=self.trigger_size,
                seed=self.seed,
                device=images.device,
            )
        return apply_adaptive_trigger(
            images,
            pattern,
            trigger_alpha=self.trigger_alpha,
        )
