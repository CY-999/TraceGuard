"""Model Replacement attack support.

This module implements only the training-time data poisoning and update scaling
needed for the classic Model Replacement baseline. It does not implement DBA,
Neurotoxin, A3FL, TRACEGuard, or any defense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from traceguard.attacks.triggers import (
    apply_blend_trigger,
    apply_frequency_trigger,
    apply_patch_trigger,
    apply_warping_trigger,
)


TriggerFn = Callable[[torch.Tensor], torch.Tensor]


def get_trigger_fn(trigger_type: str) -> TriggerFn:
    trigger_type = trigger_type.lower()
    if trigger_type == "patch":
        return apply_patch_trigger
    if trigger_type in {"blend", "low_alpha", "low-alpha"}:
        return apply_blend_trigger
    if trigger_type in {"frequency", "sine"}:
        return apply_frequency_trigger
    if trigger_type in {"warping", "warp"}:
        return apply_warping_trigger
    raise ValueError(f"Unsupported trigger_type for Model Replacement: {trigger_type}")


class PoisonedDataset(Dataset):
    """Dataset wrapper that poisons a fixed subset of local samples."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        target_label: int,
        poison_ratio: float,
        trigger_fn: TriggerFn,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.target_label = int(target_label)
        self.trigger_fn = trigger_fn

        poison_ratio = max(0.0, min(float(poison_ratio), 1.0))
        candidate_indices = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if int(label) != self.target_label:
                candidate_indices.append(idx)

        num_poison = int(round(len(candidate_indices) * poison_ratio))
        rng = np.random.default_rng(seed)
        selected = rng.choice(candidate_indices, size=num_poison, replace=False) if num_poison else []
        self.poison_indices = set(int(idx) for idx in selected)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        if idx in self.poison_indices:
            image = self.trigger_fn(image)
            label = self.target_label
        return image, label


@dataclass
class ModelReplacementAttack:
    target_label: int = 0
    poison_ratio: float = 0.1
    trigger_type: str = "patch"
    scale_factor: float = 1.0
    num_malicious: int = 0
    seed: int = 42

    @classmethod
    def from_config(cls, config: dict) -> "ModelReplacementAttack":
        attack_cfg = config.get("attack", {})
        return cls(
            target_label=int(attack_cfg.get("target_label", 0)),
            poison_ratio=float(attack_cfg.get("poison_ratio", 0.1)),
            trigger_type=str(attack_cfg.get("trigger_type", "patch")),
            scale_factor=float(attack_cfg.get("scale_factor", 1.0)),
            num_malicious=int(attack_cfg.get("num_malicious", 0)),
            seed=int(config.get("training", {}).get("seed", 42)),
        )

    @property
    def trigger_fn(self) -> TriggerFn:
        return get_trigger_fn(self.trigger_type)

    def malicious_client_ids(self, total_clients: int) -> set[int]:
        count = max(0, min(int(self.num_malicious), int(total_clients)))
        return set(range(count))

    def poison_dataset(self, dataset: Dataset, *, client_id: int) -> Dataset:
        return PoisonedDataset(
            dataset,
            target_label=self.target_label,
            poison_ratio=self.poison_ratio,
            trigger_fn=self.trigger_fn,
            seed=self.seed + int(client_id),
        )

    def scale_update(self, update: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value * self.scale_factor for key, value in update.items()}
