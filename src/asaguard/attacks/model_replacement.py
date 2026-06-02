"""Model Replacement attack support.

This module implements only the training-time data poisoning and update scaling
needed for the classic Model Replacement baseline. It does not implement DBA,
Neurotoxin, A3FL, asaguard, or any defense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from asaguard.attacks.triggers import (
    TriggerSpec,
    apply_configured_trigger,
    build_trigger_from_config,
)


TriggerFn = Callable[[torch.Tensor], torch.Tensor]
ScaleFactor = float | str


def get_trigger_fn(trigger_type: str) -> TriggerFn:
    spec = build_trigger_from_config({"trigger_type": trigger_type}, {})
    return lambda images: apply_configured_trigger(images, spec)


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
            image = self.trigger_fn(image)
            label = self.target_label
        return image, label


@dataclass
class ModelReplacementAttack:
    target_label: int = 0
    poison_ratio: float = 0.1
    trigger_type: str = "patch"
    trigger_spec: TriggerSpec | None = None
    scale_factor: ScaleFactor = "auto"
    num_malicious: int = 0
    seed: int = 42

    @classmethod
    def from_config(cls, config: dict) -> "ModelReplacementAttack":
        attack_cfg = config.get("attack", {})
        data_cfg = config.get("dataset", {})
        trigger_spec = build_trigger_from_config(attack_cfg, data_cfg)
        scale_factor = parse_scale_factor(attack_cfg.get("scale_factor", "auto"))
        return cls(
            target_label=int(attack_cfg.get("target_label", 0)),
            poison_ratio=float(attack_cfg.get("poison_ratio", 0.1)),
            trigger_type=str(attack_cfg.get("trigger_type", "patch")),
            trigger_spec=trigger_spec,
            scale_factor=scale_factor,
            num_malicious=int(attack_cfg.get("num_malicious", 0)),
            seed=int(config.get("training", {}).get("seed", 42)),
        )

    @property
    def trigger_fn(self) -> TriggerFn:
        spec = self.trigger_spec
        if spec is None:
            spec = build_trigger_from_config({"trigger_type": self.trigger_type}, {})
        return lambda images: apply_configured_trigger(images, spec)

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

    def scale_factor_for_client(
        self,
        *,
        aggregation_weight: float,
        num_selected_malicious: int,
    ) -> float:
        if self.scale_factor != "auto":
            return float(self.scale_factor)

        num_selected_malicious = int(num_selected_malicious)
        if num_selected_malicious <= 0:
            return 1.0

        aggregation_weight = float(aggregation_weight)
        if aggregation_weight <= 0.0:
            raise ValueError("Model Replacement auto scale requires a positive aggregation weight")
        return 1.0 / (aggregation_weight * float(num_selected_malicious))

    def scale_update(
        self,
        update: dict[str, torch.Tensor],
        *,
        scale_factor: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if scale_factor is None:
            if self.scale_factor == "auto":
                raise ValueError("Model Replacement auto scale requires per-round aggregation weights")
            scale_factor = float(self.scale_factor)
        if float(scale_factor) <= 0.0:
            raise ValueError("Model Replacement scale_factor must be positive")

        return {
            key: value * float(scale_factor) if torch.is_floating_point(value) else value
            for key, value in update.items()
        }


def parse_scale_factor(value) -> ScaleFactor:  # noqa: ANN001
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise ValueError("Model Replacement scale_factor must be 'auto' or a positive float") from exc
    else:
        parsed = float(value)

    if parsed <= 0.0:
        raise ValueError("Model Replacement numeric scale_factor must be positive")
    return parsed
