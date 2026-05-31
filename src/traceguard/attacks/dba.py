"""DBA: Distributed Backdoor Attack utilities.

This module implements the core DBA mechanism:
one global trigger is split into local trigger fragments; each malicious client
trains only with its assigned fragment, while ASR evaluation uses the full
global trigger.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DBAFragment:
    top: int
    left: int
    size: int
    value: float = 1.0


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError("Expected image tensor with shape CxHxW or BxCxHxW")


def _restore_shape(images: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return images.squeeze(0) if squeezed else images


def build_global_dba_trigger(
    *,
    image_size: int | tuple[int, int] = 32,
    trigger_size: int = 4,
    trigger_value: float = 1.0,
    fragment_layout: str = "four_corners",
) -> list[DBAFragment]:
    """Build the full DBA trigger as a list of local fragments."""
    if isinstance(image_size, int):
        height = width = int(image_size)
    else:
        height, width = int(image_size[0]), int(image_size[1])

    size = max(1, min(int(trigger_size), height, width))
    value = float(trigger_value)
    layout = fragment_layout.lower()

    if layout != "four_corners":
        raise ValueError(f"Unsupported DBA fragment_layout: {fragment_layout}")

    return [
        DBAFragment(top=0, left=0, size=size, value=value),
        DBAFragment(top=0, left=width - size, size=size, value=value),
        DBAFragment(top=height - size, left=0, size=size, value=value),
        DBAFragment(top=height - size, left=width - size, size=size, value=value),
    ]


def split_global_trigger_into_fragments(
    global_trigger: list[DBAFragment],
) -> list[DBAFragment]:
    """Return local fragments from a global DBA trigger representation."""
    return list(global_trigger)


def apply_dba_fragment(
    images: torch.Tensor,
    fragment: DBAFragment,
    *,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Apply one local DBA fragment to an image or image batch."""
    batch, squeezed = _as_batch(images)
    output = batch.clone()
    _, _, height, width = output.shape
    size = max(1, min(int(fragment.size), height, width))
    top = max(0, min(int(fragment.top), height - size))
    left = max(0, min(int(fragment.left), width - size))
    output[:, :, top : top + size, left : left + size] = float(fragment.value)
    return _restore_shape(output.clamp(*value_range), squeezed)


def apply_full_dba_trigger(
    images: torch.Tensor,
    *,
    global_trigger: list[DBAFragment] | None = None,
    trigger_size: int = 4,
    trigger_value: float = 1.0,
    fragment_layout: str = "four_corners",
    value_range: tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Apply the complete DBA global trigger for ASR evaluation."""
    batch, squeezed = _as_batch(images)
    _, _, height, width = batch.shape
    fragments = global_trigger or build_global_dba_trigger(
        image_size=(height, width),
        trigger_size=trigger_size,
        trigger_value=trigger_value,
        fragment_layout=fragment_layout,
    )

    output = batch
    for fragment in split_global_trigger_into_fragments(fragments):
        output = apply_dba_fragment(
            output,
            fragment,
            value_range=value_range,
        )
    return _restore_shape(output, squeezed)


class DBAPoisonedDataset(Dataset):
    """Dataset wrapper that poisons samples with one local DBA fragment."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        target_label: int,
        poison_ratio: float,
        fragment: DBAFragment,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.target_label = int(target_label)
        self.fragment = fragment

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
            image = apply_dba_fragment(image, self.fragment)
            label = self.target_label
        return image, label


@dataclass
class DBAAttack:
    target_label: int = 0
    poison_ratio: float = 0.1
    num_malicious: int = 0
    trigger_size: int = 4
    trigger_value: float = 1.0
    fragment_layout: str = "four_corners"
    seed: int = 42

    @classmethod
    def from_config(cls, config: dict) -> "DBAAttack":
        attack_cfg = config.get("attack", {})
        return cls(
            target_label=int(attack_cfg.get("target_label", 0)),
            poison_ratio=float(attack_cfg.get("poison_ratio", 0.1)),
            num_malicious=int(attack_cfg.get("num_malicious", 0)),
            trigger_size=int(attack_cfg.get("trigger_size", 4)),
            trigger_value=float(attack_cfg.get("trigger_value", 1.0)),
            fragment_layout=str(attack_cfg.get("fragment_layout", "four_corners")),
            seed=int(config.get("training", {}).get("seed", 42)),
        )

    @property
    def global_trigger(self) -> list[DBAFragment]:
        return build_global_dba_trigger(
            trigger_size=self.trigger_size,
            trigger_value=self.trigger_value,
            fragment_layout=self.fragment_layout,
        )

    def malicious_client_ids(self, total_clients: int) -> set[int]:
        count = max(0, min(int(self.num_malicious), int(total_clients)))
        return set(range(count))

    def fragment_for_malicious_client_index(self, malicious_client_index: int) -> DBAFragment:
        fragments = split_global_trigger_into_fragments(self.global_trigger)
        return fragments[int(malicious_client_index) % len(fragments)]

    def poison_dataset(self, dataset: Dataset, *, client_id: int) -> Dataset:
        fragment = self.fragment_for_malicious_client_index(client_id)
        return DBAPoisonedDataset(
            dataset,
            target_label=self.target_label,
            poison_ratio=self.poison_ratio,
            fragment=fragment,
            seed=self.seed + int(client_id),
        )

    def trigger_fn(self, images: torch.Tensor) -> torch.Tensor:
        return apply_full_dba_trigger(
            images,
            global_trigger=self.global_trigger,
        )
