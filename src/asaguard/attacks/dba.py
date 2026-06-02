"""DBA: Distributed Backdoor Attack utilities.

The DBA benchmark decomposes one global pattern into local fragments. Each
selected malicious client trains with only its assigned fragment, while ASR
evaluation applies the complete global pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
import torch
from torch.utils.data import Dataset

from asaguard.data.datasets import DATASET_NORMALIZATION


Bounds = tuple[tuple[float, ...], tuple[float, ...]]


@dataclass(frozen=True)
class DBAPatternSpec:
    num_fragments: int
    trigger_size: int
    trigger_gap: int
    trigger_location: int
    fragment_rows: int
    trigger_value: float
    image_size: tuple[int, int]
    normalized_input_bounds: Bounds
    input_mean: tuple[float, ...] | None = None
    input_std: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DBAFragment:
    top: int
    left: int
    height: int
    width: int
    value: float = 1.0


DATASET_DBA_DEFAULTS = {
    "fakedata": {
        "num_fragments": 4,
        "trigger_size": 4,
        "trigger_gap": 2,
        "trigger_location": 0,
        "fragment_rows": 1,
        "image_size": (32, 32),
    },
    "cifar10": {
        "num_fragments": 4,
        "trigger_size": 6,
        "trigger_gap": 3,
        "trigger_location": 0,
        "fragment_rows": 1,
        "image_size": (32, 32),
    },
    "cifar100": {
        "num_fragments": 4,
        "trigger_size": 6,
        "trigger_gap": 3,
        "trigger_location": 0,
        "fragment_rows": 1,
        "image_size": (32, 32),
    },
    "tinyimagenet": {
        "num_fragments": 4,
        "trigger_size": 10,
        "trigger_gap": 2,
        "trigger_location": 0,
        "fragment_rows": 2,
        "image_size": (64, 64),
    },
}


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError("Expected image tensor with shape CxHxW or BxCxHxW")


def _restore_shape(images: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return images.squeeze(0) if squeezed else images


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


def _image_size_from_config(data_cfg: dict, defaults: dict) -> tuple[int, int]:
    configured = data_cfg.get("image_size")
    if configured is None:
        return tuple(defaults["image_size"])
    if isinstance(configured, int):
        return int(configured), int(configured)
    if len(configured) != 2:
        raise ValueError("dataset.image_size must be an int or a two-element sequence")
    return int(configured[0]), int(configured[1])


def build_dba_pattern_spec(config: dict) -> DBAPatternSpec:
    attack_cfg = config.get("attack", {})
    data_cfg = config.get("dataset", {})
    dba_cfg = config.get("dba", {})
    dataset_name = str(data_cfg.get("name", "")).lower()
    defaults = DATASET_DBA_DEFAULTS.get(dataset_name, DATASET_DBA_DEFAULTS["fakedata"])

    normalization = DATASET_NORMALIZATION.get(dataset_name)
    if normalization is None:
        input_mean = None
        input_std = None
        bounds = _raw_bounds()
    else:
        input_mean = tuple(float(value) for value in normalization[0])
        input_std = tuple(float(value) for value in normalization[1])
        bounds = _bounds_from_normalization(input_mean, input_std)

    spec = DBAPatternSpec(
        num_fragments=int(dba_cfg.get("num_fragments", defaults["num_fragments"])),
        trigger_size=int(dba_cfg.get("trigger_size", defaults["trigger_size"])),
        trigger_gap=int(dba_cfg.get("trigger_gap", defaults["trigger_gap"])),
        trigger_location=int(dba_cfg.get("trigger_location", defaults["trigger_location"])),
        fragment_rows=int(dba_cfg.get("fragment_rows", defaults["fragment_rows"])),
        trigger_value=float(dba_cfg.get("trigger_value", attack_cfg.get("trigger_value", 1.0))),
        image_size=_image_size_from_config(data_cfg, defaults),
        normalized_input_bounds=bounds,
        input_mean=input_mean,
        input_std=input_std,
    )
    _validate_dba_pattern_spec(spec)
    return spec


def _validate_dba_pattern_spec(spec: DBAPatternSpec) -> None:
    if spec.num_fragments <= 0:
        raise ValueError("DBA num_fragments must be positive")
    if spec.trigger_size <= 0:
        raise ValueError("DBA trigger_size must be positive")
    if spec.trigger_gap < 0:
        raise ValueError("DBA trigger_gap must be non-negative")
    if spec.trigger_location < 0:
        raise ValueError("DBA trigger_location must be non-negative")
    if spec.fragment_rows <= 0:
        raise ValueError("DBA fragment_rows must be positive")
    if spec.image_size[0] <= 0 or spec.image_size[1] <= 0:
        raise ValueError("DBA image_size must be positive")


def build_global_dba_trigger(spec: DBAPatternSpec) -> list[DBAFragment]:
    """Build the complete DBA global pattern as local fragments."""
    height, width = int(spec.image_size[0]), int(spec.image_size[1])
    fragment_height = max(1, min(int(spec.fragment_rows), height))
    fragment_width = max(1, min(int(spec.trigger_size), width))
    offset = int(spec.trigger_location)
    gap = int(spec.trigger_gap)
    columns = min(2, int(spec.num_fragments))

    fragments: list[DBAFragment] = []
    for idx in range(int(spec.num_fragments)):
        row = idx // columns
        col = idx % columns
        top = offset + row * (fragment_height + gap)
        left = offset + col * (fragment_width + gap)
        if top + fragment_height > height or left + fragment_width > width:
            raise ValueError(
                "DBA pattern does not fit image_size; adjust trigger_size, "
                "trigger_gap, trigger_location, fragment_rows, or image_size"
            )
        fragments.append(
            DBAFragment(
                top=top,
                left=left,
                height=fragment_height,
                width=fragment_width,
                value=float(spec.trigger_value),
            )
        )
    return fragments


def split_global_trigger_into_fragments(
    global_trigger: list[DBAFragment],
) -> list[DBAFragment]:
    return list(global_trigger)


def _coerce_bounds(
    images: torch.Tensor,
    spec: DBAPatternSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _ = _as_batch(images)
    channels = int(batch.shape[1])
    lower_values, upper_values = spec.normalized_input_bounds
    if len(lower_values) != channels or len(upper_values) != channels:
        raise ValueError("DBA input bounds channel count must match images")
    lower = torch.tensor(lower_values, device=images.device, dtype=images.dtype).view(1, channels, 1, 1)
    upper = torch.tensor(upper_values, device=images.device, dtype=images.dtype).view(1, channels, 1, 1)
    return lower, upper


def _clamp_to_spec(images: torch.Tensor, spec: DBAPatternSpec) -> torch.Tensor:
    lower, upper = _coerce_bounds(images, spec)
    return torch.maximum(torch.minimum(images, upper), lower)


def _trigger_value_tensor(
    images: torch.Tensor,
    spec: DBAPatternSpec,
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


def _spec_for_images(images: torch.Tensor, spec: DBAPatternSpec) -> DBAPatternSpec:
    batch, _ = _as_batch(images)
    image_size = (int(batch.shape[2]), int(batch.shape[3]))
    if image_size == spec.image_size:
        return spec
    resized = replace(spec, image_size=image_size)
    _validate_dba_pattern_spec(resized)
    return resized


def apply_dba_fragment(
    images: torch.Tensor,
    fragment: DBAFragment,
    spec: DBAPatternSpec,
) -> torch.Tensor:
    """Apply one local DBA fragment in the model input space."""
    batch, squeezed = _as_batch(images)
    output = batch.clone()
    _, _, height, width = output.shape
    top = max(0, min(int(fragment.top), height - int(fragment.height)))
    left = max(0, min(int(fragment.left), width - int(fragment.width)))
    value = _trigger_value_tensor(output, spec)
    output[:, :, top : top + int(fragment.height), left : left + int(fragment.width)] = value
    return _restore_shape(_clamp_to_spec(output, spec), squeezed)


def apply_full_dba_trigger(
    images: torch.Tensor,
    spec: DBAPatternSpec,
) -> torch.Tensor:
    """Apply the complete DBA global pattern for ASR evaluation."""
    active_spec = _spec_for_images(images, spec)
    output = images
    for fragment in split_global_trigger_into_fragments(build_global_dba_trigger(active_spec)):
        output = apply_dba_fragment(output, fragment, active_spec)
    return output


class DBAPoisonedDataset(Dataset):
    """Dataset wrapper that poisons samples with one local DBA fragment."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        target_label: int,
        poison_ratio: float,
        fragment_index: int,
        spec: DBAPatternSpec,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.target_label = int(target_label)
        self.fragment_index = int(fragment_index)
        self.spec = spec

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
            active_spec = _spec_for_images(image, self.spec)
            fragments = split_global_trigger_into_fragments(build_global_dba_trigger(active_spec))
            fragment = fragments[self.fragment_index % len(fragments)]
            image = apply_dba_fragment(image, fragment, active_spec)
            label = self.target_label
        return image, label


@dataclass
class DBAAttack:
    target_label: int = 0
    poison_ratio: float = 0.1
    num_malicious: int = 0
    pattern_spec: DBAPatternSpec | None = None
    seed: int = 42

    @classmethod
    def from_config(cls, config: dict) -> "DBAAttack":
        attack_cfg = config.get("attack", {})
        spec = build_dba_pattern_spec(config)
        num_malicious = int(attack_cfg.get("num_malicious", 0))
        if num_malicious < spec.num_fragments:
            raise ValueError(
                "DBA requires num_malicious >= num_fragments "
                f"({num_malicious} < {spec.num_fragments})"
            )
        return cls(
            target_label=int(attack_cfg.get("target_label", 0)),
            poison_ratio=float(attack_cfg.get("poison_ratio", 0.1)),
            num_malicious=num_malicious,
            pattern_spec=spec,
            seed=int(config.get("training", {}).get("seed", 42)),
        )

    @property
    def spec(self) -> DBAPatternSpec:
        if self.pattern_spec is None:
            return build_dba_pattern_spec({"dataset": {"name": "fakedata"}, "attack": {}})
        return self.pattern_spec

    @property
    def global_trigger(self) -> list[DBAFragment]:
        return build_global_dba_trigger(self.spec)

    def malicious_client_ids(self, total_clients: int) -> set[int]:
        count = max(0, min(int(self.num_malicious), int(total_clients)))
        return set(range(count))

    def fragment_for_malicious_client_index(self, malicious_client_index: int) -> int:
        return int(malicious_client_index) % int(self.spec.num_fragments)

    def poison_dataset(self, dataset: Dataset, *, client_id: int) -> Dataset:
        fragment_index = self.fragment_for_malicious_client_index(client_id)
        return DBAPoisonedDataset(
            dataset,
            target_label=self.target_label,
            poison_ratio=self.poison_ratio,
            fragment_index=fragment_index,
            spec=self.spec,
            seed=self.seed + int(client_id),
        )

    def trigger_fn(self, images: torch.Tensor) -> torch.Tensor:
        return apply_full_dba_trigger(images, self.spec)
