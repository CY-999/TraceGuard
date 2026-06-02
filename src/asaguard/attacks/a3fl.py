"""A3FL adaptive trigger benchmark.

The attacker optimizes a learnable trigger against the current global model and
an adversarially adapted model copy that simulates trigger unlearning on
malicious local data. It does not access any server-secret asaguard probe
instances.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from asaguard.attacks.triggers import TriggerSpec, build_trigger_from_config


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError("Expected image tensor with shape CxHxW or BxCxHxW")


def _restore_shape(images: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return images.squeeze(0) if squeezed else images


def _bounds_for_images(
    images: torch.Tensor,
    trigger_spec: TriggerSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _ = _as_batch(images)
    channels = int(batch.shape[1])
    lower_values, upper_values = trigger_spec.normalized_input_bounds
    if len(lower_values) != channels or len(upper_values) != channels:
        raise ValueError("A3FL trigger bounds channel count must match images")
    lower = torch.tensor(lower_values, device=images.device, dtype=images.dtype).view(1, channels, 1, 1)
    upper = torch.tensor(upper_values, device=images.device, dtype=images.dtype).view(1, channels, 1, 1)
    return lower, upper


def _bounds_for_pattern(
    channels: int,
    trigger_spec: TriggerSpec,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_values, upper_values = trigger_spec.normalized_input_bounds
    if len(lower_values) != int(channels) or len(upper_values) != int(channels):
        raise ValueError("A3FL trigger bounds channel count must match pattern")
    lower = torch.tensor(lower_values, device=device, dtype=dtype).view(int(channels), 1, 1)
    upper = torch.tensor(upper_values, device=device, dtype=dtype).view(int(channels), 1, 1)
    return lower, upper


def _clamp_images_to_bounds(images: torch.Tensor, trigger_spec: TriggerSpec) -> torch.Tensor:
    lower, upper = _bounds_for_images(images, trigger_spec)
    return torch.maximum(torch.minimum(images, upper), lower)


def _clamp_pattern_to_bounds(pattern: torch.Tensor, trigger_spec: TriggerSpec) -> torch.Tensor:
    lower, upper = _bounds_for_pattern(
        int(pattern.shape[0]),
        trigger_spec,
        device=pattern.device,
        dtype=pattern.dtype,
    )
    return torch.maximum(torch.minimum(pattern, upper), lower)


def _trainable_parameter_names(model: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and torch.is_floating_point(parameter)
    ]


def _set_trainable_parameter_grads(
    model: nn.Module,
    parameter_names: set[str],
    *,
    requires_grad: bool,
) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(requires_grad and name in parameter_names))


def _parameter_cosine_similarity(
    model_a: nn.Module,
    model_b: nn.Module,
    parameter_names: list[str],
    *,
    eps: float = 1.0e-12,
) -> float:
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    numerator: torch.Tensor | None = None
    norm_a: torch.Tensor | None = None
    norm_b: torch.Tensor | None = None

    for name in parameter_names:
        if name not in params_a or name not in params_b:
            raise ValueError(f"A3FL parameter missing while computing lambda similarity: {name}")
        a = params_a[name].detach().reshape(-1).float()
        b = params_b[name].detach().reshape(-1).float()
        if a.shape != b.shape:
            raise ValueError(f"A3FL parameter shape mismatch while computing lambda similarity: {name}")
        dot = torch.dot(a, b)
        aa = torch.dot(a, a)
        bb = torch.dot(b, b)
        numerator = dot if numerator is None else numerator + dot
        norm_a = aa if norm_a is None else norm_a + aa
        norm_b = bb if norm_b is None else norm_b + bb

    if numerator is None or norm_a is None or norm_b is None:
        return 0.0
    similarity = numerator / (torch.sqrt(norm_a) * torch.sqrt(norm_b) + float(eps))
    if not bool(torch.isfinite(similarity)):
        return 0.0
    return float(torch.clamp(similarity, min=0.0, max=1.0).item())


def _sample_non_target_batch(
    batches: list,
    rng: np.random.Generator,
    *,
    target_label: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    for _ in range(len(batches)):
        images, labels = batches[int(rng.integers(0, len(batches)))]
        labels = torch.as_tensor(labels)
        mask = labels != int(target_label)
        if bool(mask.any()):
            return images[mask].to(device), labels[mask].to(device=device, dtype=torch.long)
    return None


def initialize_adaptive_trigger(
    *,
    channels: int = 3,
    trigger_size: int = 4,
    init_value: float = 0.5,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    trigger_spec: TriggerSpec | None = None,
) -> torch.Tensor:
    """Initialize a learnable adaptive trigger patch."""
    if trigger_spec is None:
        trigger_spec = build_trigger_from_config({"trigger_size": trigger_size}, {})
    dtype = torch.float32
    lower, upper = _bounds_for_pattern(
        int(channels),
        trigger_spec,
        device=device,
        dtype=dtype,
    )
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        raw = torch.rand(
            channels,
            int(trigger_size),
            int(trigger_size),
            generator=generator,
        ).to(device=device, dtype=dtype)
        trigger = lower + raw * (upper - lower)
    else:
        raw = torch.full(
            (channels, int(trigger_size), int(trigger_size)),
            float(init_value),
            device=device,
            dtype=dtype,
        )
        trigger = lower + raw * (upper - lower)
    return _clamp_pattern_to_bounds(trigger, trigger_spec)


def apply_adaptive_trigger(
    images: torch.Tensor,
    trigger_pattern: torch.Tensor,
    *,
    trigger_alpha: float = 0.2,
    location: str | tuple[int, int] | None = None,
    trigger_spec: TriggerSpec | None = None,
) -> torch.Tensor:
    """Blend an adaptive trigger patch into an image or image batch."""
    if trigger_spec is None:
        trigger_spec = build_trigger_from_config({}, {})
    if location is None:
        location = trigger_spec.trigger_location
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
    elif isinstance(location, tuple):
        top = max(0, min(int(location[0]), height - patch_h))
        left = max(0, min(int(location[1]), width - patch_w))
    else:
        raise ValueError(f"Unsupported adaptive trigger location: {location}")

    alpha = max(0.0, min(float(trigger_alpha), 1.0))
    patch = pattern[:, :patch_h, :patch_w]
    region = output[:, :, top : top + patch_h, left : left + patch_w]
    output[:, :, top : top + patch_h, left : left + patch_w] = (
        (1.0 - alpha) * region + alpha * patch
    )
    return _restore_shape(_clamp_images_to_bounds(output, trigger_spec), squeezed)


def optimize_adaptive_trigger(
    *,
    global_model: nn.Module,
    dataloader: DataLoader,
    target_label: int,
    trigger_size: int = 4,
    trigger_alpha: float = 0.2,
    trigger_lr: float = 0.05,
    outer_steps: int = 20,
    trigger_steps: int = 1,
    adv_model_lr: float = 0.01,
    lambda0: float = 1.0,
    adv_model_steps: int = 1,
    trigger_location: str | tuple[int, int] = "bottom_right",
    trigger_spec: TriggerSpec | None = None,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    """Optimize a trigger with A3FL-style adversarial adaptation."""
    device = torch.device(device)
    if trigger_spec is None:
        trigger_spec = build_trigger_from_config(
            {
                "trigger_size": trigger_size,
                "trigger_alpha": trigger_alpha,
                "trigger_location": trigger_location,
            },
            {},
        )
    trainable_names = _trainable_parameter_names(global_model)
    if not trainable_names:
        raise ValueError("A3FL requires at least one trainable floating-point parameter")
    trainable_name_set = set(trainable_names)

    theta_t = copy.deepcopy(global_model).to(device)
    theta_t.eval()
    for parameter in theta_t.parameters():
        parameter.requires_grad_(False)

    theta_adv = copy.deepcopy(global_model).to(device)
    theta_adv.train()
    _set_trainable_parameter_grads(theta_adv, trainable_name_set, requires_grad=True)
    adv_parameters = [
        parameter
        for name, parameter in theta_adv.named_parameters()
        if name in trainable_name_set
    ]
    adv_optimizer = torch.optim.SGD(adv_parameters, lr=float(adv_model_lr))

    trigger = initialize_adaptive_trigger(
        channels=3,
        trigger_size=trigger_size,
        seed=seed,
        device=device,
        trigger_spec=trigger_spec,
    ).requires_grad_(True)
    trigger_optimizer = torch.optim.Adam([trigger], lr=float(trigger_lr))
    criterion = nn.CrossEntropyLoss()
    batches = list(dataloader)
    if not batches:
        return _clamp_pattern_to_bounds(trigger.detach(), trigger_spec).cpu()

    rng = np.random.default_rng(seed)
    target_label = int(target_label)
    outer_steps = max(1, int(outer_steps))
    trigger_steps = max(1, int(trigger_steps))
    adv_model_steps = max(0, int(adv_model_steps))
    lambda0 = max(0.0, float(lambda0))

    for _ in range(outer_steps):
        theta_adv.eval()
        _set_trainable_parameter_grads(theta_adv, trainable_name_set, requires_grad=False)
        for _ in range(trigger_steps):
            batch = _sample_non_target_batch(
                batches,
                rng,
                target_label=target_label,
                device=device,
            )
            if batch is None:
                continue
            images, _ = batch
            targets = torch.full(
                (images.shape[0],),
                target_label,
                dtype=torch.long,
                device=device,
            )
            trigger_optimizer.zero_grad(set_to_none=True)

            triggered = apply_adaptive_trigger(
                images,
                trigger,
                trigger_alpha=trigger_alpha,
                location=trigger_location,
                trigger_spec=trigger_spec,
            )
            lambda_t = lambda0 * _parameter_cosine_similarity(
                theta_adv,
                theta_t,
                trainable_names,
            )
            loss = criterion(theta_t(triggered), targets) + float(lambda_t) * criterion(
                theta_adv(triggered),
                targets,
            )
            loss.backward()
            trigger_optimizer.step()
            with torch.no_grad():
                trigger.copy_(_clamp_pattern_to_bounds(trigger, trigger_spec))

        if adv_model_steps == 0:
            continue

        theta_adv.train()
        _set_trainable_parameter_grads(theta_adv, trainable_name_set, requires_grad=True)
        for _ in range(adv_model_steps):
            batch = _sample_non_target_batch(
                batches,
                rng,
                target_label=target_label,
                device=device,
            )
            if batch is None:
                continue
            images, true_labels = batch
            adv_optimizer.zero_grad(set_to_none=True)
            triggered = apply_adaptive_trigger(
                images,
                trigger.detach(),
                trigger_alpha=trigger_alpha,
                location=trigger_location,
                trigger_spec=trigger_spec,
            )
            adv_loss = criterion(theta_adv(triggered), true_labels)
            adv_loss.backward()
            adv_optimizer.step()

    return _clamp_pattern_to_bounds(trigger.detach(), trigger_spec).cpu()


class A3FLPoisonedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        target_label: int,
        poison_ratio: float,
        trigger_pattern: torch.Tensor,
        trigger_alpha: float,
        trigger_location: str | tuple[int, int],
        trigger_spec: TriggerSpec,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.target_label = int(target_label)
        self.trigger_pattern = trigger_pattern.detach().cpu().clone()
        self.trigger_alpha = float(trigger_alpha)
        self.trigger_location = trigger_location
        self.trigger_spec = trigger_spec

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
                location=self.trigger_location,
                trigger_spec=self.trigger_spec,
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
    trigger_location: str = "bottom_right"
    trigger_spec: TriggerSpec | None = None
    trigger_lr: float = 0.05
    outer_steps: int = 20
    trigger_steps: int = 1
    adv_model_lr: float = 0.01
    lambda0: float = 1.0
    adv_model_steps: int = 1
    seed: int = 42
    trigger_pattern: torch.Tensor | None = None
    trigger_round: int | None = None

    @classmethod
    def from_config(cls, config: dict) -> "A3FLAttack":
        attack_cfg = config.get("attack", {})
        data_cfg = config.get("dataset", {})
        trigger_spec = build_trigger_from_config(attack_cfg, data_cfg)
        return cls(
            target_label=int(attack_cfg.get("target_label", 0)),
            poison_ratio=float(attack_cfg.get("poison_ratio", 0.1)),
            num_malicious=int(attack_cfg.get("num_malicious", 0)),
            trigger_size=int(attack_cfg.get("trigger_size", 4)),
            trigger_alpha=float(attack_cfg.get("trigger_alpha", 0.2)),
            trigger_location=str(attack_cfg.get("trigger_location", "bottom_right")),
            trigger_spec=trigger_spec,
            trigger_lr=float(attack_cfg.get("trigger_lr", 0.05)),
            outer_steps=int(attack_cfg.get("outer_steps", attack_cfg.get("adaptive_steps", 20))),
            trigger_steps=int(attack_cfg.get("trigger_steps", 1)),
            adv_model_lr=float(attack_cfg.get("adv_model_lr", 0.01)),
            lambda0=float(attack_cfg.get("lambda0", 1.0)),
            adv_model_steps=int(attack_cfg.get("adv_model_steps", 1)),
            seed=int(config.get("training", {}).get("seed", 42)),
        )

    def malicious_client_ids(self, total_clients: int) -> set[int]:
        count = max(0, min(int(self.num_malicious), int(total_clients)))
        return set(range(count))

    def _ensure_trigger_pattern(
        self,
        *,
        channels: int = 3,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        if self.trigger_pattern is None:
            self.trigger_pattern = initialize_adaptive_trigger(
                channels=int(channels),
                trigger_size=self.trigger_size,
                seed=self.seed,
                device=device,
                trigger_spec=self.trigger_spec,
            ).cpu()
        return self.trigger_pattern

    def prepare_round_trigger(
        self,
        global_model: nn.Module,
        selected_client_datasets: list[Dataset],
        *,
        round_idx: int,
        device: torch.device | str,
        batch_size: int,
        num_workers: int,
    ) -> None:
        if not selected_client_datasets:
            return
        if self.trigger_round == int(round_idx) and self.trigger_pattern is not None:
            return

        round_dataset: Dataset
        if len(selected_client_datasets) == 1:
            round_dataset = selected_client_datasets[0]
        else:
            round_dataset = ConcatDataset(selected_client_datasets)
        loader = DataLoader(
            round_dataset,
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
            outer_steps=self.outer_steps,
            trigger_steps=self.trigger_steps,
            adv_model_lr=self.adv_model_lr,
            lambda0=self.lambda0,
            adv_model_steps=self.adv_model_steps,
            trigger_location=self.trigger_location,
            trigger_spec=self.trigger_spec,
            device=device,
            seed=self.seed + int(round_idx),
        )
        self.trigger_round = int(round_idx)

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
        pattern = self._ensure_trigger_pattern(device=device)
        return A3FLPoisonedDataset(
            dataset,
            target_label=self.target_label,
            poison_ratio=self.poison_ratio,
            trigger_pattern=pattern,
            trigger_alpha=self.trigger_alpha,
            trigger_location=self.trigger_location,
            trigger_spec=self.trigger_spec or build_trigger_from_config({}, {}),
            seed=self.seed + int(client_id),
        )

    def trigger_fn(self, images: torch.Tensor) -> torch.Tensor:
        pattern = self._ensure_trigger_pattern(
            channels=images.shape[-3],
            device=images.device,
        )
        return apply_adaptive_trigger(
            images,
            pattern,
            trigger_alpha=self.trigger_alpha,
            location=self.trigger_location,
            trigger_spec=self.trigger_spec,
        )
