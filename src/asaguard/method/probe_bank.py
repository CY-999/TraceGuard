"""Server-side trigger-family probe bank for ASAGuard.

Probe instances are sampled by the server for auditing uploaded updates. They
are never sent to clients and are separate from training-time attack triggers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class ProbeBatch:
    clean_x: torch.Tensor
    trigger_x: torch.Tensor
    target_y: torch.Tensor
    family_name: list[str]


def _as_batch(images: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError("Expected image tensor with shape CxHxW or BxCxHxW")


def _restore_shape(images: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return images.squeeze(0) if squeezed else images


def _safe_clamp(images: torch.Tensor) -> torch.Tensor:
    # CIFAR tensors in this repo may be normalized, so use a broad legal range
    # instead of forcing [0, 1].
    return images.clamp(-3.0, 3.0)


class TriggerFamilyProbeBank:
    def __init__(
        self,
        *,
        num_probes: int = 32,
        probe_families: list[str] | None = None,
        num_classes: int = 10,
        adaptive_steps: int = 5,
        base_seed: int = 42,
    ) -> None:
        self.num_probes = int(num_probes)
        self.probe_families = probe_families or [
            "patch",
            "blend_low_alpha",
            "frequency_sine",
            "warping",
        ]
        self.num_classes = int(num_classes)
        self.adaptive_steps = int(adaptive_steps)
        self.base_seed = int(base_seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TriggerFamilyProbeBank":
        tg_cfg = config.get("asaguard", {})
        return cls(
            num_probes=int(tg_cfg.get("num_probes", 32)),
            probe_families=list(tg_cfg.get("probe_families", [])) or None,
            num_classes=int(config.get("model", {}).get("num_classes", 10)),
            adaptive_steps=int(tg_cfg.get("adaptive_steps", 5)),
            base_seed=int(config.get("training", {}).get("seed", 42)),
        )

    def sample(
        self,
        reference_dataset: Dataset,
        *,
        round_idx: int,
        model: torch.nn.Module | None = None,
        device: torch.device | str = "cpu",
    ) -> ProbeBatch:
        if len(reference_dataset) == 0:
            raise ValueError("ASAGuard reference dataset is empty")

        rng = np.random.default_rng(self.base_seed + int(round_idx))
        clean_images: list[torch.Tensor] = []
        trigger_images: list[torch.Tensor] = []
        target_labels: list[int] = []
        family_names: list[str] = []

        for _ in range(self.num_probes):
            index = int(rng.integers(0, len(reference_dataset)))
            clean_x, label = reference_dataset[index]
            clean_x = clean_x.detach().clone() if isinstance(clean_x, torch.Tensor) else torch.as_tensor(clean_x)
            target_y = int(rng.integers(0, self.num_classes))
            if self.num_classes > 1 and target_y == int(label):
                target_y = (target_y + 1) % self.num_classes
            family = str(rng.choice(self.probe_families))

            trigger_x = self._apply_family(
                clean_x,
                family,
                target_y=target_y,
                rng=rng,
                model=model,
                device=device,
            )
            clean_images.append(clean_x)
            trigger_images.append(trigger_x)
            target_labels.append(target_y)
            family_names.append(family)

        return ProbeBatch(
            clean_x=torch.stack(clean_images, dim=0),
            trigger_x=torch.stack(trigger_images, dim=0),
            target_y=torch.tensor(target_labels, dtype=torch.long),
            family_name=family_names,
        )

    def _apply_family(
        self,
        image: torch.Tensor,
        family: str,
        *,
        target_y: int,
        rng: np.random.Generator,
        model: torch.nn.Module | None,
        device: torch.device | str,
    ) -> torch.Tensor:
        if family == "patch":
            return self._patch(image, rng)
        if family == "blend_low_alpha":
            return self._blend_low_alpha(image, rng)
        if family == "frequency_sine":
            return self._frequency_sine(image, rng)
        if family == "warping":
            return self._warping(image, rng)
        if family == "adaptive_like":
            return self._adaptive_like(image, target_y=target_y, model=model, device=device)
        raise ValueError(f"Unsupported ASAGuard probe family: {family}")

    def _patch(self, image: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        output = image.clone()
        _, height, width = output.shape
        size = int(rng.integers(3, max(4, min(height, width) // 4 + 1)))
        top = int(rng.integers(0, height - size + 1))
        left = int(rng.integers(0, width - size + 1))
        value = float(rng.uniform(1.0, 2.5))
        output[:, top : top + size, left : left + size] = value
        return _safe_clamp(output)

    def _blend_low_alpha(self, image: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        alpha = float(rng.uniform(0.05, 0.2))
        pattern_value = float(rng.uniform(0.5, 2.0))
        pattern = torch.full_like(image, pattern_value)
        return _safe_clamp((1.0 - alpha) * image + alpha * pattern)

    def _frequency_sine(self, image: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        _, height, width = image.shape
        xs = torch.linspace(0.0, 2.0 * pi, width, dtype=image.dtype, device=image.device)
        ys = torch.linspace(0.0, 2.0 * pi, height, dtype=image.dtype, device=image.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        frequency = float(rng.uniform(3.0, 8.0))
        phase = float(rng.uniform(0.0, 2.0 * pi))
        amplitude = float(rng.uniform(0.05, 0.15))
        sine = torch.sin(frequency * (grid_x + grid_y) + phase).unsqueeze(0)
        return _safe_clamp(image + amplitude * sine)

    def _warping(self, image: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        batch, squeezed = _as_batch(image)
        _, _, height, width = batch.shape
        ys = torch.linspace(-1.0, 1.0, height, dtype=batch.dtype, device=batch.device)
        xs = torch.linspace(-1.0, 1.0, width, dtype=batch.dtype, device=batch.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        strength = float(rng.uniform(0.03, 0.12))
        frequency = float(rng.uniform(1.0, 3.0))
        offset_x = strength * torch.sin(frequency * pi * grid_y)
        offset_y = strength * torch.sin(frequency * pi * grid_x)
        grid = torch.stack((grid_x + offset_x, grid_y + offset_y), dim=-1).unsqueeze(0)
        warped = F.grid_sample(
            batch,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )
        return _restore_shape(_safe_clamp(warped), squeezed)

    def _adaptive_like(
        self,
        image: torch.Tensor,
        *,
        target_y: int,
        model: torch.nn.Module | None,
        device: torch.device | str,
    ) -> torch.Tensor:
        if model is None or self.adaptive_steps <= 0:
            return self._blend_low_alpha(image, np.random.default_rng(self.base_seed))

        device = torch.device(device)
        was_training = model.training
        model.eval()
        x = image.unsqueeze(0).to(device)
        _, _, height, width = x.shape
        size = max(3, min(6, height, width))
        pattern = x[:, :, height - size : height, width - size : width].detach().clone()
        pattern.requires_grad_(True)
        optimizer = torch.optim.Adam([pattern], lr=0.05)
        target = torch.tensor([int(target_y)], dtype=torch.long, device=device)

        for _ in range(self.adaptive_steps):
            optimizer.zero_grad(set_to_none=True)
            patched = x.clone()
            patched[:, :, height - size : height, width - size : width] = pattern
            loss = torch.nn.functional.cross_entropy(model(_safe_clamp(patched)), target)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pattern.clamp_(-3.0, 3.0)

        if was_training:
            model.train()
        output = image.clone().to(device)
        output[:, height - size : height, width - size : width] = pattern.detach().squeeze(0)
        return _safe_clamp(output).detach().cpu()
