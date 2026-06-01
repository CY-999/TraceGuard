"""Simplified FDCR-style defense baseline.

This module implements a lightweight baseline inspired by FDCR's use of Fisher
Information parameter-importance profiles in heterogeneous FL. It is not a
claim of exact official FDCR reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


Update = dict[str, torch.Tensor]
Importance = dict[str, torch.Tensor]


def estimate_fisher_importance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    *,
    fisher_batches: int = 1,
) -> Importance:
    """Estimate diagonal Fisher Information with averaged squared gradients."""
    device = torch.device(device)
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    fisher = {
        key: torch.zeros_like(value, dtype=torch.float32, device="cpu")
        for key, value in model.state_dict().items()
        if torch.is_floating_point(value)
    }

    batches_used = 0
    for images, labels in dataloader:
        if batches_used >= max(1, int(fisher_batches)):
            break
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()

        named_params = dict(model.named_parameters())
        for key in fisher:
            parameter = named_params.get(key)
            if parameter is not None and parameter.grad is not None:
                fisher[key] += parameter.grad.detach().cpu().float().pow(2)
        batches_used += 1

    model.zero_grad(set_to_none=True)
    denom = float(max(batches_used, 1))
    return {key: value / denom for key, value in fisher.items()}


def flatten_importance_profile(importance: Importance) -> torch.Tensor:
    if not importance:
        raise ValueError("Cannot flatten an empty Fisher importance profile")
    return torch.cat([value.detach().reshape(-1).float().cpu() for value in importance.values()])


def _normalized_profiles(importances: list[Importance]) -> torch.Tensor:
    profiles = torch.stack([flatten_importance_profile(importance) for importance in importances])
    return torch.nn.functional.normalize(profiles, p=2, dim=1, eps=1e-12)


def compute_fisher_discrepancy(
    importances: list[Importance],
    *,
    metric: str = "cosine",
) -> torch.Tensor:
    """Distance from each client's normalized profile to the cohort median."""
    if not importances:
        raise ValueError("FDCR discrepancy requires at least one importance profile")

    profiles = _normalized_profiles(importances)
    cohort_median = profiles.median(dim=0).values
    cohort_median = torch.nn.functional.normalize(cohort_median, p=2, dim=0, eps=1e-12)
    metric = metric.lower()

    if metric == "cosine":
        similarities = profiles @ cohort_median
        return (1.0 - similarities).clamp(min=0.0, max=2.0)
    if metric == "l2":
        return torch.linalg.vector_norm(profiles - cohort_median.unsqueeze(0), ord=2, dim=1)
    raise ValueError(f"Unsupported FDCR discrepancy_metric: {metric}")


def cluster_or_score_clients_by_discrepancy(
    importances: list[Importance],
    *,
    metric: str = "cosine",
    tau: float = 4.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return discrepancy scores and robust admission-like client weights."""
    scores = compute_fisher_discrepancy(importances, metric=metric)
    median = scores.median()
    mad = (scores - median).abs().median()
    z_scores = (scores - median) / (mad + float(eps))
    weights = torch.clamp(1.0 - z_scores / float(tau), min=0.0, max=1.0)
    return scores, weights


def _weighted_average_updates(
    updates: list[Update],
    weights: torch.Tensor,
) -> Update:
    if not updates:
        raise ValueError("Cannot average an empty update list")
    if len(updates) != int(weights.numel()):
        raise ValueError("updates and weights must have the same length")

    weights = weights.float().cpu()
    weight_sum = float(weights.sum().item())
    if weight_sum <= 0.0:
        weights = torch.ones_like(weights)
        weight_sum = float(weights.sum().item())

    averaged: Update = {}
    for key in updates[0]:
        # FDCR weights are applied to floating-point update coordinates.
        # Non-floating buffers are metadata and are left as placeholders;
        # apply_update preserves the server/global value for them.
        if not torch.is_floating_point(updates[0][key]):
            averaged[key] = updates[0][key].detach().clone()
            continue
        value = torch.zeros_like(updates[0][key])
        for update, weight in zip(updates, weights):
            value = value + update[key] * (float(weight.item()) / weight_sum)
        averaged[key] = value
    return averaged


def _cohort_importance(importances: list[Importance], weights: torch.Tensor) -> Importance:
    weights = weights.float().cpu()
    weight_sum = float(weights.sum().item())
    if weight_sum <= 0.0:
        weights = torch.ones_like(weights)
        weight_sum = float(weights.sum().item())

    cohort: Importance = {}
    for key in importances[0]:
        value = torch.zeros_like(importances[0][key], dtype=torch.float32)
        for importance, weight in zip(importances, weights):
            value = value + importance[key].float() * (float(weight.item()) / weight_sum)
        cohort[key] = value
    return cohort


def rescale_update_by_importance(
    update: Update,
    cohort_importance: Importance,
    *,
    rescale_strength: float = 0.5,
    eps: float = 1e-12,
) -> Update:
    """Rescale aggregated update using cohort-level Fisher importance."""
    strength = max(0.0, float(rescale_strength))
    rescaled: Update = {}
    for key, value in update.items():
        if key not in cohort_importance:
            rescaled[key] = value.detach().clone()
            continue
        importance = cohort_importance[key].to(device=value.device, dtype=value.dtype)
        normalized = importance / (importance.mean() + float(eps))
        factor = 1.0 + strength * (normalized - 1.0)
        factor = factor.clamp(min=max(0.0, 1.0 - strength), max=1.0 + strength)
        rescaled[key] = value.detach().clone() * factor
    return rescaled


@dataclass
class FDCRDefense:
    fisher_batches: int = 1
    discrepancy_metric: str = "cosine"
    tau: float = 4.0
    rescale_strength: float = 0.5

    @classmethod
    def from_config(cls, config: dict) -> "FDCRDefense":
        defense_cfg = config.get("defense", {})
        return cls(
            fisher_batches=int(defense_cfg.get("fisher_batches", 1)),
            discrepancy_metric=str(defense_cfg.get("discrepancy_metric", "cosine")),
            tau=float(defense_cfg.get("fdcr_tau", 4.0)),
            rescale_strength=float(defense_cfg.get("rescale_strength", 0.5)),
        )

    def aggregate(
        self,
        updates: list[Update],
        importances: list[Importance],
    ) -> Update:
        if len(updates) != len(importances):
            raise ValueError("FDCR requires one Fisher profile per client update")
        _, weights = cluster_or_score_clients_by_discrepancy(
            importances,
            metric=self.discrepancy_metric,
            tau=self.tau,
        )
        aggregated = _weighted_average_updates(updates, weights)
        cohort = _cohort_importance(importances, weights)
        return rescale_update_by_importance(
            aggregated,
            cohort,
            rescale_strength=self.rescale_strength,
        )
