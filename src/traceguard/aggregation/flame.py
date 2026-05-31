"""FLAME aggregation baseline."""

from __future__ import annotations

from dataclasses import dataclass
from warnings import warn

import torch
from sklearn.cluster import AgglomerativeClustering


Update = dict[str, torch.Tensor]


def flatten_update(update: Update) -> torch.Tensor:
    if not update:
        raise ValueError("Cannot flatten an empty update")
    return torch.cat([tensor.detach().reshape(-1).float().cpu() for tensor in update.values()])


def compute_pairwise_cosine_distances(flat_updates: torch.Tensor) -> torch.Tensor:
    if flat_updates.ndim != 2:
        raise ValueError("flat_updates must have shape NxD")
    normalized = torch.nn.functional.normalize(flat_updates.float(), p=2, dim=1, eps=1e-12)
    similarities = normalized @ normalized.T
    distances = (1.0 - similarities).clamp(min=0.0, max=2.0)
    distances.fill_diagonal_(0.0)
    return distances


def cluster_updates(
    updates: list[Update],
    *,
    cluster_method: str = "agglomerative",
    cluster_metric: str = "cosine",
) -> list[int]:
    """Cluster updates and return indices belonging to the largest cluster."""
    if not updates:
        raise ValueError("FLAME requires at least one update")
    if len(updates) <= 2:
        return list(range(len(updates)))

    method = cluster_method.lower()
    metric = cluster_metric.lower()
    if method not in {"agglomerative", "hdbscan_or_agglomerative"}:
        raise ValueError(f"Unsupported FLAME cluster_method: {cluster_method}")
    if metric != "cosine":
        raise ValueError(f"Unsupported FLAME cluster_metric: {cluster_metric}")

    try:
        flat = torch.stack([flatten_update(update) for update in updates])
        distances = compute_pairwise_cosine_distances(flat).numpy()
        model = AgglomerativeClustering(
            n_clusters=2,
            metric="precomputed",
            linkage="average",
        )
        labels = model.fit_predict(distances)
        counts = {int(label): int((labels == label).sum()) for label in set(labels)}
        main_label = max(counts, key=counts.get)
        selected = [idx for idx, label in enumerate(labels) if int(label) == int(main_label)]
        return selected or list(range(len(updates)))
    except Exception as exc:  # pragma: no cover - warning path for runtime compatibility
        warn(f"FLAME clustering failed; keeping all updates. Reason: {exc}", RuntimeWarning)
        return list(range(len(updates)))


def _update_l2_norm(update: Update) -> torch.Tensor:
    pieces = [tensor.detach().reshape(-1).float() for tensor in update.values()]
    if not pieces:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.cat(pieces), ord=2)


def clip_updates_by_norm(
    updates: list[Update],
    clip_norm: float | None = None,
) -> list[Update]:
    if not updates:
        raise ValueError("Cannot clip an empty update list")

    norms = [_update_l2_norm(update).cpu() for update in updates]
    bound = float(torch.median(torch.stack(norms)).item()) if clip_norm is None else float(clip_norm)
    if bound <= 0.0:
        return [{key: value.detach().clone() for key, value in update.items()} for update in updates]

    clipped: list[Update] = []
    for update, norm in zip(updates, norms):
        scale = min(1.0, bound / (float(norm.item()) + 1e-12))
        clipped.append({key: value.detach().clone() * scale for key, value in update.items()})
    return clipped


def _mean_updates(updates: list[Update]) -> Update:
    if not updates:
        raise ValueError("Cannot average an empty update list")
    averaged: Update = {}
    for key in updates[0]:
        value = torch.zeros_like(updates[0][key])
        for update in updates:
            value = value + update[key]
        averaged[key] = value / float(len(updates))
    return averaged


def add_gaussian_noise(
    update: Update,
    noise_std: float = 0.0,
) -> Update:
    noise_std = float(noise_std)
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    if noise_std == 0.0:
        return {key: value.detach().clone() for key, value in update.items()}
    return {
        key: value.detach().clone() + torch.randn_like(value) * noise_std
        for key, value in update.items()
    }


def flame(
    updates: list[Update],
    clip_norm: float | None = None,
    noise_std: float = 0.0,
    cluster_method: str = "agglomerative",
    cluster_metric: str = "cosine",
) -> Update:
    if not updates:
        raise ValueError("FLAME requires at least one update")

    selected_indices = cluster_updates(
        updates,
        cluster_method=cluster_method,
        cluster_metric=cluster_metric,
    )
    selected_updates = [updates[index] for index in selected_indices]
    clipped_updates = clip_updates_by_norm(selected_updates, clip_norm=clip_norm)
    aggregated = _mean_updates(clipped_updates)
    return add_gaussian_noise(aggregated, noise_std=noise_std)


@dataclass
class FLAMEAggregator:
    clip_norm: float | None = None
    noise_std: float = 0.0
    cluster_method: str = "agglomerative"
    cluster_metric: str = "cosine"

    def aggregate(self, updates: list[Update]) -> Update:
        return flame(
            updates,
            clip_norm=self.clip_norm,
            noise_std=self.noise_std,
            cluster_method=self.cluster_method,
            cluster_metric=self.cluster_metric,
        )
