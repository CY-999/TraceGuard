"""Multi-Krum aggregation baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch


Update = dict[str, torch.Tensor]


def flatten_update(update: Update) -> torch.Tensor:
    """Flatten a model update dict into one vector without mutating it."""
    if not update:
        raise ValueError("Cannot flatten an empty update")
    pieces = [
        tensor.detach().reshape(-1).float()
        for tensor in update.values()
        if torch.is_floating_point(tensor)
    ]
    if not pieces:
        raise ValueError("Update contains no floating-point tensors to flatten")
    return torch.cat(pieces)


def _mean_updates(updates: list[Update]) -> Update:
    if not updates:
        raise ValueError("Cannot average an empty update list")
    averaged: Update = {}
    for key in updates[0]:
        # Multi-Krum is defined over floating-point update vectors. Non-floating
        # buffers are metadata and are kept only as placeholders; apply_update
        # preserves the server/global value for them.
        if not torch.is_floating_point(updates[0][key]):
            averaged[key] = updates[0][key].detach().clone()
            continue
        value = torch.zeros_like(updates[0][key])
        for update in updates:
            value = value + update[key]
        averaged[key] = value / float(len(updates))
    return averaged


def multi_krum(
    updates: list[Update],
    num_byzantine: int,
    num_selected: int | None = None,
) -> Update:
    """Aggregate updates with Multi-Krum.

    For each update, the Krum score is the sum of squared L2 distances to its
    nearest n - f - 2 neighbors. Multi-Krum selects m lowest-score updates and
    averages them, where m defaults to n - f.
    """
    n = len(updates)
    f = int(num_byzantine)
    if n == 0:
        raise ValueError("Multi-Krum requires at least one update")
    if f < 0:
        raise ValueError("num_byzantine must be non-negative")
    if n <= 2 * f + 2:
        raise ValueError(
            f"Multi-Krum condition not satisfied: n={n}, f={f}; require n > 2f + 2"
        )

    neighbor_count = n - f - 2
    if neighbor_count <= 0:
        raise ValueError(
            f"Multi-Krum needs at least one neighbor, got n-f-2={neighbor_count}"
        )

    m = n - f if num_selected is None else int(num_selected)
    if m <= 0 or m > n:
        raise ValueError(f"num_selected must be in [1, n], got {m} for n={n}")

    flat = torch.stack([flatten_update(update) for update in updates])
    distances = torch.cdist(flat, flat, p=2).pow(2)
    distances.fill_diagonal_(float("inf"))

    nearest_distances, _ = torch.topk(
        distances,
        k=neighbor_count,
        largest=False,
        dim=1,
    )
    scores = nearest_distances.sum(dim=1)
    selected_indices = torch.topk(scores, k=m, largest=False).indices.tolist()
    selected_updates = [updates[int(index)] for index in selected_indices]
    return _mean_updates(selected_updates)


@dataclass
class MultiKrumAggregator:
    num_byzantine: int
    num_selected: int | None = None

    def aggregate(self, updates: list[Update]) -> Update:
        return multi_krum(
            updates,
            num_byzantine=self.num_byzantine,
            num_selected=self.num_selected,
        )
