"""Coordinate-wise Trimmed Mean aggregation baseline."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import torch


Update = dict[str, torch.Tensor]


def _resolve_trim_count(
    *,
    n: int,
    trim_ratio: float | None,
    num_byzantine: int | None,
) -> int:
    if n <= 0:
        raise ValueError("Trimmed Mean requires at least one update")

    if trim_ratio is not None:
        ratio = float(trim_ratio)
        if ratio < 0.0 or ratio >= 0.5:
            raise ValueError(f"trim_ratio must be in [0, 0.5), got {ratio}")
        b = int(floor(ratio * n))
    elif num_byzantine is not None:
        b = int(num_byzantine)
        if b < 0:
            raise ValueError(f"num_byzantine must be non-negative, got {b}")
    else:
        b = 0

    if 2 * b >= n:
        raise ValueError(
            f"Trimmed Mean condition not satisfied: n={n}, b={b}; require 2b < n"
        )
    return b


def trimmed_mean(
    updates: list[Update],
    trim_ratio: float | None = None,
    num_byzantine: int | None = None,
) -> Update:
    """Aggregate updates with coordinate-wise trimmed mean."""
    n = len(updates)
    b = _resolve_trim_count(
        n=n,
        trim_ratio=trim_ratio,
        num_byzantine=num_byzantine,
    )
    if not updates:
        raise ValueError("Trimmed Mean requires at least one update")

    aggregated: Update = {}
    for key in updates[0]:
        stacked = torch.stack([update[key].detach().clone() for update in updates], dim=0)
        if b == 0:
            aggregated[key] = stacked.mean(dim=0)
            continue

        sorted_values, _ = torch.sort(stacked, dim=0)
        kept = sorted_values[b : n - b]
        aggregated[key] = kept.mean(dim=0)

    return aggregated


@dataclass
class TrimmedMeanAggregator:
    trim_ratio: float | None = None
    num_byzantine: int | None = None

    def aggregate(self, updates: list[Update]) -> Update:
        return trimmed_mean(
            updates,
            trim_ratio=self.trim_ratio,
            num_byzantine=self.num_byzantine,
        )
