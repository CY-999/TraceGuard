"""FedAvg aggregation over client updates."""

from __future__ import annotations

import torch


Update = dict[str, torch.Tensor]


def fedavg(updates: list[Update], sample_counts: list[int]) -> Update:
    if not updates:
        raise ValueError("FedAvg requires at least one update")
    if len(updates) != len(sample_counts):
        raise ValueError("updates and sample_counts must have the same length")

    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("total sample count must be positive")

    averaged: Update = {}
    for key in updates[0]:
        # Robust/FedAvg-style update aggregation is defined over floating-point
        # model updates. Non-floating buffers such as BatchNorm
        # num_batches_tracked are metadata and are excluded from numeric
        # aggregation; apply_update keeps the server value for those keys.
        if not torch.is_floating_point(updates[0][key]):
            averaged[key] = updates[0][key].detach().clone()
            continue
        value = torch.zeros_like(updates[0][key])
        for update, count in zip(updates, sample_counts):
            value = value + update[key] * (float(count) / total)
        averaged[key] = value
    return averaged


def apply_update(model: torch.nn.Module, update: Update) -> None:
    state = model.state_dict()
    new_state = {}
    for key, value in state.items():
        if key in update and torch.is_floating_point(value):
            new_state[key] = value + update[key].to(value.device)
        else:
            # Non-floating persistent buffers are metadata/counters, not
            # gradient coordinates. Preserve the server/global state.
            new_state[key] = value
    model.load_state_dict(new_state)
