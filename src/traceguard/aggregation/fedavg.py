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
        value = torch.zeros_like(updates[0][key])
        for update, count in zip(updates, sample_counts):
            value = value + update[key] * (float(count) / total)
        averaged[key] = value
    return averaged


def apply_update(model: torch.nn.Module, update: Update) -> None:
    state = model.state_dict()
    new_state = {key: value + update[key].to(value.device) for key, value in state.items()}
    model.load_state_dict(new_state)
