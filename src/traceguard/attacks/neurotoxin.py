"""Neurotoxin-style persistent backdoor attack support.

Neurotoxin is not a new visual trigger. It combines ordinary poisoned local
training with a parameter-selection strategy that keeps malicious updates on
low-change coordinates, which are less likely to be overwritten later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from traceguard.attacks.model_replacement import PoisonedDataset, get_trigger_fn


StateDict = dict[str, torch.Tensor]
Update = dict[str, torch.Tensor]


def estimate_low_change_mask(
    global_state_history: list[StateDict],
    *,
    mask_ratio: float = 0.2,
    history_window: int = 5,
) -> dict[str, torch.Tensor]:
    """Estimate a binary mask for low-change parameter coordinates.

    The score for each coordinate is the average absolute parameter movement
    across the most recent global model transitions. Coordinates with the
    smallest scores are selected.
    """
    if not global_state_history:
        raise ValueError("global_state_history must contain at least one state dict")

    mask_ratio = max(0.0, min(float(mask_ratio), 1.0))
    history_window = max(1, int(history_window))
    states = global_state_history[-(history_window + 1) :]
    reference_state = states[-1]

    change_scores: StateDict = {}
    if len(states) >= 2:
        for key in reference_state:
            value = reference_state[key]
            if not torch.is_floating_point(value):
                continue
            score = torch.zeros_like(value, dtype=torch.float32, device="cpu")
            for prev_state, next_state in zip(states[:-1], states[1:]):
                score += (
                    next_state[key].detach().cpu().float()
                    - prev_state[key].detach().cpu().float()
                ).abs()
            change_scores[key] = score / float(len(states) - 1)
    else:
        for key, value in reference_state.items():
            if torch.is_floating_point(value):
                change_scores[key] = torch.zeros_like(value, dtype=torch.float32, device="cpu")

    flat_scores = [
        score.reshape(-1)
        for score in change_scores.values()
        if score.numel() > 0
    ]
    if not flat_scores:
        return {key: torch.ones_like(value, dtype=torch.bool, device="cpu") for key, value in reference_state.items()}

    all_scores = torch.cat(flat_scores)
    keep_count = max(1, int(round(all_scores.numel() * mask_ratio)))
    keep_count = min(keep_count, all_scores.numel())
    threshold = torch.topk(all_scores, keep_count, largest=False).values.max()

    mask: dict[str, torch.Tensor] = {}
    for key, value in reference_state.items():
        if key in change_scores:
            mask[key] = change_scores[key] <= threshold
        else:
            mask[key] = torch.ones_like(value, dtype=torch.bool, device="cpu")
    return mask


def apply_neurotoxin_mask_to_update(
    update: Update,
    mask: dict[str, torch.Tensor],
) -> Update:
    """Zero out update coordinates outside the Neurotoxin low-change mask."""
    masked_update: Update = {}
    for key, value in update.items():
        if key not in mask:
            masked_update[key] = value
            continue
        masked_update[key] = value * mask[key].to(device=value.device, dtype=value.dtype)
    return masked_update


@dataclass
class NeurotoxinAttack:
    target_label: int = 0
    poison_ratio: float = 0.1
    trigger_type: str = "patch"
    num_malicious: int = 0
    mask_ratio: float = 0.2
    history_window: int = 5
    seed: int = 42

    @classmethod
    def from_config(cls, config: dict) -> "NeurotoxinAttack":
        attack_cfg = config.get("attack", {})
        return cls(
            target_label=int(attack_cfg.get("target_label", 0)),
            poison_ratio=float(attack_cfg.get("poison_ratio", 0.1)),
            trigger_type=str(attack_cfg.get("trigger_type", "patch")),
            num_malicious=int(attack_cfg.get("num_malicious", 0)),
            mask_ratio=float(attack_cfg.get("mask_ratio", 0.2)),
            history_window=int(attack_cfg.get("history_window", 5)),
            seed=int(config.get("training", {}).get("seed", 42)),
        )

    @property
    def trigger_fn(self):
        return get_trigger_fn(self.trigger_type)

    def malicious_client_ids(self, total_clients: int) -> set[int]:
        count = max(0, min(int(self.num_malicious), int(total_clients)))
        return set(range(count))

    def poison_dataset(self, dataset: Dataset, *, client_id: int) -> Dataset:
        return PoisonedDataset(
            dataset,
            target_label=self.target_label,
            poison_ratio=self.poison_ratio,
            trigger_fn=self.trigger_fn,
            seed=self.seed + int(client_id),
        )

    def mask_update(
        self,
        update: Update,
        global_state_history: list[StateDict],
    ) -> Update:
        mask = estimate_low_change_mask(
            global_state_history,
            mask_ratio=self.mask_ratio,
            history_window=self.history_window,
        )
        return apply_neurotoxin_mask_to_update(update, mask)
