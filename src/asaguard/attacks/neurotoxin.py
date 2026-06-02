"""Neurotoxin-style persistent backdoor attack support.

Neurotoxin is not a new visual trigger. It combines ordinary poisoned local
training with a parameter-selection strategy that keeps malicious updates on
low-change coordinates, which are less likely to be overwritten later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from asaguard.attacks.model_replacement import PoisonedDataset
from asaguard.attacks.triggers import (
    TriggerSpec,
    apply_configured_trigger,
    build_trigger_from_config,
)


StateDict = dict[str, torch.Tensor]
Update = dict[str, torch.Tensor]


def _trainable_parameter_tensors(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and torch.is_floating_point(parameter)
    }


def estimate_low_change_mask(
    global_state_history: list[StateDict],
    model: torch.nn.Module,
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
        raise ValueError("global_state_history must contain at least one global state")

    parameter_tensors = _trainable_parameter_tensors(model)
    if not parameter_tensors:
        return {}

    mask_ratio = max(0.0, min(float(mask_ratio), 1.0))
    history_window = max(1, int(history_window))
    states = global_state_history[-(history_window + 1) :]
    reference_state = states[-1]

    for name, parameter in parameter_tensors.items():
        if name not in reference_state:
            raise ValueError(f"Neurotoxin parameter {name!r} is missing from global state history")
        if tuple(reference_state[name].shape) != tuple(parameter.shape):
            raise ValueError(
                f"Neurotoxin parameter {name!r} shape mismatch: "
                f"history={tuple(reference_state[name].shape)}, model={tuple(parameter.shape)}"
            )

    change_scores: StateDict = {}
    if len(states) >= 2:
        for key, parameter in parameter_tensors.items():
            value = reference_state[key]
            score = torch.zeros_like(value, dtype=torch.float32, device="cpu")
            for prev_state, next_state in zip(states[:-1], states[1:]):
                if key not in prev_state or key not in next_state:
                    raise ValueError(f"Neurotoxin parameter {key!r} is missing from global state history")
                if tuple(prev_state[key].shape) != tuple(parameter.shape) or tuple(next_state[key].shape) != tuple(parameter.shape):
                    raise ValueError(f"Neurotoxin parameter {key!r} shape mismatch in global state history")
                score += (
                    next_state[key].detach().cpu().float()
                    - prev_state[key].detach().cpu().float()
                ).abs()
            change_scores[key] = score / float(len(states) - 1)
    else:
        # First round has no transition yet, so low-change coordinates cannot
        # be estimated. Keep the original semantics: all trainable parameter
        # coordinates are allowed.
        return {
            key: torch.ones_like(reference_state[key], dtype=torch.bool, device="cpu")
            for key in parameter_tensors
        }

    flat_scores = [
        score.reshape(-1)
        for score in change_scores.values()
        if score.numel() > 0
    ]
    if not flat_scores:
        return {
            key: torch.ones_like(reference_state[key], dtype=torch.bool, device="cpu")
            for key in parameter_tensors
        }

    all_scores = torch.cat(flat_scores)
    keep_count = max(1, int(round(all_scores.numel() * mask_ratio)))
    keep_count = min(keep_count, all_scores.numel())
    threshold = torch.topk(all_scores, keep_count, largest=False).values.max()

    mask: dict[str, torch.Tensor] = {}
    for key in parameter_tensors:
        if key in change_scores:
            mask[key] = change_scores[key] <= threshold
        else:
            mask[key] = torch.ones_like(reference_state[key], dtype=torch.bool, device="cpu")
    return mask


def apply_neurotoxin_mask_to_update(
    update: Update,
    mask: dict[str, torch.Tensor],
) -> Update:
    """Zero out update coordinates outside the Neurotoxin low-change mask."""
    masked_update: Update = {}
    for key, mask_tensor in mask.items():
        if key not in update:
            raise ValueError(f"Neurotoxin update is missing trainable parameter {key!r}")
        if tuple(update[key].shape) != tuple(mask_tensor.shape):
            raise ValueError(
                f"Neurotoxin mask shape mismatch for {key!r}: "
                f"update={tuple(update[key].shape)}, mask={tuple(mask_tensor.shape)}"
            )

    for key, value in update.items():
        if key not in mask:
            masked_update[key] = value
            continue
        if not torch.is_floating_point(value):
            raise ValueError(f"Neurotoxin trainable parameter update {key!r} must be floating point")
        masked_update[key] = value * mask[key].to(device=value.device, dtype=value.dtype)
    return masked_update


@dataclass
class NeurotoxinAttack:
    target_label: int = 0
    poison_ratio: float = 0.1
    trigger_type: str = "patch"
    trigger_spec: TriggerSpec | None = None
    num_malicious: int = 0
    mask_ratio: float = 0.2
    history_window: int = 5
    attack_start_round: int = 1
    attack_stop_round: int | None = None
    attack_interval: int = 1
    seed: int = 42

    @classmethod
    def from_config(cls, config: dict) -> "NeurotoxinAttack":
        attack_cfg = config.get("attack", {})
        data_cfg = config.get("dataset", {})
        trigger_spec = build_trigger_from_config(attack_cfg, data_cfg)
        attack_start_round = int(attack_cfg.get("attack_start_round", 1))
        attack_stop_round_value = attack_cfg.get("attack_stop_round")
        attack_stop_round = (
            None
            if attack_stop_round_value is None
            else int(attack_stop_round_value)
        )
        attack_interval = int(attack_cfg.get("attack_interval", 1))
        if attack_start_round < 1:
            raise ValueError("Neurotoxin attack_start_round must be >= 1")
        if attack_stop_round is not None and attack_stop_round < attack_start_round:
            raise ValueError("Neurotoxin attack_stop_round must be null or >= attack_start_round")
        if attack_interval < 1:
            raise ValueError("Neurotoxin attack_interval must be >= 1")
        return cls(
            target_label=int(attack_cfg.get("target_label", 0)),
            poison_ratio=float(attack_cfg.get("poison_ratio", 0.1)),
            trigger_type=str(attack_cfg.get("trigger_type", "patch")),
            trigger_spec=trigger_spec,
            num_malicious=int(attack_cfg.get("num_malicious", 0)),
            mask_ratio=float(attack_cfg.get("mask_ratio", 0.2)),
            history_window=int(attack_cfg.get("history_window", 5)),
            attack_start_round=attack_start_round,
            attack_stop_round=attack_stop_round,
            attack_interval=attack_interval,
            seed=int(config.get("training", {}).get("seed", 42)),
        )

    @property
    def trigger_fn(self):
        spec = self.trigger_spec
        if spec is None:
            spec = build_trigger_from_config({"trigger_type": self.trigger_type}, {})
        return lambda images: apply_configured_trigger(images, spec)

    def malicious_client_ids(self, total_clients: int) -> set[int]:
        count = max(0, min(int(self.num_malicious), int(total_clients)))
        return set(range(count))

    def is_active_round(self, round_idx: int) -> bool:
        round_idx = int(round_idx)
        if round_idx < int(self.attack_start_round):
            return False
        if self.attack_stop_round is not None and round_idx > int(self.attack_stop_round):
            return False
        return (round_idx - int(self.attack_start_round)) % int(self.attack_interval) == 0

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
        *,
        model: torch.nn.Module,
    ) -> Update:
        mask = estimate_low_change_mask(
            global_state_history,
            model,
            mask_ratio=self.mask_ratio,
            history_window=self.history_window,
        )
        return apply_neurotoxin_mask_to_update(update, mask)
