"""TRACEGuard weighted aggregation."""

from __future__ import annotations

import torch


Update = dict[str, torch.Tensor]


def traceguard_aggregate(
    updates: list[Update],
    weights: torch.Tensor,
    sample_counts: list[int] | None = None,
    risk_scores: torch.Tensor | None = None,
    z_scores: torch.Tensor | None = None,
    tau: float | None = None,
    eps: float = 1e-12,
) -> Update:
    """Aggregate updates with TRACEGuard admission weights.

    If all admission weights are zero, TRACEGuard rejects the full cohort and
    raises a clear error. We intentionally do not fallback to FedAvg, because
    doing so would bypass the admission decision.
    """
    if not updates:
        raise ValueError("TRACEGuard aggregation requires at least one update")
    if len(updates) != int(weights.numel()):
        raise ValueError("updates and weights must have the same length")

    weights = weights.detach().float().cpu()
    total_weight = float(weights.sum().item())
    if total_weight <= float(eps):
        risk_values = [] if risk_scores is None else [float(value) for value in risk_scores.detach().float().cpu().tolist()]
        z_values = [] if z_scores is None else [float(value) for value in z_scores.detach().float().cpu().tolist()]
        weight_values = [float(value) for value in weights.tolist()]
        raise RuntimeError(
            "TRACEGuard rejected all client updates; "
            f"risk_scores={risk_values}; "
            f"z_scores={z_values}; "
            f"admission_weights={weight_values}; "
            f"tau={tau}; "
            f"eps={eps}"
        )

    aggregated: Update = {}
    for key in updates[0]:
        # TRACEGuard admission weights apply to floating-point model updates.
        # Non-floating buffers such as BatchNorm num_batches_tracked are
        # metadata/counters and are not numerically aggregated; apply_update
        # preserves the server/global value for those keys.
        if not torch.is_floating_point(updates[0][key]):
            aggregated[key] = updates[0][key].detach().clone()
            continue
        value = torch.zeros_like(updates[0][key])
        for update, weight in zip(updates, weights):
            value = value + update[key] * (float(weight.item()) / total_weight)
        aggregated[key] = value
    return aggregated
