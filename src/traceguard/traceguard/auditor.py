"""Update response auditor for TRACEGuard."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from traceguard.aggregation.fedavg import apply_update
from traceguard.traceguard.probe_bank import ProbeBatch


@dataclass
class AuditResult:
    client_id: int
    risk: float
    amplifications: torch.Tensor


def target_margin(logits: torch.Tensor, target_y: torch.Tensor) -> torch.Tensor:
    """M_y(x; w) = logit_y - max_{c != y} logit_c."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape NxC")
    target_y = target_y.to(device=logits.device, dtype=torch.long)
    target_logits = logits.gather(1, target_y.view(-1, 1)).squeeze(1)
    masked_logits = logits.clone()
    masked_logits.scatter_(1, target_y.view(-1, 1), float("-inf"))
    other_logits = masked_logits.max(dim=1).values
    return target_logits - other_logits


class UpdateResponseAuditor:
    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)

    @torch.no_grad()
    def _margins(
        self,
        model: torch.nn.Module,
        probes: ProbeBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        clean_x = probes.clean_x.to(self.device)
        trigger_x = probes.trigger_x.to(self.device)
        target_y = probes.target_y.to(self.device)
        clean_margin = target_margin(model(clean_x), target_y)
        trigger_margin = target_margin(model(trigger_x), target_y)
        return clean_margin.detach().cpu(), trigger_margin.detach().cpu()

    def audit_update(
        self,
        *,
        client_id: int,
        global_model: torch.nn.Module,
        update: dict[str, torch.Tensor],
        probes: ProbeBatch,
    ) -> AuditResult:
        """Audit one update using only Paired Trigger Amplification Score."""
        base_clean, base_trigger = self._margins(global_model, probes)

        shadow_model = deepcopy(global_model).to(self.device)
        apply_update(shadow_model, update)
        shadow_clean, shadow_trigger = self._margins(shadow_model, probes)

        trigger_gain = shadow_trigger - base_trigger
        clean_gain = shadow_clean - base_clean
        amplifications = trigger_gain - clean_gain
        risk = float(torch.median(amplifications).item())
        return AuditResult(
            client_id=int(client_id),
            risk=risk,
            amplifications=amplifications,
        )

    def audit_many(
        self,
        *,
        global_model: torch.nn.Module,
        updates: list[dict[str, torch.Tensor]],
        probes: ProbeBatch,
        client_ids: list[int] | None = None,
    ) -> list[AuditResult]:
        if client_ids is None:
            client_ids = list(range(len(updates)))
        return [
            self.audit_update(
                client_id=client_id,
                global_model=global_model,
                update=update,
                probes=probes,
            )
            for client_id, update in zip(client_ids, updates)
        ]
