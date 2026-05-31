"""Robust admission controller for TRACEGuard."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RobustAdmissionController:
    tau: float = 4.0
    eps: float = 1e-12

    def compute_z_scores(self, risks: torch.Tensor) -> torch.Tensor:
        risks = risks.detach().float().cpu()
        median = risks.median()
        mad = (risks - median).abs().median()
        return (risks - median) / (mad + float(self.eps))

    def compute_weights(self, risks: torch.Tensor) -> torch.Tensor:
        z_scores = self.compute_z_scores(risks)
        return torch.clamp(1.0 - z_scores / float(self.tau), min=0.0, max=1.0)
