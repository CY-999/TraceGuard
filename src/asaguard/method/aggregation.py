"""ASAGuard association-safe projection aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from asaguard.method.probe_bank import ProbeBatch


Update = dict[str, torch.Tensor]


@dataclass(frozen=True)
class VectorSpec:
    key: str
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    numel: int


@dataclass(frozen=True)
class ProjectionStats:
    ac_before: float
    ac_after: float
    projected_energy_ratio: float


@dataclass(frozen=True)
class ASAGuardResult:
    update: Update
    metrics: dict[str, float | int]


def target_margin(logits: torch.Tensor, target_y: torch.Tensor) -> torch.Tensor:
    """M_y(x; w) = logit_y - max_{c != y} logit_c."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape NxC")
    target_y = target_y.to(device=logits.device, dtype=torch.long).view(-1)
    target_logits = logits.gather(1, target_y.view(-1, 1)).squeeze(1)
    masked_logits = logits.clone()
    masked_logits.scatter_(1, target_y.view(-1, 1), float("-inf"))
    other_logits = masked_logits.max(dim=1).values
    return target_logits - other_logits


def vector_specs_from_state(state: dict[str, torch.Tensor]) -> list[VectorSpec]:
    return [
        VectorSpec(
            key=key,
            shape=value.shape,
            dtype=value.dtype,
            device=value.device,
            numel=value.numel(),
        )
        for key, value in state.items()
        if torch.is_floating_point(value)
    ]


def update_to_vector(update: Update, specs: list[VectorSpec]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for spec in specs:
        if spec.key not in update:
            raise KeyError(f"Update is missing floating key: {spec.key}")
        pieces.append(update[spec.key].detach().reshape(-1).float().cpu())
    if not pieces:
        raise ValueError("Update contains no floating-point tensors")
    return torch.cat(pieces)


def vector_to_update(vector: torch.Tensor, template_update: Update, specs: list[VectorSpec]) -> Update:
    restored: Update = {
        key: value.detach().clone()
        for key, value in template_update.items()
        if not torch.is_floating_point(value)
    }
    offset = 0
    for spec in specs:
        next_offset = offset + spec.numel
        if next_offset > int(vector.numel()):
            raise ValueError("Vector is too short for update restoration")
        value = vector[offset:next_offset].reshape(spec.shape).to(dtype=spec.dtype)
        template_value = template_update[spec.key]
        restored[spec.key] = value.to(device=template_value.device)
        offset = next_offset
    if offset != int(vector.numel()):
        raise ValueError("Vector has trailing values after update restoration")
    return restored


def _parameter_gradient_vector(
    model: torch.nn.Module,
    grads: tuple[torch.Tensor | None, ...],
    specs: list[VectorSpec],
) -> torch.Tensor:
    grad_by_name = {
        name: grad.detach().reshape(-1).float().cpu()
        for (name, _), grad in zip(model.named_parameters(), grads)
        if grad is not None
    }
    pieces: list[torch.Tensor] = []
    for spec in specs:
        grad_piece = grad_by_name.get(spec.key)
        if grad_piece is None:
            pieces.append(torch.zeros(spec.numel, dtype=torch.float32))
        else:
            pieces.append(grad_piece)
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)


def _margin_grad_vector(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_y: torch.Tensor,
    specs: list[VectorSpec],
    device: torch.device,
) -> torch.Tensor:
    parameters = tuple(model.parameters())
    logits = model(images.to(device))
    margin = target_margin(logits, target_y.to(device)).mean()
    grads = torch.autograd.grad(
        margin,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return _parameter_gradient_vector(model, grads, specs)


def estimate_sensitive_subspace(
    model: torch.nn.Module,
    probes: ProbeBatch,
    specs: list[VectorSpec],
    *,
    rank: int,
    device: torch.device | str,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, int]:
    """Estimate U_t from clean/trigger target-margin gradient differences."""
    if rank <= 0 or not specs:
        return torch.empty((sum(spec.numel for spec in specs), 0), dtype=torch.float32), 0

    device = torch.device(device)
    was_training = model.training
    model.eval()
    q_vectors: list[torch.Tensor] = []
    clean_x = probes.clean_x
    trigger_x = probes.trigger_x
    target_y = probes.target_y

    try:
        for idx in range(int(clean_x.shape[0])):
            model.zero_grad(set_to_none=True)
            clean_grad = _margin_grad_vector(
                model,
                clean_x[idx : idx + 1],
                target_y[idx : idx + 1],
                specs,
                device,
            )
            model.zero_grad(set_to_none=True)
            trigger_grad = _margin_grad_vector(
                model,
                trigger_x[idx : idx + 1],
                target_y[idx : idx + 1],
                specs,
                device,
            )
            q = trigger_grad - clean_grad
            if torch.linalg.vector_norm(q, ord=2).item() > float(eps):
                q_vectors.append(q)
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

    dim = sum(spec.numel for spec in specs)
    if not q_vectors:
        return torch.empty((dim, 0), dtype=torch.float32), 0

    q_matrix = torch.stack(q_vectors, dim=1)
    max_rank = min(int(rank), q_matrix.shape[0], q_matrix.shape[1])
    if max_rank <= 0:
        return torch.empty((dim, 0), dtype=torch.float32), len(q_vectors)

    try:
        left, _, _ = torch.linalg.svd(q_matrix, full_matrices=False)
        basis = left[:, :max_rank].contiguous()
    except RuntimeError:
        basis, _ = torch.linalg.qr(q_matrix, mode="reduced")
        basis = basis[:, :max_rank].contiguous()
    return basis.float().cpu(), len(q_vectors)


def association_coefficient(delta: torch.Tensor, basis: torch.Tensor, *, eps: float = 1e-12) -> float:
    norm = torch.linalg.vector_norm(delta.float(), ord=2)
    if basis.numel() == 0 or basis.shape[1] == 0:
        return 0.0
    projected = basis.T @ delta.float()
    return float((torch.linalg.vector_norm(projected, ord=2) / (norm + float(eps))).item())


def project_vector(delta: torch.Tensor, basis: torch.Tensor, *, eps: float = 1e-12) -> tuple[torch.Tensor, ProjectionStats]:
    delta = delta.float().cpu()
    if basis.numel() == 0 or basis.shape[1] == 0:
        return delta, ProjectionStats(
            ac_before=0.0,
            ac_after=0.0,
            projected_energy_ratio=0.0,
        )

    basis = basis.float().cpu()
    removed = basis @ (basis.T @ delta)
    projected = delta - removed
    delta_norm = torch.linalg.vector_norm(delta, ord=2)
    removed_norm = torch.linalg.vector_norm(removed, ord=2)
    return projected, ProjectionStats(
        ac_before=association_coefficient(delta, basis, eps=eps),
        ac_after=association_coefficient(projected, basis, eps=eps),
        projected_energy_ratio=float((removed_norm / (delta_norm + float(eps))).item()),
    )


def project_update(
    update: Update,
    basis: torch.Tensor,
    specs: list[VectorSpec],
    *,
    eps: float = 1e-12,
) -> tuple[Update, ProjectionStats]:
    delta = update_to_vector(update, specs)
    projected, stats = project_vector(delta, basis, eps=eps)
    return vector_to_update(projected, update, specs), stats


def _mean_updates(updates: list[Update]) -> Update:
    if not updates:
        raise ValueError("ASAGuard requires at least one update")
    averaged: Update = {}
    for key in updates[0]:
        if not torch.is_floating_point(updates[0][key]):
            averaged[key] = updates[0][key].detach().clone()
            continue
        value = torch.zeros_like(updates[0][key])
        for update in updates:
            value = value + update[key]
        averaged[key] = value / float(len(updates))
    return averaged


def asaguard_aggregate(
    model: torch.nn.Module,
    updates: list[Update],
    probes: ProbeBatch,
    *,
    rank: int,
    device: torch.device | str,
    eps: float = 1e-12,
) -> ASAGuardResult:
    """Project every client update away from U_t, then average projected updates."""
    if not updates:
        raise ValueError("ASAGuard requires at least one update")

    specs = vector_specs_from_state(model.state_dict())
    basis, num_q_vectors = estimate_sensitive_subspace(
        model,
        probes,
        specs,
        rank=int(rank),
        device=device,
        eps=eps,
    )
    projected_updates: list[Update] = []
    stats: list[ProjectionStats] = []
    for update in updates:
        projected_update, projection_stats = project_update(update, basis, specs, eps=eps)
        projected_updates.append(projected_update)
        stats.append(projection_stats)

    aggregated = _mean_updates(projected_updates)
    count = max(len(stats), 1)
    metrics: dict[str, float | int] = {
        "asaguard_ac_mean_before": sum(item.ac_before for item in stats) / count,
        "asaguard_ac_mean_after": sum(item.ac_after for item in stats) / count,
        "asaguard_projected_energy_ratio_mean": (
            sum(item.projected_energy_ratio for item in stats) / count
        ),
        "asaguard_subspace_rank": int(basis.shape[1]),
        "asaguard_num_q_vectors": int(num_q_vectors),
    }
    return ASAGuardResult(update=aggregated, metrics=metrics)
