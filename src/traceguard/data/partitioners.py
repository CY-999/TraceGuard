"""Client data partitioning."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset


def iid_partition(
    num_samples: int,
    num_clients: int,
    seed: int,
) -> list[list[int]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if num_samples < num_clients:
        raise ValueError("num_samples must be at least num_clients for IID partitioning")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    splits = np.array_split(indices, num_clients)
    return [split.astype(int).tolist() for split in splits]


def _labels_from_dataset(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        parent_labels = _labels_from_dataset(dataset.dataset)
        return parent_labels[np.asarray(dataset.indices, dtype=int)]

    if isinstance(dataset, TensorDataset):
        labels = dataset.tensors[1]
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        return np.asarray(labels, dtype=int)

    if hasattr(dataset, "targets"):
        return np.asarray(getattr(dataset, "targets"), dtype=int)

    if hasattr(dataset, "labels"):
        return np.asarray(getattr(dataset, "labels"), dtype=int)

    return np.asarray([int(dataset[idx][1]) for idx in range(len(dataset))], dtype=int)


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("partition.alpha must be positive")

    labels = np.asarray(labels, dtype=int)
    num_samples = len(labels)
    if num_samples < num_clients:
        raise ValueError("num_samples must be at least num_clients for Dirichlet partitioning")

    rng = np.random.default_rng(seed)
    all_indices = rng.permutation(num_samples)
    partitions = [[int(idx)] for idx in all_indices[:num_clients]]
    remaining_by_class: dict[int, list[int]] = {}

    for idx in all_indices[num_clients:]:
        label = int(labels[idx])
        remaining_by_class.setdefault(label, []).append(int(idx))

    for label in sorted(remaining_by_class):
        class_indices = np.asarray(remaining_by_class[label], dtype=int)
        if len(class_indices) == 0:
            continue
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = rng.multinomial(len(class_indices), proportions)
        start = 0
        for client_id, count in enumerate(counts):
            end = start + int(count)
            partitions[client_id].extend(class_indices[start:end].astype(int).tolist())
            start = end

    for partition in partitions:
        rng.shuffle(partition)

    assigned = sorted(idx for partition in partitions for idx in partition)
    if assigned != list(range(num_samples)):
        raise RuntimeError("Dirichlet partitioning dropped or duplicated samples")
    return partitions


def partition_dataset(config: dict, dataset_or_num_samples, labels=None) -> list[list[int]]:  # noqa: ANN001
    partition_type = config.get("partition", {}).get("type", "iid").lower()
    num_clients = int(config["federated"]["num_clients"])
    seed = int(config["training"]["seed"])

    if isinstance(dataset_or_num_samples, int):
        num_samples = dataset_or_num_samples
        dataset_labels = None if labels is None else np.asarray(labels, dtype=int)
    else:
        num_samples = len(dataset_or_num_samples)
        dataset_labels = _labels_from_dataset(dataset_or_num_samples) if labels is None else np.asarray(labels, dtype=int)

    if partition_type == "iid":
        return iid_partition(num_samples=num_samples, num_clients=num_clients, seed=seed)

    if partition_type == "dirichlet":
        if dataset_labels is None:
            raise ValueError("Dirichlet partitioning requires labels or a dataset with labels")
        return dirichlet_partition(
            labels=dataset_labels,
            num_clients=num_clients,
            alpha=float(config.get("partition", {}).get("alpha", 0.5)),
            seed=seed,
        )

    raise ValueError(f"Supported partition types are iid and dirichlet; got: {partition_type}")
