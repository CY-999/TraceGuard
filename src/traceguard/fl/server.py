"""Minimal FedAvg server loop."""

from __future__ import annotations

from pathlib import Path
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from traceguard.aggregation.fedavg import apply_update, fedavg
from traceguard.fl.client import FLClient
from traceguard.metrics.classification import clean_accuracy
from traceguard.utils.jsonl import JsonlWriter


class FedAvgServer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        partitions: list[list[int]],
        config: dict,
        device: torch.device | str,
    ) -> None:
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.partitions = partitions
        self.config = config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(int(config["training"]["seed"]))

    def _num_workers(self) -> int:
        if platform.system().lower() == "windows":
            return 0
        return int(self.config["dataset"].get("num_workers", 0))

    def _make_client(self, client_id: int) -> FLClient:
        dataset = Subset(self.train_dataset, self.partitions[client_id])
        loader = DataLoader(
            dataset,
            batch_size=int(self.config["dataset"]["batch_size"]),
            shuffle=True,
            num_workers=self._num_workers(),
        )
        return FLClient(client_id=client_id, dataloader=loader, device=self.device)

    def _test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.config["dataset"].get("test_batch_size", 128)),
            shuffle=False,
            num_workers=self._num_workers(),
        )

    def _select_clients(self) -> list[int]:
        num_clients = int(self.config["federated"]["num_clients"])
        clients_per_round = int(self.config["federated"]["clients_per_round"])
        if clients_per_round > num_clients:
            raise ValueError("clients_per_round cannot exceed num_clients")
        return self.rng.choice(num_clients, size=clients_per_round, replace=False).astype(int).tolist()

    def run(self) -> Path:
        output_dir = Path(self.config["project"]["output_dir"])
        log_path = output_dir / "metrics.jsonl"
        test_loader = self._test_loader()

        with JsonlWriter(log_path) as writer:
            for round_idx in range(1, int(self.config["training"]["rounds"]) + 1):
                selected_clients = self._select_clients()
                results = [
                    self._make_client(client_id).train(
                        self.model,
                        local_epochs=int(self.config["training"]["local_epochs"]),
                        lr=float(self.config["training"]["lr"]),
                        momentum=float(self.config["training"].get("momentum", 0.0)),
                    )
                    for client_id in selected_clients
                ]

                update = fedavg(
                    [result.update for result in results],
                    [result.num_samples for result in results],
                )
                apply_update(self.model, update)

                train_loss_mean = float(np.mean([result.train_loss for result in results]))
                acc = clean_accuracy(self.model, test_loader, self.device)
                record = {
                    "round": round_idx,
                    "clean_acc": acc,
                    "train_loss_mean": train_loss_mean,
                    "selected_clients": selected_clients,
                }
                writer.write(record)
                print(
                    f"round={round_idx} clean_acc={acc:.4f} "
                    f"train_loss_mean={train_loss_mean:.4f} "
                    f"selected_clients={selected_clients}"
                )

        return log_path
