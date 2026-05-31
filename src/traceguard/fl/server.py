"""Minimal FedAvg server loop."""

from __future__ import annotations

from pathlib import Path
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from traceguard.aggregation.fedavg import apply_update, fedavg
from traceguard.aggregation.flame import flame
from traceguard.aggregation.multi_krum import multi_krum
from traceguard.aggregation.trimmed_mean import trimmed_mean
from traceguard.attacks.a3fl import A3FLAttack
from traceguard.attacks.dba import DBAAttack
from traceguard.attacks.model_replacement import ModelReplacementAttack
from traceguard.attacks.neurotoxin import NeurotoxinAttack
from traceguard.defenses.fdcr import FDCRDefense, estimate_fisher_importance
from traceguard.defenses.flip import FLIPDefense
from traceguard.fl.client import FLClient
from traceguard.metrics.classification import attack_success_rate, clean_accuracy
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
        self.attack = self._build_attack()
        self.malicious_client_ids = (
            self.attack.malicious_client_ids(int(config["federated"]["num_clients"]))
            if self.attack is not None
            else set()
        )
        self.global_state_history = [self._snapshot_state()]
        self.flip_defense = (
            FLIPDefense.from_config(config)
            if config.get("defense", {}).get("name", "fedavg").lower() == "flip"
            else None
        )
        self.fdcr_defense = (
            FDCRDefense.from_config(config)
            if config.get("defense", {}).get("name", "fedavg").lower() == "fdcr"
            else None
        )

    def _snapshot_state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _build_attack(self):
        attack_name = self.config.get("attack", {}).get("name", "none").lower()
        if attack_name == "none":
            return None
        if attack_name == "model_replacement":
            return ModelReplacementAttack.from_config(self.config)
        if attack_name == "dba":
            return DBAAttack.from_config(self.config)
        if attack_name == "neurotoxin":
            return NeurotoxinAttack.from_config(self.config)
        if attack_name == "a3fl":
            return A3FLAttack.from_config(self.config)
        raise ValueError(f"Unsupported attack in this stage: {attack_name}")

    def _num_workers(self) -> int:
        if platform.system().lower() == "windows":
            return 0
        return int(self.config["dataset"].get("num_workers", 0))

    def _make_client(self, client_id: int) -> FLClient:
        dataset = Subset(self.train_dataset, self.partitions[client_id])
        is_malicious = client_id in self.malicious_client_ids
        if client_id in self.malicious_client_ids and self.attack is not None:
            if hasattr(self.attack, "poison_dataset_with_model"):
                dataset = self.attack.poison_dataset_with_model(
                    dataset,
                    client_id=client_id,
                    global_model=self.model,
                    device=self.device,
                    batch_size=int(self.config["dataset"]["batch_size"]),
                    num_workers=self._num_workers(),
                )
            else:
                dataset = self.attack.poison_dataset(dataset, client_id=client_id)
        hardening = None
        if self.flip_defense is not None and not is_malicious:
            hardening = self.flip_defense.build_client_hardening(
                global_model=self.model,
                dataset=dataset,
                batch_size=int(self.config["dataset"]["batch_size"]),
                num_workers=self._num_workers(),
                device=self.device,
            )
        loader = DataLoader(
            dataset,
            batch_size=int(self.config["dataset"]["batch_size"]),
            shuffle=True,
            num_workers=self._num_workers(),
        )
        return FLClient(
            client_id=client_id,
            dataloader=loader,
            device=self.device,
            hardening=hardening,
        )

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

    def _aggregate_updates(self, results, importances=None) -> dict[str, torch.Tensor]:  # noqa: ANN001
        defense_name = self.config.get("defense", {}).get("name", "fedavg").lower()
        updates = [result.update for result in results]
        if defense_name == "fedavg":
            return fedavg(
                updates,
                [result.num_samples for result in results],
            )

        if defense_name == "flip":
            return fedavg(
                updates,
                [result.num_samples for result in results],
            )

        if defense_name == "multi_krum":
            defense_cfg = self.config.get("defense", {})
            num_byzantine = defense_cfg.get("num_byzantine")
            if num_byzantine is None:
                num_byzantine = self.config.get("attack", {}).get("num_malicious", 0)
            return multi_krum(
                updates,
                num_byzantine=int(num_byzantine),
                num_selected=defense_cfg.get("num_selected"),
            )

        if defense_name == "trimmed_mean":
            defense_cfg = self.config.get("defense", {})
            num_byzantine = defense_cfg.get("num_byzantine")
            if defense_cfg.get("trim_ratio") is None and num_byzantine is None:
                num_byzantine = self.config.get("attack", {}).get("num_malicious", 0)
            return trimmed_mean(
                updates,
                trim_ratio=defense_cfg.get("trim_ratio"),
                num_byzantine=num_byzantine,
            )

        if defense_name == "flame":
            defense_cfg = self.config.get("defense", {})
            return flame(
                updates,
                clip_norm=defense_cfg.get("clip_norm"),
                noise_std=float(defense_cfg.get("noise_std", 0.0)),
                cluster_method=str(defense_cfg.get("cluster_method", "agglomerative")),
                cluster_metric=str(defense_cfg.get("cluster_metric", "cosine")),
            )

        if defense_name == "fdcr":
            if self.fdcr_defense is None or importances is None:
                raise ValueError("FDCR aggregation requires Fisher importance profiles")
            return self.fdcr_defense.aggregate(updates, importances)

        raise ValueError(f"Unsupported defense in this stage: {defense_name}")

    def run(self) -> Path:
        output_dir = Path(self.config["project"]["output_dir"])
        log_path = output_dir / "metrics.jsonl"
        test_loader = self._test_loader()

        with JsonlWriter(log_path) as writer:
            for round_idx in range(1, int(self.config["training"]["rounds"]) + 1):
                selected_clients = self._select_clients()
                clients = [self._make_client(client_id) for client_id in selected_clients]
                importances = None
                if self.fdcr_defense is not None:
                    importances = [
                        estimate_fisher_importance(
                            self.model,
                            client.dataloader,
                            self.device,
                            fisher_batches=self.fdcr_defense.fisher_batches,
                        )
                        for client in clients
                    ]
                results = [
                    client.train(
                        self.model,
                        local_epochs=int(self.config["training"]["local_epochs"]),
                        lr=float(self.config["training"]["lr"]),
                        momentum=float(self.config["training"].get("momentum", 0.0)),
                    )
                    for client in clients
                ]
                if self.attack is not None:
                    for result in results:
                        if (
                            result.client_id in self.malicious_client_ids
                            and hasattr(self.attack, "scale_update")
                        ):
                            result.update = self.attack.scale_update(result.update)
                        if (
                            result.client_id in self.malicious_client_ids
                            and hasattr(self.attack, "mask_update")
                        ):
                            result.update = self.attack.mask_update(
                                result.update,
                                self.global_state_history,
                            )

                update = self._aggregate_updates(results, importances=importances)
                apply_update(self.model, update)
                self.global_state_history.append(self._snapshot_state())

                train_loss_mean = float(np.mean([result.train_loss for result in results]))
                acc = clean_accuracy(self.model, test_loader, self.device)
                record = {
                    "round": round_idx,
                    "clean_acc": acc,
                    "train_loss_mean": train_loss_mean,
                    "selected_clients": selected_clients,
                }
                if self.attack is not None:
                    record["asr"] = attack_success_rate(
                        self.model,
                        test_loader,
                        self.attack.trigger_fn,
                        self.attack.target_label,
                        self.device,
                    )
                writer.write(record)
                status = (
                    f"round={round_idx} clean_acc={acc:.4f} "
                    f"train_loss_mean={train_loss_mean:.4f} "
                    f"selected_clients={selected_clients}"
                )
                if "asr" in record:
                    status += f" asr={record['asr']:.4f}"
                print(status)

        return log_path
