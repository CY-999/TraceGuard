"""Minimal FedAvg server loop."""

from __future__ import annotations

from pathlib import Path
import platform
import math

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
from traceguard.traceguard.admission import RobustAdmissionController
from traceguard.traceguard.aggregation import traceguard_aggregate
from traceguard.traceguard.auditor import UpdateResponseAuditor
from traceguard.traceguard.probe_bank import TriggerFamilyProbeBank
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
        self.traceguard_probe_bank = (
            TriggerFamilyProbeBank.from_config(config)
            if config.get("defense", {}).get("name", "fedavg").lower() == "traceguard"
            else None
        )
        self.traceguard_auditor = (
            UpdateResponseAuditor(device=self.device)
            if self.traceguard_probe_bank is not None
            else None
        )
        self.traceguard_admission = (
            RobustAdmissionController(
                tau=float(config.get("traceguard", {}).get("tau", config.get("defense", {}).get("tau", 4.0)))
            )
            if self.traceguard_probe_bank is not None
            else None
        )
        self.last_traceguard_log: dict = {}

    def _output_dir(self) -> Path:
        return (
            Path(self.config["project"].get("output_dir", "outputs"))
            / str(self.config.get("dataset", {}).get("name", "unknown_dataset"))
            / str(self.config.get("attack", {}).get("name", "none"))
            / str(self.config.get("defense", {}).get("name", "fedavg"))
            / f"seed_{self.config.get('training', {}).get('seed', 'unknown')}"
        )

    def _format_clients(self, clients: list[int]) -> str:
        visible = clients[:10]
        suffix = "" if len(clients) <= 10 else f", ...(+{len(clients) - 10} more)"
        return f"[{', '.join(str(client) for client in visible)}{suffix}]"

    def _print_run_header(self, output_dir: Path) -> None:
        print("=" * 60)
        print("[Run]")
        print(f"  dataset     : {self.config.get('dataset', {}).get('name')}")
        print(f"  attack      : {self.config.get('attack', {}).get('name')}")
        print(f"  defense     : {self.config.get('defense', {}).get('name')}")
        print(f"  model       : {self.config.get('model', {}).get('name')}")
        print(f"  seed        : {self.config.get('training', {}).get('seed')}")
        print(f"  output_dir  : {output_dir}")
        print("-" * 60)
        print("[Config]")
        print(
            "  clients     : "
            f"{self.config.get('federated', {}).get('clients_per_round')} / "
            f"{self.config.get('federated', {}).get('num_clients')} per round"
        )
        print(f"  rounds      : {self.config.get('training', {}).get('rounds')}")
        print(f"  local_epoch : {self.config.get('training', {}).get('local_epochs')}")
        print(f"  batch_size  : {self.config.get('dataset', {}).get('batch_size')}")
        print("-" * 60)

    def _print_round_log(self, record: dict, total_rounds: int) -> None:
        print(f"[Round {int(record['round']):03d}/{int(total_rounds):03d}]")
        print(f"  ACC         : {float(record['clean_acc']) * 100.0:.2f}%")
        if "asr" in record:
            print(f"  ASR         : {float(record['asr']) * 100.0:.2f}%")
        print(f"  Loss        : {float(record['train_loss_mean']):.4f}")
        print(f"  Selected    : {self._format_clients(record['selected_clients'])}")
        if "traceguard_weights" in record:
            mean_weight = sum(record["traceguard_weights"]) / max(len(record["traceguard_weights"]), 1)
            mean_risk = sum(record["traceguard_risk_scores"]) / max(len(record["traceguard_risk_scores"]), 1)
            print("[TRACEGuard]")
            print(f"  accepted    : {record.get('num_accepted')}")
            print(f"  downweighted: {record.get('num_downweighted')}")
            print(f"  rejected    : {record.get('num_rejected')}")
            print(f"  mean_weight : {mean_weight:.4f}")
            print(f"  mean_risk   : {mean_risk:.4f}")
        print("-" * 60)

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

    def _aggregate_updates(self, results, importances=None, round_idx: int = 0) -> dict[str, torch.Tensor]:  # noqa: ANN001
        defense_name = self.config.get("defense", {}).get("name", "fedavg").lower()
        updates = [result.update for result in results]
        self.last_traceguard_log = {}
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
            num_byzantine = self._round_byzantine_bound(defense_cfg.get("num_byzantine"))
            return multi_krum(
                updates,
                num_byzantine=int(num_byzantine),
                num_selected=defense_cfg.get("num_selected"),
            )

        if defense_name == "trimmed_mean":
            defense_cfg = self.config.get("defense", {})
            num_byzantine = defense_cfg.get("num_byzantine")
            if defense_cfg.get("trim_ratio") is None and num_byzantine is None:
                num_byzantine = self._round_byzantine_bound(None)
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

        if defense_name == "traceguard":
            if (
                self.traceguard_probe_bank is None
                or self.traceguard_auditor is None
                or self.traceguard_admission is None
            ):
                raise ValueError("TRACEGuard components were not initialized")
            probes = self.traceguard_probe_bank.sample(
                self.test_dataset,
                round_idx=round_idx,
                model=self.model,
                device=self.device,
            )
            client_ids = [result.client_id for result in results]
            audit_results = self.traceguard_auditor.audit_many(
                global_model=self.model,
                updates=updates,
                probes=probes,
                client_ids=client_ids,
            )
            risks = torch.tensor([result.risk for result in audit_results], dtype=torch.float32)
            z_scores = self.traceguard_admission.compute_z_scores(risks)
            weights = self.traceguard_admission.compute_weights(risks)
            accepted = int((weights >= 1.0 - 1e-12).sum().item())
            rejected = int((weights <= 1e-12).sum().item())
            downweighted = int(((weights > 1e-12) & (weights < 1.0 - 1e-12)).sum().item())
            self.last_traceguard_log = {
                "traceguard_risk_scores": [float(value) for value in risks.tolist()],
                "traceguard_z_scores": [float(value) for value in z_scores.tolist()],
                "traceguard_weights": [float(value) for value in weights.tolist()],
                "num_accepted": accepted,
                "num_downweighted": downweighted,
                "num_rejected": rejected,
            }
            return traceguard_aggregate(
                updates,
                weights,
                sample_counts=[result.num_samples for result in results],
                risk_scores=risks,
                z_scores=z_scores,
                tau=self.traceguard_admission.tau,
                eps=self.traceguard_admission.eps,
            )

        raise ValueError(f"Unsupported defense in this stage: {defense_name}")

    def _round_byzantine_bound(self, configured_value) -> int:  # noqa: ANN001
        if configured_value is not None:
            return int(configured_value)

        total_clients = int(self.config.get("federated", {}).get("num_clients", 0))
        clients_per_round = int(self.config.get("federated", {}).get("clients_per_round", 0))
        num_malicious = int(self.config.get("attack", {}).get("num_malicious", 0))
        if total_clients <= 0 or clients_per_round <= 0:
            raise ValueError("Cannot estimate per-round Byzantine bound without positive federated client counts")
        malicious_ratio = num_malicious / float(total_clients)
        return int(math.ceil(malicious_ratio * clients_per_round))

    def run(self) -> Path:
        output_dir = self._output_dir()
        log_path = output_dir / "metrics.jsonl"
        test_loader = self._test_loader()
        total_rounds = int(self.config["training"]["rounds"])
        self._print_run_header(output_dir)

        with JsonlWriter(log_path) as writer:
            for round_idx in range(1, total_rounds + 1):
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

                update = self._aggregate_updates(
                    results,
                    importances=importances,
                    round_idx=round_idx,
                )
                apply_update(self.model, update)
                self.global_state_history.append(self._snapshot_state())

                train_loss_mean = float(np.mean([result.train_loss for result in results]))
                acc = clean_accuracy(self.model, test_loader, self.device)
                record = {
                    "dataset": self.config.get("dataset", {}).get("name"),
                    "attack": self.config.get("attack", {}).get("name"),
                    "defense": self.config.get("defense", {}).get("name"),
                    "seed": self.config.get("training", {}).get("seed"),
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
                if self.last_traceguard_log:
                    record.update(self.last_traceguard_log)
                writer.write(record)
                self._print_round_log(record, total_rounds)

        print("[Saved]")
        print(f"  metrics     : {log_path}")
        print("=" * 60)
        return log_path
