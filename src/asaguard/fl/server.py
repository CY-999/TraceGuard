"""Minimal FedAvg server loop."""

from __future__ import annotations

from pathlib import Path
import platform
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from asaguard.aggregation.fedavg import apply_update, fedavg
from asaguard.aggregation.flame import flame
from asaguard.aggregation.multi_krum import multi_krum
from asaguard.aggregation.trimmed_mean import trimmed_mean
from asaguard.attacks.a3fl import A3FLAttack
from asaguard.attacks.dba import DBAAttack
from asaguard.attacks.model_replacement import ModelReplacementAttack
from asaguard.attacks.neurotoxin import NeurotoxinAttack
from asaguard.defenses.fdcr import FDCRDefense, estimate_fisher_importance
from asaguard.defenses.flip import FLIPDefense
from asaguard.fl.client import FLClient
from asaguard.metrics.classification import attack_success_rate, clean_accuracy
from asaguard.method.aggregation import asaguard_aggregate
from asaguard.method.probe_bank import TriggerFamilyProbeBank
from asaguard.utils.jsonl import JsonlWriter


class FedAvgServer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        clean_reference_buffer: Dataset | None,
        partitions: list[list[int]],
        config: dict,
        device: torch.device | str,
    ) -> None:
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.server_clean_reference_buffer = clean_reference_buffer
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
        self.asaguard_probe_bank = (
            TriggerFamilyProbeBank.from_config(config)
            if config.get("defense", {}).get("name", "fedavg").lower() == "asaguard"
            else None
        )
        self.last_asaguard_log: dict = {}

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
        if "asaguard_ac_mean_before" in record:
            print("[ASAGuard]")
            print(f"  AC before   : {float(record['asaguard_ac_mean_before']):.4f}")
            print(f"  AC after    : {float(record['asaguard_ac_mean_after']):.4f}")
            print(f"  PER mean    : {float(record['asaguard_projected_energy_ratio_mean']):.4f}")
            print(f"  rank / q    : {record.get('asaguard_subspace_rank')} / {record.get('asaguard_num_q_vectors')}")
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

    def _make_client(self, client_id: int, *, attack_active: bool = True) -> FLClient:
        dataset = Subset(self.train_dataset, self.partitions[client_id])
        is_malicious = bool(attack_active and client_id in self.malicious_client_ids)
        if is_malicious and self.attack is not None:
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

    def _attack_schedule(self) -> tuple[int, int | None, int]:
        if self.attack is not None and hasattr(self.attack, "attack_start_round"):
            return (
                int(self.attack.attack_start_round),
                self.attack.attack_stop_round,
                int(self.attack.attack_interval),
            )
        attack_cfg = self.config.get("attack", {})
        return (
            int(attack_cfg.get("attack_start_round", 1)),
            attack_cfg.get("attack_stop_round"),
            int(attack_cfg.get("attack_interval", 1)),
        )

    def _is_attack_active(self, round_idx: int) -> bool:
        if self.attack is None:
            return False
        if hasattr(self.attack, "is_active_round"):
            return bool(self.attack.is_active_round(round_idx))
        return True

    def _prepare_round_attack(self, selected_clients: list[int], round_idx: int, attack_active: bool) -> None:
        if not attack_active or self.attack is None:
            return
        if not hasattr(self.attack, "prepare_round_trigger"):
            return
        selected_malicious_datasets = [
            Subset(self.train_dataset, self.partitions[client_id])
            for client_id in selected_clients
            if client_id in self.malicious_client_ids
        ]
        if not selected_malicious_datasets:
            return
        self.attack.prepare_round_trigger(
            self.model,
            selected_malicious_datasets,
            round_idx=round_idx,
            device=self.device,
            batch_size=int(self.config["dataset"]["batch_size"]),
            num_workers=self._num_workers(),
        )

    def _model_replacement_aggregation_weights(self, results) -> dict[int, float]:  # noqa: ANN001
        if not results:
            return {}

        defense_name = self.config.get("defense", {}).get("name", "fedavg").lower()
        if defense_name in {"fedavg", "flip"}:
            total_samples = float(sum(result.num_samples for result in results))
            if total_samples <= 0.0:
                raise ValueError("Model Replacement auto scale requires positive sample counts")
            return {
                int(result.client_id): float(result.num_samples) / total_samples
                for result in results
            }

        uniform_weight = 1.0 / float(len(results))
        return {int(result.client_id): uniform_weight for result in results}

    def _aggregate_updates(self, results, importances=None, round_idx: int = 0) -> dict[str, torch.Tensor]:  # noqa: ANN001
        defense_name = self.config.get("defense", {}).get("name", "fedavg").lower()
        updates = [result.update for result in results]
        self.last_asaguard_log = {}
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
            multi_krum_cfg = self.config.get("multi_krum", {})
            num_byzantine = self._round_byzantine_bound(multi_krum_cfg.get("num_byzantine"))
            return multi_krum(
                updates,
                num_byzantine=int(num_byzantine),
                num_selected=multi_krum_cfg.get("num_selected"),
            )

        if defense_name == "trimmed_mean":
            trimmed_mean_cfg = self.config.get("trimmed_mean", {})
            num_byzantine = trimmed_mean_cfg.get("num_byzantine")
            if trimmed_mean_cfg.get("trim_ratio") is None and num_byzantine is None:
                num_byzantine = self._round_byzantine_bound(None)
            return trimmed_mean(
                updates,
                trim_ratio=trimmed_mean_cfg.get("trim_ratio"),
                num_byzantine=num_byzantine,
            )

        if defense_name == "flame":
            flame_cfg = self.config.get("flame", {})
            return flame(
                updates,
                clip_norm=flame_cfg.get("clip_norm"),
                noise_std=float(flame_cfg.get("noise_std", 0.0)),
                cluster_method=str(flame_cfg.get("cluster_method", "agglomerative")),
                cluster_metric=str(flame_cfg.get("cluster_metric", "cosine")),
            )

        if defense_name == "fdcr":
            if self.fdcr_defense is None or importances is None:
                raise ValueError("FDCR aggregation requires Fisher importance profiles")
            return self.fdcr_defense.aggregate(updates, importances)

        if defense_name == "asaguard":
            if self.asaguard_probe_bank is None:
                raise ValueError("ASAGuard components were not initialized")
            if self.server_clean_reference_buffer is None or len(self.server_clean_reference_buffer) == 0:
                raise ValueError("ASAGuard requires a positive server clean reference buffer size.")
            probes = self.asaguard_probe_bank.sample(
                self.server_clean_reference_buffer,
                round_idx=round_idx,
                model=self.model,
                device=self.device,
            )
            asaguard_cfg = self.config.get("asaguard", {})
            result = asaguard_aggregate(
                self.model,
                updates,
                probes,
                rank=int(asaguard_cfg.get("subspace_rank", 4)),
                device=self.device,
                eps=float(asaguard_cfg.get("eps", 1e-12)),
            )
            self.last_asaguard_log = result.metrics
            return result.update

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
                attack_active = self._is_attack_active(round_idx)
                self._prepare_round_attack(selected_clients, round_idx, attack_active)
                clients = [
                    self._make_client(client_id, attack_active=attack_active)
                    for client_id in selected_clients
                ]
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
                    needs_replacement_scaling = hasattr(self.attack, "scale_update")
                    aggregation_weights = (
                        self._model_replacement_aggregation_weights(results)
                        if needs_replacement_scaling
                        else {}
                    )
                    selected_malicious_count = (
                        sum(1 for result in results if result.client_id in self.malicious_client_ids)
                        if needs_replacement_scaling
                        else 0
                    )
                    for result in results:
                        if (
                            result.client_id in self.malicious_client_ids
                            and needs_replacement_scaling
                        ):
                            scale_factor = None
                            if hasattr(self.attack, "scale_factor_for_client"):
                                scale_factor = self.attack.scale_factor_for_client(
                                    aggregation_weight=aggregation_weights[int(result.client_id)],
                                    num_selected_malicious=selected_malicious_count,
                                )
                            result.update = self.attack.scale_update(
                                result.update,
                                scale_factor=scale_factor,
                            )
                        if (
                            result.client_id in self.malicious_client_ids
                            and hasattr(self.attack, "mask_update")
                            and attack_active
                        ):
                            result.update = self.attack.mask_update(
                                result.update,
                                self.global_state_history,
                                model=self.model,
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
                attack_start_round, attack_stop_round, attack_interval = self._attack_schedule()
                record = {
                    "dataset": self.config.get("dataset", {}).get("name"),
                    "attack": self.config.get("attack", {}).get("name"),
                    "defense": self.config.get("defense", {}).get("name"),
                    "seed": self.config.get("training", {}).get("seed"),
                    "round": round_idx,
                    "attack_active": attack_active,
                    "attack_start_round": attack_start_round,
                    "attack_stop_round": attack_stop_round,
                    "attack_interval": attack_interval,
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
                if self.last_asaguard_log:
                    record.update(self.last_asaguard_log)
                writer.write(record)
                self._print_round_log(record, total_rounds)

        print("[Saved]")
        print(f"  metrics     : {log_path}")
        print("=" * 60)
        return log_path
