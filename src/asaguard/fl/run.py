"""Minimal FedAvg CLI for ASAGuard."""

from __future__ import annotations

import argparse
from typing import Sequence

import torch

from asaguard.data.datasets import load_datasets
from asaguard.data.partitioners import build_server_reference_split, partition_dataset
from asaguard.fl.server import FedAvgServer
from asaguard.models.cnn import build_model
from asaguard.utils.config import dump_config, load_config
from asaguard.utils.seed import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m asaguard.fl.run",
        description="ASAGuard minimal federated learning CLI scaffold.",
    )
    parser.add_argument("--config", help="Path to a YAML config override.")
    parser.add_argument("--dataset", help="Dataset name, e.g. cifar10.")
    parser.add_argument("--attack", help="Attack name. Default config uses none.")
    parser.add_argument("--defense", help="Defense name. Default config uses fedavg.")
    parser.add_argument("--num-malicious", type=int, help="Number of malicious clients.")
    parser.add_argument(
        "--num-byzantine",
        type=int,
        help="Byzantine client bound for Multi-Krum smoke/debug runs.",
    )
    parser.add_argument("--rounds", type=int, help="Number of FL rounds.")
    parser.add_argument("--num-clients", type=int, help="Total number of clients.")
    parser.add_argument(
        "--clients-per-round",
        type=int,
        help="Number of selected clients per round.",
    )
    parser.add_argument("--seed", type=int, help="Deterministic seed.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use configs/debug.yaml when --config is not provided.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the final merged config and exit.",
    )
    return parser


def cli_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset.name": args.dataset,
        "attack.name": args.attack,
        "attack.num_malicious": args.num_malicious,
        "defense.name": args.defense,
        "multi_krum.num_byzantine": args.num_byzantine,
        "training.rounds": args.rounds,
        "federated.num_clients": args.num_clients,
        "federated.clients_per_round": args.clients_per_round,
        "training.seed": args.seed,
    }


def apply_debug_limits(config: dict) -> None:
    if not config.get("debug", {}).get("enabled", False):
        return

    dataset = config.setdefault("dataset", {})
    federated = config.setdefault("federated", {})
    training = config.setdefault("training", {})

    federated["num_clients"] = min(int(federated.get("num_clients", 5)), 5)
    federated["clients_per_round"] = min(
        int(federated.get("clients_per_round", 3)),
        int(federated["num_clients"]),
        3,
    )
    training["rounds"] = min(int(training.get("rounds", 1)), 1)
    training["local_epochs"] = min(int(training.get("local_epochs", 1)), 1)

    max_train = int(federated["num_clients"]) * 64
    current_train = dataset.get("max_train_samples")
    dataset["max_train_samples"] = min(
        int(current_train) if current_train is not None else max_train,
        max_train,
    )

    current_test = dataset.get("max_test_samples")
    dataset["max_test_samples"] = min(
        int(current_test) if current_test is not None else 256,
        256,
    )
    dataset["num_workers"] = 0
    dataset["download"] = False
    dataset.setdefault("fake_data_on_missing", True)
    config["asaguard_reference_size"] = min(
        int(config.get("asaguard_reference_size", 32)),
        32,
    )


def normalize_config_names(config: dict) -> None:
    for section, key in (
        ("dataset", "name"),
        ("attack", "name"),
        ("defense", "name"),
        ("partition", "type"),
    ):
        value = config.get(section, {}).get(key)
        if isinstance(value, str):
            config[section][key] = value.lower()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(
        config_path=args.config,
        debug=args.debug,
        cli_overrides=cli_overrides_from_args(args),
    )
    apply_debug_limits(config)
    normalize_config_names(config)

    if args.print_config:
        print(dump_config(config), end="")
        return 0

    if config["attack"]["name"] not in {"none", "model_replacement", "dba", "neurotoxin", "a3fl"}:
        raise ValueError(
            "Only attack=none, attack=model_replacement, attack=dba, attack=neurotoxin, and attack=a3fl are supported in this stage."
        )
    if config["defense"]["name"] not in {"fedavg", "multi_krum", "trimmed_mean", "flame", "flip", "fdcr", "asaguard"}:
        raise ValueError(
            "Only defense=fedavg, defense=multi_krum, defense=trimmed_mean, defense=flame, defense=flip, defense=fdcr, and defense=asaguard are supported in this stage"
        )

    seed_everything(int(config["training"]["seed"]))
    train_dataset, test_dataset = load_datasets(config)
    reference_seed = config.get("asaguard_reference_seed")
    if reference_seed is None:
        reference_seed = int(config["training"]["seed"])
    reference_split = build_server_reference_split(
        train_dataset,
        reference_size=int(config.get("asaguard_reference_size", 256)),
        seed=int(reference_seed),
        stratified=bool(config.get("asaguard_reference_stratified", True)),
    )
    if (
        config["defense"]["name"] == "asaguard"
        and len(reference_split.server_clean_reference_buffer) == 0
    ):
        raise ValueError("ASAGuard requires a positive server clean reference buffer size.")

    client_train_dataset = reference_split.remaining_client_train_dataset
    partitions = partition_dataset(config, client_train_dataset)
    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    server = FedAvgServer(
        model=model,
        train_dataset=client_train_dataset,
        test_dataset=test_dataset,
        clean_reference_buffer=reference_split.server_clean_reference_buffer,
        partitions=partitions,
        config=config,
        device=device,
    )
    log_path = server.run()
    print(f"metrics_jsonl={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
