"""Minimal TRACEGuard CLI.

This entry point only loads and prints configuration. Training, attacks,
defense baselines, and TRACEGuard auditing are intentionally not implemented
in this scaffold.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from traceguard.utils.config import dump_config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traceguard.fl.run",
        description="TRACEGuard minimal federated learning CLI scaffold.",
    )
    parser.add_argument("--config", help="Path to a YAML config override.")
    parser.add_argument("--dataset", help="Dataset name, e.g. cifar10.")
    parser.add_argument("--attack", help="Attack name. Default config uses none.")
    parser.add_argument("--defense", help="Defense name. Default config uses fedavg.")
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
        "defense.name": args.defense,
        "training.rounds": args.rounds,
        "federated.num_clients": args.num_clients,
        "federated.clients_per_round": args.clients_per_round,
        "training.seed": args.seed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(
        config_path=args.config,
        debug=args.debug,
        cli_overrides=cli_overrides_from_args(args),
    )

    if args.print_config:
        print(dump_config(config), end="")
        return 0

    print("TRACEGuard scaffold: configuration loaded. Training is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
