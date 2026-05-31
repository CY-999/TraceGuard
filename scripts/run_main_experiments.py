"""Generate or run TRACEGuard main experiment commands."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


DATASETS = ("cifar10", "cifar100", "tinyimagenet")
ATTACKS = ("model_replacement", "dba", "neurotoxin", "a3fl")
DEFENSES = (
    "fedavg",
    "multi_krum",
    "trimmed_mean",
    "flame",
    "flip",
    "fdcr",
    "traceguard",
)


@dataclass(frozen=True)
class ExperimentCommand:
    dataset: str
    attack: str
    defense: str
    seed: int
    command: list[str]

    def shell_text(self) -> str:
        return " ".join(self.command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or run TRACEGuard main experiment matrix commands.",
    )
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--attack", choices=ATTACKS)
    parser.add_argument("--defense", choices=DEFENSES)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them. This is the default.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute generated commands. Without this flag, commands are only printed.",
    )
    return parser


def config_path(dataset: str, attack: str) -> str:
    return f"configs/experiments/{dataset}_{attack}.yaml"


def generate_commands(args: argparse.Namespace) -> list[ExperimentCommand]:
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    attacks = [args.attack] if args.attack else list(ATTACKS)
    defenses = [args.defense] if args.defense else list(DEFENSES)

    commands: list[ExperimentCommand] = []
    for dataset in datasets:
        for attack in attacks:
            for defense in defenses:
                command = [
                    sys.executable,
                    "-m",
                    "traceguard.fl.run",
                    "--config",
                    config_path(dataset, attack),
                    "--defense",
                    defense,
                    "--seed",
                    str(args.seed),
                ]
                commands.append(
                    ExperimentCommand(
                        dataset=dataset,
                        attack=attack,
                        defense=defense,
                        seed=args.seed,
                        command=command,
                    )
                )
    return commands


def print_command(item: ExperimentCommand) -> None:
    print(
        f"dataset={item.dataset} attack={item.attack} "
        f"defense={item.defense} seed={item.seed}"
    )
    print(f"command={item.shell_text()}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    commands = generate_commands(args)

    for item in commands:
        print_command(item)

    if not args.run:
        return 0

    for item in commands:
        subprocess.run(item.command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
