"""Generate or run a tiny fakedata sanity matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


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
class SanityCommand:
    attack: str
    defense: str
    command: list[str]

    def shell_text(self) -> str:
        return " ".join(self.command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate or run tiny fakedata sanity commands.")
    parser.add_argument("--attack", choices=ATTACKS)
    parser.add_argument("--defense", choices=DEFENSES)
    parser.add_argument("--run", action="store_true", help="Execute generated commands.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them. This is the default.",
    )
    return parser


def generate_commands(args: argparse.Namespace) -> list[SanityCommand]:
    attacks = [args.attack] if args.attack else list(ATTACKS)
    defenses = [args.defense] if args.defense else list(DEFENSES)
    commands: list[SanityCommand] = []
    for attack in attacks:
        for defense in defenses:
            command = [
                sys.executable,
                "-m",
                "traceguard.fl.run",
                "--dataset",
                "fakedata",
                "--attack",
                attack,
                "--defense",
                defense,
                "--rounds",
                "1",
                "--num-clients",
                "8",
                "--clients-per-round",
                "6",
            ]
            commands.append(SanityCommand(attack=attack, defense=defense, command=command))
    return commands


def main() -> int:
    args = build_parser().parse_args()
    commands = generate_commands(args)
    for item in commands:
        print(f"attack={item.attack} defense={item.defense}")
        print(f"command={item.shell_text()}")

    if not args.run:
        return 0

    for item in commands:
        subprocess.run(item.command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
