import subprocess
import sys

import yaml


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "traceguard.fl.run", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_help_runs():
    result = run_cli("--help")

    assert result.returncode == 0
    assert "--print-config" in result.stdout


def test_debug_print_config_runs():
    result = run_cli("--debug", "--print-config")

    assert result.returncode == 0
    config = yaml.safe_load(result.stdout)
    assert config["debug"]["enabled"] is True
    assert config["training"]["rounds"] == 1


def test_cli_dataset_and_defense_override_yaml():
    result = run_cli(
        "--config",
        "configs/default.yaml",
        "--dataset",
        "cifar100",
        "--defense",
        "traceguard",
        "--print-config",
    )

    assert result.returncode == 0
    config = yaml.safe_load(result.stdout)
    assert config["dataset"]["name"] == "cifar100"
    assert config["defense"]["name"] == "traceguard"
