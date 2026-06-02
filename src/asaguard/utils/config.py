"""Small YAML configuration loader for asaguard."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEBUG_CONFIG_PATH = REPO_ROOT / "configs" / "debug.yaml"


Config = dict[str, Any]


def load_yaml(path: str | Path) -> Config:
    """Load a YAML mapping from disk."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Config:
    """Recursively merge two dictionaries without mutating either input."""
    merged: Config = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def set_by_dotted_key(config: Config, dotted_key: str, value: Any) -> None:
    """Set a nested configuration value such as ``dataset.name``."""
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set nested key below scalar: {part}")
        current = child
    current[parts[-1]] = value


def load_config(
    *,
    config_path: str | Path | None = None,
    debug: bool = False,
    cli_overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load default config, optional YAML override, then CLI overrides."""
    config = load_yaml(DEFAULT_CONFIG_PATH)

    selected_path: str | Path | None = config_path
    if selected_path is None and debug:
        selected_path = DEBUG_CONFIG_PATH

    if selected_path is not None:
        config = deep_merge(config, load_yaml(selected_path))

    for dotted_key, value in (cli_overrides or {}).items():
        if value is not None:
            set_by_dotted_key(config, dotted_key, value)

    if debug:
        set_by_dotted_key(config, "debug.enabled", True)

    return config


def dump_config(config: Mapping[str, Any]) -> str:
    """Render a config mapping as stable, readable YAML."""
    return yaml.safe_dump(
        dict(config),
        sort_keys=False,
        allow_unicode=True,
    )
