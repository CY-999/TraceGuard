"""JSONL writing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        self._handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        self.close()
