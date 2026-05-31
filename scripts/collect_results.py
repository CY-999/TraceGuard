"""Collect TRACEGuard JSONL metrics into CSV summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import pandas as pd


NUMERIC_COLUMNS = [
    "round",
    "clean_acc",
    "asr",
    "train_loss_mean",
    "traceguard_mean_risk",
    "traceguard_mean_weight",
    "num_accepted",
    "num_downweighted",
    "num_rejected",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect metrics.jsonl files into CSV summaries.")
    parser.add_argument("--results-dir", default="outputs", help="Experiment output root directory.")
    parser.add_argument("--output", default="outputs/summary.csv", help="Output CSV path.")
    return parser


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_from_path(path: Path) -> dict[str, str | None]:
    parts = [part.lower() for part in path.parts]
    known_attacks = {"model_replacement", "dba", "neurotoxin", "a3fl", "none"}
    known_defenses = {"fedavg", "multi_krum", "trimmed_mean", "flame", "flip", "fdcr", "traceguard"}
    known_datasets = {"cifar10", "cifar100", "tinyimagenet", "tiny_imagenet", "fakedata"}

    parsed = {"dataset": None, "attack": None, "defense": None, "seed": None}
    for part in parts:
        tokens = part.replace("-", "_").split("_")
        joined = part.replace("-", "_")
        if joined in known_datasets:
            parsed["dataset"] = joined
        if joined in known_attacks:
            parsed["attack"] = joined
        if joined in known_defenses:
            parsed["defense"] = joined
        if joined.startswith("seed"):
            parsed["seed"] = joined.replace("seed", "").strip("_") or None
        for token in tokens:
            if token.startswith("seed") and token != "seed":
                parsed["seed"] = token.replace("seed", "")
    return parsed


def safe_mean(values) -> float | None:  # noqa: ANN001
    if not isinstance(values, list) or not values:
        return None
    try:
        return float(mean(float(value) for value in values))
    except (TypeError, ValueError):
        return None


def summarize_file(path: Path) -> dict:
    records = read_jsonl(path)
    if not records:
        row = parse_from_path(path)
        row["path"] = str(path)
        return row

    final = records[-1]
    row = parse_from_path(path)
    row.update(
        {
            "path": str(path),
            "dataset": final.get("dataset", row.get("dataset")),
            "attack": final.get("attack", row.get("attack")),
            "defense": final.get("defense", row.get("defense")),
            "seed": final.get("seed", row.get("seed")),
            "round": final.get("round"),
            "clean_acc": final.get("clean_acc"),
            "asr": final.get("asr"),
            "train_loss_mean": final.get("train_loss_mean"),
            "traceguard_mean_risk": safe_mean(final.get("traceguard_risk_scores")),
            "traceguard_mean_weight": safe_mean(final.get("traceguard_weights")),
            "num_accepted": final.get("num_accepted"),
            "num_downweighted": final.get("num_downweighted"),
            "num_rejected": final.get("num_rejected"),
        }
    )
    return row


def averaged_output_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_avg{output.suffix}")


def write_average_summary(df: pd.DataFrame, output: Path) -> Path:
    avg_path = averaged_output_path(output)
    group_cols = ["dataset", "attack", "defense"]
    value_cols = [column for column in NUMERIC_COLUMNS if column in df.columns]
    for column in value_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df.empty:
        avg_df = pd.DataFrame(columns=group_cols)
    else:
        avg_df = (
            df.groupby(group_cols, dropna=False)[value_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        avg_df.columns = [
            "_".join(part for part in column if part)
            if isinstance(column, tuple)
            else column
            for column in avg_df.columns
        ]

    avg_path.parent.mkdir(parents=True, exist_ok=True)
    avg_df.to_csv(avg_path, index=False)
    return avg_path


def main() -> int:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    output = Path(args.output)

    metric_paths = sorted(results_dir.rglob("metrics.jsonl"))
    rows = [summarize_file(path) for path in metric_paths]
    df = pd.DataFrame(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    avg_path = write_average_summary(df, output)

    print(f"found_metrics_jsonl={len(metric_paths)}")
    print(f"wrote_csv={output}")
    print(f"wrote_averaged_csv={avg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
