"""Generate paper-ready TRACEGuard result tables and curves from JSONL metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


EXPECTED_ROUNDS = {
    "cifar10": 200,
    "cifar100": 300,
    "tinyimagenet": 300,
    "tiny_imagenet": 300,
}

DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tinyimagenet": "Tiny-ImageNet",
    "tiny_imagenet": "Tiny-ImageNet",
}

ATTACK_LABELS = {
    "model_replacement": "Model Replacement",
    "dba": "DBA",
    "neurotoxin": "Neurotoxin",
    "a3fl": "A3FL",
    "none": "None",
}

DEFENSE_ORDER = [
    "fedavg",
    "multi_krum",
    "trimmed_mean",
    "flame",
    "flip",
    "fdcr",
    "traceguard",
]

DEFENSE_LABELS = {
    "fedavg": "FedAvg",
    "multi_krum": "Multi-Krum",
    "trimmed_mean": "Trimmed Mean",
    "flame": "FLAME",
    "flip": "FLIP",
    "fdcr": "FDCR",
    "traceguard": "TRACEGuard",
}

DEFENSE_COLORS = {
    "fedavg": "#4C78A8",
    "multi_krum": "#F58518",
    "trimmed_mean": "#E45756",
    "flame": "#72B7B2",
    "flip": "#54A24B",
    "fdcr": "#B279A2",
    "traceguard": "#000000",
}

KNOWN_ATTACKS = set(ATTACK_LABELS)
KNOWN_DATASETS = set(DATASET_LABELS) | {"fakedata"}
KNOWN_DEFENSES = set(DEFENSE_LABELS)


@dataclass(frozen=True)
class RunMetrics:
    dataset: str
    attack: str
    defense: str
    seed: str
    path: Path
    records: tuple[dict[str, Any], ...]
    final_round: int | None
    expected_round: int | None
    status: str

    @property
    def complete(self) -> bool:
        return self.status == "complete"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready TRACEGuard tables and curves from metrics.jsonl files.",
    )
    parser.add_argument("--results-dir", default="outputs", help="Experiment output root.")
    parser.add_argument(
        "--analysis-dir",
        default="outputs/analysis",
        help="Root directory for paper-ready analysis outputs.",
    )
    parser.add_argument("--dataset", help="Optional dataset filter, e.g. cifar10.")
    parser.add_argument("--attack", help="Optional attack filter, e.g. model_replacement.")
    parser.add_argument(
        "--last-k",
        type=int,
        default=10,
        help="Number of final raw rounds used for table statistics.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Rolling-mean window used only for plotted curves.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
    return parser


def normalize_name(value: str | None) -> str | None:
    if value is None:
        return None
    return value.lower().replace("-", "_")


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                return records, f"malformed JSON at line {line_no}: {exc}"
    return records, None


def parse_from_path(path: Path) -> dict[str, str | None]:
    parsed: dict[str, str | None] = {
        "dataset": None,
        "attack": None,
        "defense": None,
        "seed": None,
    }
    for part in path.parts:
        joined = normalize_name(part) or part
        if joined in KNOWN_DATASETS:
            parsed["dataset"] = joined
        if joined in KNOWN_ATTACKS:
            parsed["attack"] = joined
        if joined in KNOWN_DEFENSES:
            parsed["defense"] = joined
        if joined.startswith("seed"):
            parsed["seed"] = joined.replace("seed", "").strip("_") or None
    return parsed


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def list_metric_paths(results_dir: Path) -> list[Path]:
    return sorted(
        path for path in results_dir.rglob("metrics.jsonl") if "analysis" not in path.parts
    )


def load_runs(results_dir: Path) -> list[RunMetrics]:
    runs = []
    for path in list_metric_paths(results_dir):
        parsed = parse_from_path(path)
        records, read_error = read_jsonl(path)
        final = records[-1] if records else {}
        dataset = normalize_name(final.get("dataset") or parsed.get("dataset")) or "unknown_dataset"
        attack = normalize_name(final.get("attack") or parsed.get("attack")) or "unknown_attack"
        defense = normalize_name(final.get("defense") or parsed.get("defense")) or "unknown_defense"
        seed = str(final.get("seed") or parsed.get("seed") or "unknown")
        final_round = safe_int(final.get("round")) if records else None
        expected_round = EXPECTED_ROUNDS.get(dataset)

        if read_error:
            status = "malformed"
        elif not records:
            status = "empty"
        elif expected_round is not None and (final_round is None or final_round < expected_round):
            status = "incomplete"
        elif expected_round is None:
            status = "unknown_expected_round"
        else:
            status = "complete"

        runs.append(
            RunMetrics(
                dataset=dataset,
                attack=attack,
                defense=defense,
                seed=seed,
                path=path,
                records=tuple(records),
                final_round=final_round,
                expected_round=expected_round,
                status=status,
            )
        )
    return runs


def selected_pairs(
    runs: list[RunMetrics],
    dataset_filter: str | None,
    attack_filter: str | None,
) -> list[tuple[str, str]]:
    dataset_filter = normalize_name(dataset_filter)
    attack_filter = normalize_name(attack_filter)

    pairs = sorted({(run.dataset, run.attack) for run in runs})
    if dataset_filter:
        pairs = [pair for pair in pairs if pair[0] == dataset_filter]
    if attack_filter:
        pairs = [pair for pair in pairs if pair[1] == attack_filter]
    if dataset_filter and attack_filter and (dataset_filter, attack_filter) not in pairs:
        pairs = [(dataset_filter, attack_filter)]
    return pairs


def runs_for_pair(runs: list[RunMetrics], dataset: str, attack: str) -> list[RunMetrics]:
    return [run for run in runs if run.dataset == dataset and run.attack == attack]


def complete_runs_for_pair(runs: list[RunMetrics], dataset: str, attack: str) -> list[RunMetrics]:
    return [run for run in runs_for_pair(runs, dataset, attack) if run.complete]


def mean_std(values: Iterable[float]) -> tuple[float | None, float | None, int]:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None, None, 0
    if len(clean_values) == 1:
        return clean_values[0], None, 1
    return mean(clean_values), stdev(clean_values), len(clean_values)


def format_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100.0:.2f}"


def format_pct_mean_std(values: list[float]) -> str:
    avg, std, count = mean_std(values)
    if count == 0 or avg is None:
        return "--"
    if count == 1 or std is None:
        return format_pct(avg)
    return f"{avg * 100.0:.2f} +/- {std * 100.0:.2f}"


def format_tex_pct_mean_std(values: list[float], *, bold: bool) -> str:
    avg, std, count = mean_std(values)
    if count == 0 or avg is None:
        return "--"
    if count == 1 or std is None:
        text = f"{avg * 100.0:.2f}"
    else:
        text = f"{avg * 100.0:.2f} $\\pm$ {std * 100.0:.2f}"
    return f"\\textbf{{{text}}}" if bold else text


def mean_last_k(records: tuple[dict[str, Any], ...], metric: str, last_k: int) -> float | None:
    values = [safe_float(record.get(metric)) for record in records[-max(1, last_k) :]]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def run_table_value(run: RunMetrics, metric: str, last_k: int) -> float | None:
    return mean_last_k(run.records, metric, last_k)


def rounds_text(runs: list[RunMetrics]) -> str:
    rounds = sorted({run.final_round for run in runs if run.final_round is not None})
    if not rounds:
        return "--"
    if len(rounds) == 1:
        return str(rounds[0])
    return f"{rounds[0]}-{rounds[-1]}"


def table_rows(pair_runs: list[RunMetrics], last_k: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunMetrics]] = defaultdict(list)
    for run in pair_runs:
        grouped[run.defense].append(run)

    rows = []
    for defense in DEFENSE_ORDER:
        defense_runs = sorted(grouped.get(defense, []), key=lambda run: run.seed)
        if not defense_runs:
            continue
        clean_values = [
            value
            for run in defense_runs
            if (value := run_table_value(run, "clean_acc", last_k)) is not None
        ]
        asr_values = [
            value
            for run in defense_runs
            if (value := run_table_value(run, "asr", last_k)) is not None
        ]
        clean_avg, _, clean_count = mean_std(clean_values)
        asr_avg, _, asr_count = mean_std(asr_values)
        if clean_count == 0 and asr_count == 0:
            continue
        rows.append(
            {
                "defense": defense,
                "clean_values": clean_values,
                "asr_values": asr_values,
                "clean_avg": clean_avg,
                "asr_avg": asr_avg,
                "rounds": rounds_text(defense_runs),
            }
        )
    return rows


def write_markdown_table(pair_runs: list[RunMetrics], output: Path, last_k: int) -> None:
    rows = table_rows(pair_runs, last_k)
    best_clean = max((row["clean_avg"] for row in rows if row["clean_avg"] is not None), default=None)
    best_asr = min((row["asr_avg"] for row in rows if row["asr_avg"] is not None), default=None)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("| Defense | Clean Acc. | ASR | Rounds |\n")
        handle.write("|---|---:|---:|---:|\n")
        for row in rows:
            clean = format_pct_mean_std(row["clean_values"])
            asr = format_pct_mean_std(row["asr_values"])
            if row["clean_avg"] is not None and row["clean_avg"] == best_clean:
                clean = f"**{clean}**"
            if row["asr_avg"] is not None and row["asr_avg"] == best_asr:
                asr = f"**{asr}**"
            handle.write(
                f"| {DEFENSE_LABELS[row['defense']]} | {clean} | {asr} | {row['rounds']} |\n"
            )


def table_label(dataset: str, attack: str) -> str:
    return f"tab:{dataset}_{attack}".replace("-", "_")


def write_latex_table(
    dataset: str,
    attack: str,
    pair_runs: list[RunMetrics],
    output: Path,
    last_k: int,
) -> None:
    rows = table_rows(pair_runs, last_k)
    best_clean = max((row["clean_avg"] for row in rows if row["clean_avg"] is not None), default=None)
    best_asr = min((row["asr_avg"] for row in rows if row["asr_avg"] is not None), default=None)
    caption = f"Results on {DATASET_LABELS.get(dataset, dataset)} under {ATTACK_LABELS.get(attack, attack)}."

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[t]\n")
        handle.write("\\centering\n")
        handle.write(f"\\caption{{{caption}}}\n")
        handle.write(f"\\label{{{table_label(dataset, attack)}}}\n")
        handle.write("\\begin{tabular}{lccc}\n")
        handle.write("\\toprule\n")
        handle.write("Defense & Clean Acc. & ASR & Rounds \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            clean = format_tex_pct_mean_std(
                row["clean_values"],
                bold=row["clean_avg"] is not None and row["clean_avg"] == best_clean,
            )
            asr = format_tex_pct_mean_std(
                row["asr_values"],
                bold=row["asr_avg"] is not None and row["asr_avg"] == best_asr,
            )
            handle.write(
                f"{DEFENSE_LABELS[row['defense']]} & {clean} & {asr} & {row['rounds']} \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")


def rolling_mean(values: list[float], window: int) -> list[float]:
    window = max(1, window)
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed.append(mean(values[start : idx + 1]))
    return smoothed


def metric_by_round(runs: list[RunMetrics], metric: str) -> dict[int, list[float]]:
    values: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        for record in run.records:
            round_idx = safe_int(record.get("round"))
            value = safe_float(record.get(metric))
            if round_idx is not None and value is not None:
                values[round_idx].append(value)
    return values


def plot_metric(
    ax: plt.Axes,
    pair_runs: list[RunMetrics],
    metric: str,
    smooth_window: int,
) -> None:
    grouped: dict[str, list[RunMetrics]] = defaultdict(list)
    for run in pair_runs:
        grouped[run.defense].append(run)

    for defense in DEFENSE_ORDER:
        defense_runs = grouped.get(defense, [])
        if not defense_runs:
            continue
        values_by_round = metric_by_round(defense_runs, metric)
        rounds = sorted(values_by_round)
        means = [mean(values_by_round[round_idx]) * 100.0 for round_idx in rounds]
        smoothed = rolling_mean(means, smooth_window)
        ax.plot(
            rounds,
            smoothed,
            label=DEFENSE_LABELS.get(defense, defense),
            color=DEFENSE_COLORS.get(defense, "#777777"),
            linewidth=2.0 if defense == "traceguard" else 1.6,
        )


def write_curves(
    dataset: str,
    attack: str,
    pair_runs: list[RunMetrics],
    output_png: Path,
    output_pdf: Path,
    smooth_window: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    title = f"{DATASET_LABELS.get(dataset, dataset)} / {ATTACK_LABELS.get(attack, attack)}"
    axes[0].set_title(title)

    plot_metric(axes[0], pair_runs, "clean_acc", smooth_window)
    plot_metric(axes[1], pair_runs, "asr", smooth_window)

    axes[0].set_ylabel("Clean Accuracy (%)")
    axes[1].set_ylabel("ASR (%)")
    axes[1].set_xlabel("Communication Round")
    for ax in axes:
        ax.set_ylim(0.0, 100.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(True, color="#D9D9D9", alpha=0.8, linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def warning_lines(pair_runs: list[RunMetrics]) -> list[str]:
    lines = []
    for run in sorted(pair_runs, key=lambda item: (item.defense, item.seed)):
        if run.complete:
            continue
        lines.append(
            "\t".join(
                [
                    run.dataset,
                    run.attack,
                    run.defense,
                    run.seed,
                    str(run.final_round) if run.final_round is not None else "-",
                    str(run.expected_round) if run.expected_round is not None else "-",
                    str(run.path),
                ]
            )
        )
    return lines


def write_warnings(pair_runs: list[RunMetrics], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("dataset\tattack\tdefense\tseed\tfinal_round\texpected_round\tmetrics_path\n")
        for line in warning_lines(pair_runs):
            handle.write(line + "\n")


def analyze_pair(
    dataset: str,
    attack: str,
    runs: list[RunMetrics],
    analysis_dir: Path,
    last_k: int,
    smooth_window: int,
    dpi: int,
) -> list[Path]:
    pair_runs = runs_for_pair(runs, dataset, attack)
    complete_runs = [run for run in pair_runs if run.complete]
    output_dir = analysis_dir / dataset / attack
    stale_outputs = [
        output_dir / "traceguard_diagnostics.csv",
        output_dir / "model_replacement_final_bars.png",
    ]
    for stale_output in stale_outputs:
        if stale_output.exists():
            stale_output.unlink()
    table_md = output_dir / "table.md"
    table_tex = output_dir / "table.tex"
    curves_png = output_dir / "curves.png"
    curves_pdf = output_dir / "curves.pdf"
    warnings = output_dir / "warnings.txt"

    write_markdown_table(complete_runs, table_md, last_k)
    write_latex_table(dataset, attack, complete_runs, table_tex, last_k)
    write_curves(dataset, attack, complete_runs, curves_png, curves_pdf, smooth_window, dpi)
    write_warnings(pair_runs, warnings)
    return [table_md, table_tex, curves_png, curves_pdf, warnings]


def main() -> int:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    analysis_dir = Path(args.analysis_dir)
    dataset_filter = normalize_name(args.dataset)
    attack_filter = normalize_name(args.attack)

    runs = load_runs(results_dir)
    pairs = selected_pairs(runs, dataset_filter, attack_filter)
    if not pairs:
        print("found_pairs=0")
        return 0

    written = []
    for dataset, attack in pairs:
        written.extend(
            analyze_pair(
                dataset,
                attack,
                runs,
                analysis_dir,
                args.last_k,
                args.smooth_window,
                args.dpi,
            )
        )

    print(f"found_pairs={len(pairs)}")
    for path in written:
        print(f"wrote={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
