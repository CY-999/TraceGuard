# TRACEGuard Main CIFAR-10 Experiment Templates

These YAML files are templates for the main CIFAR-10 experiments in the TRACEGuard paper.

They do not automatically download data. Prepare CIFAR-10 under the configured `dataset.data_dir` before running an experiment.

Example:

```bash
python -m traceguard.fl.run --config configs/experiments/cifar10_dba.yaml
```

Main defense baselines can be selected with CLI overrides:

```bash
--defense fedavg
--defense multi_krum
--defense trimmed_mean
--defense flame
--defense flip
--defense fdcr
--defense traceguard
```

Main attack baselines:

- `model_replacement`
- `dba`
- `neurotoxin`
- `a3fl`

## Experiment Matrix Script

Generate a single command without running it:

```bash
python scripts/run_main_experiments.py --dataset cifar10 --attack dba --defense traceguard --seed 0 --dry-run
```

Run that single experiment explicitly:

```bash
python scripts/run_main_experiments.py --dataset cifar10 --attack dba --defense traceguard --seed 0 --run
```

If `--attack` or `--defense` is omitted, the script prints the corresponding main-paper matrix. It still does not run anything unless `--run` is provided.

## Result Collection

Collect all `metrics.jsonl` files under `outputs/` into a run-level CSV and an averaged CSV:

```bash
python scripts/collect_results.py --results-dir outputs --output outputs/summary.csv
```

This also writes `outputs/summary_avg.csv`, grouped by `dataset`, `attack`, and `defense`.
