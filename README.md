# TRACEGuard

TRACEGuard is a server-side trigger-family functional auditing framework for federated backdoor defense.

This repository implements the final TRACEGuard design described in `docs/TRACEGuard_final_design.md`.

## Core Principle

TRACEGuard does not rely on client-side local filtering or purification. It audits submitted client updates on the server side before aggregation.

This initial scaffold provides only the Python package layout, YAML configuration loading, and a minimal CLI. It intentionally does not implement training, attacks, defense baselines, or the TRACEGuard method yet.

## Install

```bash
pip install -e .
```

## CLI

```bash
python -m traceguard.fl.run --help
python -m traceguard.fl.run --debug --print-config
python -m traceguard.fl.run --config configs/default.yaml --dataset cifar100 --defense traceguard --print-config
```

Configuration is loaded from `configs/default.yaml` by default. Passing `--config` deep-merges the selected YAML over the default. Passing `--debug` without `--config` deep-merges `configs/debug.yaml` over the default. CLI flags are applied last.

## Main Experiments

Main-paper experiment templates live in `configs/experiments/`.

CPU runs are intended only for code-connectivity checks such as the sanity smoke config. Main-paper experiments should be run on GPU.

Datasets:

- CIFAR-10
- CIFAR-100
- Tiny-ImageNet

Attacks:

- `model_replacement`
- `dba`
- `neurotoxin`
- `a3fl`

Defenses:

- `fedavg`
- `multi_krum`
- `trimmed_mean`
- `flame`
- `flip`
- `fdcr`
- `traceguard`

The default seed is fixed to `123`. Data is not downloaded automatically; prepare CIFAR-10, CIFAR-100, and Tiny-ImageNet under `dataset.data_dir`.

Main-paper defaults:

- CIFAR-10: `resnet18_cifar`, 200 rounds
- CIFAR-100: `resnet18_cifar`, 300 rounds
- Tiny-ImageNet: `resnet18_tiny`, 300 rounds
- `seed=123`
- 100 clients
- 10 clients per round
- 10 malicious clients
- `poison_ratio=0.5`
- Dirichlet non-IID `alpha=0.5`

Single GPU experiment:

```bash
python -m traceguard.fl.run --config configs/experiments/cifar10_dba.yaml --defense traceguard
```

Single experiment dry-run:

```bash
python scripts/run_main_experiments.py --dataset cifar10 --attack dba --defense traceguard --dry-run
```

Single experiment run:

```bash
python scripts/run_main_experiments.py --dataset cifar10 --attack dba --defense traceguard --run
```

Full CIFAR-10 matrix dry-run:

```bash
python scripts/run_main_experiments.py --dataset cifar10 --dry-run
```

Collect results:

```bash
python scripts/collect_results.py --results-dir outputs --output outputs/summary.csv
```

Collect CIFAR-10 results:

```bash
python scripts/collect_results.py --results-dir outputs --output outputs/cifar10_summary.csv
```

Sanity dry-run for one fakedata combination:

```bash
python scripts/run_sanity_matrix.py --attack dba --defense traceguard --dry-run
```

See `docs/experiment_plan.md` for the full experiment plan and Tiny-ImageNet directory layout. Results are saved under `outputs/<dataset>/<attack>/<defense>/seed_<seed>/`, and `outputs/` should not be committed to git.

TRACEGuard runs entirely on the server side: secret probe bank, update response auditor, and robust admission controller. It does not use client-side local purification.

Tau configuration is separated by method: `traceguard.tau` controls TRACEGuard admission, while `defense.fdcr_tau` controls FDCR-style weighting.
