# ASAGuard

ASAGuard is a server-side association-safe aggregation framework for federated backdoor defense.

The method does not identify malicious clients, score client risk, downweight updates, or reject clients. It estimates a trigger-target sensitive functional subspace from server-side clean/trigger counterfactual probes, projects every submitted update onto that subspace's orthogonal complement, then averages the projected updates.

## Install

```bash
pip install -e .
```

## CLI

```bash
python -m asaguard.fl.run --help
python -m asaguard.fl.run --debug --print-config
python -m asaguard.fl.run --config configs/default.yaml --dataset cifar100 --defense asaguard --print-config
```

Configuration is loaded from `configs/default.yaml` by default. Passing `--config` deep-merges the selected YAML over the default. Passing `--debug` without `--config` deep-merges `configs/debug.yaml` over the default. CLI flags are applied last.

## Main Experiments

Main-paper experiment templates live in `configs/experiments/`.

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
- `asaguard`

Data is not downloaded automatically; prepare CIFAR-10, CIFAR-100, and Tiny-ImageNet under `dataset.data_dir`. CPU runs are intended only for code-connectivity checks; main-paper experiments should run on GPU.

ASAGuard uses a server-side clean reference buffer reserved from the clean training split before client partitioning or attack injection. Reference samples are removed from the federated client training pool for every defense, so baselines and ASAGuard share the same remaining client data. The test set is used only for evaluation and must never be used for defense-time probe construction.

Single GPU experiment:

```bash
python -m asaguard.fl.run --config configs/experiments/cifar10_dba.yaml --defense asaguard
```

Single experiment dry-run:

```bash
python scripts/run_main_experiments.py --dataset cifar10 --attack dba --defense asaguard --dry-run
```

Single experiment run:

```bash
python scripts/run_main_experiments.py --dataset cifar10 --attack dba --defense asaguard --run
```

Collect results:

```bash
python scripts/collect_results.py --results-dir outputs --output outputs/summary.csv
```

Sanity dry-run for one fakedata combination:

```bash
python scripts/run_sanity_matrix.py --attack dba --defense asaguard --dry-run
```

Results are saved under `outputs/<dataset>/<attack>/<defense>/seed_<seed>/`, and `outputs/` should not be committed to git.

## ASAGuard Metrics

ASAGuard writes mechanism metrics to `metrics.jsonl`:

- `asaguard_ac_mean_before`
- `asaguard_ac_mean_after`
- `asaguard_projected_energy_ratio_mean`
- `asaguard_subspace_rank`
- `asaguard_num_q_vectors`
