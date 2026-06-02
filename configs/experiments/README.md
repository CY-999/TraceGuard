# ASAGuard Main Experiment Templates

These YAML files are templates for the main CIFAR-10, CIFAR-100, and Tiny-ImageNet experiments in the ASAGuard paper.

They do not automatically download data. Prepare each dataset under the configured `dataset.data_dir` before running an experiment.

CPU is only for code-connectivity checks. Main-paper experiments should be run on GPU.

Tiny-ImageNet should use the common local layout:

```text
data/tiny-imagenet-200/
  train/
    n01443537/
      images/
        xxx.JPEG
  val/
    images/
      xxx.JPEG
    val_annotations.txt
  wnids.txt
```

Example:

```bash
python -m asaguard.fl.run --config configs/experiments/cifar10_dba.yaml
```

Main experiments use a single default seed, `seed=123`. The provided runner does not generate multi-seed repeats by default; `--seed` is available only for temporary overrides.

Main experiments use dataset-specific ResNet-18 variants trained from scratch. No ImageNet pretrained weights are used or downloaded.

- CIFAR-10 uses `resnet18_cifar` for 200 rounds.
- CIFAR-100 uses `resnet18_cifar` for 300 rounds.
- Tiny-ImageNet uses `resnet18_tiny` for 300 rounds.

`simple_cnn` is reserved for sanity/debug runs.

Main experiments use 10 malicious clients out of 100 by default, with `attack.poison_ratio=0.5`. Appendix sensitivity can vary poison ratio over `0.1/0.3/1.0` and malicious ratio over `4%/20%/30%`.

`attack.num_malicious` is the global malicious-client count. Robust aggregation baselines use method-specific Byzantine bounds, such as `multi_krum.num_byzantine` and `trimmed_mean.num_byzantine`, among clients participating in the current round. For Multi-Krum and Trimmed Mean, main experiments use `multi_krum.num_byzantine=2` and `trimmed_mean.num_byzantine=2` with 10 clients per round; this satisfies Multi-Krum's `n > 2f + 2` and Trimmed Mean's `2b < n`. Do not pass the global malicious-client count directly as Krum `f` or Trimmed Mean `b`.

Main-paper defaults:

- `seed=123`
- 100 clients
- 10 clients per round
- 10 malicious clients
- `poison_ratio=0.5`
- Dirichlet non-IID `alpha=0.5`

Single GPU experiment:

```bash
python -m asaguard.fl.run --config configs/experiments/cifar10_dba.yaml --defense asaguard
```

Run the full CIFAR-10 matrix:

```bash
python scripts/run_main_experiments.py --dataset cifar10 --run
```

Collect CIFAR-10 results:

```bash
python scripts/collect_results.py --results-dir outputs --output outputs/cifar10_summary.csv
```

Before running, confirm CIFAR-10 and CIFAR-100 are under `data/`, Tiny-ImageNet is under `data/tiny-imagenet-200/`, and `outputs/` is not committed to git.

Main defense baselines can be selected with CLI overrides:

```bash
--defense fedavg
--defense multi_krum
--defense trimmed_mean
--defense flame
--defense flip
--defense fdcr
--defense asaguard
```

Main attack baselines:

- `model_replacement`
- `dba`
- `neurotoxin`
- `a3fl`

Supported main datasets:

- `cifar10`
- `cifar100`
- `tinyimagenet`

Main defense baselines:

- `fedavg`
- `multi_krum`
- `trimmed_mean`
- `flame`
- `flip`
- `fdcr`
- `ASAGuard`

ASAGuard is executed on the server side through counterfactual probes, target-margin gradient subspace estimation, and projection aggregation. It does not require or use client-side local purification, client risk scoring, update downweighting, or client rejection.

ASAGuard configuration is method-specific: `asaguard.subspace_rank` controls the sensitive subspace dimension, `asaguard.eps` controls numerical stabilizers, and `fdcr.tau` remains specific to FDCR-style weighting.

Experiment outputs are saved under `outputs/<dataset>/<attack>/<defense>/seed_<seed>/` by default. The `outputs/` directory should not be committed to git.

## Experiment Matrix Script

Generate a single command without running it:

```bash
python scripts/run_main_experiments.py --dataset cifar10 --attack dba --defense asaguard --dry-run
```

Run that single experiment explicitly:

```bash
python scripts/run_main_experiments.py --dataset cifar10 --attack dba --defense asaguard --run
```

If `--attack` or `--defense` is omitted, the script prints the corresponding main-paper matrix. It still does not run anything unless `--run` is provided.

## Result Collection

Collect all `metrics.jsonl` files under `outputs/` into a run-level CSV and an averaged CSV:

```bash
python scripts/collect_results.py --results-dir outputs --output outputs/summary.csv
```

This also writes `outputs/summary_avg.csv`, grouped by `dataset`, `attack`, and `defense`. The averaged CSV is mainly for future runs where multiple seeds are needed.
