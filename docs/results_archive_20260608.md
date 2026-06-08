# Experiment Results Archive - 2026-06-08

This archive contains existing CIFAR-10 experiment metrics under `outputs/`.

## Scope

- Dataset: CIFAR-10
- Seed: 123
- Attacks: A3FL, DBA, Model Replacement, Neurotoxin
- Defenses: ASAGuard, FDCR, FedAvg, FLAME, FLIP, Multi-Krum, Trimmed Mean
- Metric format: JSONL, one record per round

## Files

| Attack | Defense | Path | Records | Status |
| --- | --- | --- | ---: | --- |
| A3FL | ASAGuard | `outputs/cifar10/a3fl/asaguard/seed_123/metrics.jsonl` | 27 | partial |
| A3FL | FDCR | `outputs/cifar10/a3fl/fdcr/seed_123/metrics.jsonl` | 200 | complete |
| A3FL | FedAvg | `outputs/cifar10/a3fl/fedavg/seed_123/metrics.jsonl` | 200 | complete |
| A3FL | FLAME | `outputs/cifar10/a3fl/flame/seed_123/metrics.jsonl` | 200 | complete |
| A3FL | FLIP | `outputs/cifar10/a3fl/flip/seed_123/metrics.jsonl` | 200 | complete |
| A3FL | Multi-Krum | `outputs/cifar10/a3fl/multi_krum/seed_123/metrics.jsonl` | 200 | complete |
| A3FL | Trimmed Mean | `outputs/cifar10/a3fl/trimmed_mean/seed_123/metrics.jsonl` | 200 | complete |
| DBA | ASAGuard | `outputs/cifar10/dba/asaguard/seed_123/metrics.jsonl` | 37 | partial |
| DBA | FDCR | `outputs/cifar10/dba/fdcr/seed_123/metrics.jsonl` | 200 | complete |
| DBA | FedAvg | `outputs/cifar10/dba/fedavg/seed_123/metrics.jsonl` | 200 | complete |
| DBA | FLAME | `outputs/cifar10/dba/flame/seed_123/metrics.jsonl` | 200 | complete |
| DBA | FLIP | `outputs/cifar10/dba/flip/seed_123/metrics.jsonl` | 200 | complete |
| DBA | Multi-Krum | `outputs/cifar10/dba/multi_krum/seed_123/metrics.jsonl` | 200 | complete |
| DBA | Trimmed Mean | `outputs/cifar10/dba/trimmed_mean/seed_123/metrics.jsonl` | 200 | complete |
| Model Replacement | ASAGuard | `outputs/cifar10/model_replacement/asaguard/seed_123/metrics.jsonl` | 44 | partial |
| Model Replacement | FDCR | `outputs/cifar10/model_replacement/fdcr/seed_123/metrics.jsonl` | 200 | complete |
| Model Replacement | FedAvg | `outputs/cifar10/model_replacement/fedavg/seed_123/metrics.jsonl` | 200 | complete |
| Model Replacement | FLAME | `outputs/cifar10/model_replacement/flame/seed_123/metrics.jsonl` | 200 | complete |
| Model Replacement | FLIP | `outputs/cifar10/model_replacement/flip/seed_123/metrics.jsonl` | 200 | complete |
| Model Replacement | Multi-Krum | `outputs/cifar10/model_replacement/multi_krum/seed_123/metrics.jsonl` | 200 | complete |
| Model Replacement | Trimmed Mean | `outputs/cifar10/model_replacement/trimmed_mean/seed_123/metrics.jsonl` | 200 | complete |
| Neurotoxin | ASAGuard | `outputs/cifar10/neurotoxin/asaguard/seed_123/metrics.jsonl` | 25 | partial |
| Neurotoxin | FDCR | `outputs/cifar10/neurotoxin/fdcr/seed_123/metrics.jsonl` | 200 | complete |
| Neurotoxin | FedAvg | `outputs/cifar10/neurotoxin/fedavg/seed_123/metrics.jsonl` | 200 | complete |
| Neurotoxin | FLAME | `outputs/cifar10/neurotoxin/flame/seed_123/metrics.jsonl` | 200 | complete |
| Neurotoxin | FLIP | `outputs/cifar10/neurotoxin/flip/seed_123/metrics.jsonl` | 200 | complete |
| Neurotoxin | Multi-Krum | `outputs/cifar10/neurotoxin/multi_krum/seed_123/metrics.jsonl` | 200 | complete |
| Neurotoxin | Trimmed Mean | `outputs/cifar10/neurotoxin/trimmed_mean/seed_123/metrics.jsonl` | 200 | complete |

Total records: 4933.

## Notes

The ASAGuard runs currently present in this archive are partial and should not be treated as full 200-round runs. No datasets, checkpoints, logs, caches, or model binaries are included.
