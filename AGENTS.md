# TRACEGuard Coding Instructions

## Project Goal
Implement TRACEGuard: Server-Side Trigger-Family Auditing for Federated Backdoor Defense.

TRACEGuard is a server-side functional auditing framework. It is NOT a client-side sample filtering method.

## Forbidden Components
Do NOT implement:
- client-side local purifier
- local sample filtering defense
- TEE
- ZKP
- blockchain
- LLM module
- multi-signal weighted risk score

## Core Method
TRACEGuard consists of:
1. Trigger-Family Probe Bank
2. Update Response Auditor
3. Robust Admission Controller

The core risk score is Paired Trigger Amplification Score:

A_i(x,f) =
[M_y(T_f(x); w_t + Δw_i) - M_y(T_f(x); w_t)]
-
[M_y(x; w_t + Δw_i) - M_y(x; w_t)]

R_i = median_{x,f} A_i(x,f)

Admission:
z_i = (R_i - median_j R_j) / (MAD_j(R_j) + eps)
a_i = clip(1 - z_i / tau, 0, 1)

## Main Datasets
- CIFAR-10
- CIFAR-100
- Tiny-ImageNet

## Main Attacks
- Model Replacement
- DBA
- Neurotoxin
- A3FL

## Main Defenses
- FedAvg
- Multi-Krum
- Trimmed Mean
- FLAME
- FLIP
- FDCR
- TRACEGuard

## Engineering Rules
- Keep code modular and config-driven.
- Do not hard-code dataset paths.
- Do not commit data, checkpoints, logs, or results.
- Default behavior should be FedAvg without attacks.
- Every major feature must have a debug-mode smoke test.
- Use deterministic seeds.
- Save metrics as JSONL or CSV.
- Do not run full experiments unless explicitly asked.