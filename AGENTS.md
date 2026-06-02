# ASAGuard Coding Instructions

## Project Goal
Implement ASAGuard: Association-Safe Aggregation for Federated Backdoor Defense.

ASAGuard is a server-side trigger-target functional association editing framework. It is NOT a client-side sample filtering method.

## Forbidden Components
Do NOT implement:
- client-side local purifier
- local sample filtering defense
- TEE
- ZKP
- blockchain
- LLM module
- multi-signal weighted risk score
- client risk scoring, admission control, downweighting, or rejection as the ASAGuard method

## Core Method
ASAGuard consists of:
1. Counterfactual Probe Bank
2. Backdoor-Sensitive Subspace Estimator
3. Association-Safe Projection Aggregator

The target margin is:

M_y(x; w) = z_y(x; w) - max_{c != y} z_c(x; w)

The sensitive direction is:

q_{x,f,y} = grad_w M_y(T_f(x); w_t) - grad_w M_y(x; w_t)

The core projection is:

Delta_i_perp = Delta_i - U_t (U_t^T Delta_i)

Aggregation:

w_{t+1} = w_t + mean_i Delta_i_perp

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
- ASAGuard

## Engineering Rules
- Keep code modular and config-driven.
- Do not hard-code dataset paths.
- Do not commit data, checkpoints, logs, or results.
- Default behavior should be FedAvg without attacks.
- Every major feature must have a debug-mode smoke test.
- Use deterministic seeds.
- Save metrics as JSONL or CSV.
- Do not run full experiments unless explicitly asked.
