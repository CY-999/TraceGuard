# TRACEGuard

TRACEGuard is a server-side trigger-family functional auditing framework for federated backdoor defense.

This repository implements the final TRACEGuard design described in `docs/TRACEGuard_final_design.md`.

## Core Principle

TRACEGuard does not rely on client-side local filtering or purification. It audits submitted client updates on the server side before aggregation.