# Feedback milestone reconciliation

Target base: `707f5395c06d5bd5503697d6359fceebbf1c240d` (AgentLab 0.3.1).
Its ancestors `7cf544e8a820a58fb9254c47b6dcac500b8c914c` and
`49da5651512bf14d910bad34732986c2feacdc2d` were verified from Git.

The original delivered feedback patch was recovered without alteration and its
SHA-256 checked:
`e0e02515bfa38386b0c90937a6f5740354501323e4a873e44851232ba8677643`.
It remains a checkpoint against the original 49da565 base. The refreshed patch is
against 707f539 and must not be applied as if it were the original-base patch.

## Reconciliation

- Resolved one textual executor conflict by keeping both the new DecisionContext
  methods and the user's `execute(..., token_usage=...)` support
- Preserved benchmark adapters, native evaluation flow, resumable batch runner,
  structured action interface, and keyless loopback Qwen/MLX support
- Captured the built-in backend's full prepared JSON request before generation,
  with the exact body submitted afterward and bound to executed actions
- Retained token usage and journal context IDs for finish/failed model calls
- Labeled arbitrary/custom backend inputs unknown and excluded them from default
  SFT rather than claiming hidden prompt completeness
- Aligned native verifier registry/result names to `native/<benchmark>` so existing
  receipts satisfy strict scope/provenance validation without weakening checks
- Regenerated schemas while preserving API version 0.3.1

Model endpoint, authentication headers and API keys are not captured in the request
body. Prompt data continues to require the same private benchmark artifact handling.
No benchmark datasets, reference solutions, model weights or credentials are added.

No Docker/Postgres or real model/MLX execution is claimed by these local checks.
GitHub publication remains blocked by the previously reported integration permission
denial; no retry, remote branch modification, PR, merge or deployment was performed.

## Verification

- `OTEL_TRACES_SAMPLER=always_on .venv/bin/pytest -q`: **126 passed, 11 skipped**
- Ruff lint and format checks: passed; regenerated schemas and `git diff --check`: passed
- Independent request/context/runner review: **26 tests passed**, no remaining blockers
- Independent native receipt and preservation review: **34 tests passed**
- API stays 0.3.1; Qwen/MLX, adapters, dependencies and user SDK version changes remain
  byte-identical to the new main outside the explicitly reconciled files

The 11 skips require live Docker/Postgres. One upstream TestClient deprecation
warning remains. Explicit `always_on` avoids the executor's ambient 1% telemetry
sampling from making the existing tracing assertion stochastic. Model inference,
benchmark datasets and Apple Silicon execution were not run or claimed validated.

The aggregate actor RL projection now has its own explicit final episode boundary;
trailing native evaluator steps cannot erase it. Regression tests cover exact
request fidelity, frozen settings and command seed, credential exclusion, token
usage, custom-backend uncertainty, exec/finish/evaluator sequencing, multi-step
flows, budgets/timeouts/errors, no retries, resume, and unchanged audit trajectories.

## October 9, 2026 recovery and replay correction

Remote main was still `707f5395c06d5bd5503697d6359fceebbf1c240d`. The delivered
patch was recovered and verified byte-for-byte, including an identical Git diff
round trip. A fresh checkout reproduced **126 passed, 11 skipped** before edits.

Static review then found that exact replay recognized only legacy test criteria,
not structured test policies. The corrected comparison now uses every structured
test policy's exact argv, falling back to legacy declarations only when structured
criteria are absent. It also replaces the former blanket test-output exception
with exact stdout comparison and narrowly normalized unittest stderr timing.
Counts, verdicts, diagnostics, all other log content, exit codes and collected
artifact hashes remain significant. Unrecognized output formats compare exactly.

- Full suite: **172 passed, 11 infrastructure skips**
- New focused replay comparison regressions: **46 passed**, independently reviewed
- Ruff lint/format, generated-schema consistency and diff whitespace checks: passed
- Benchmark/MLX configuration, dependencies and CI workflow remain unchanged

Docker CLI is unavailable and the existing integration fixture reports a Docker
socket `PermissionError`; no restricted infrastructure was bypassed. Postgres,
Docker, real inference, Apple Silicon and benchmark datasets remain unverified.
Repository read access succeeds, but no evidence establishes that the earlier
GitHub integration write denial was repaired. No remote write retry, PR, CI run,
merge or deployment was performed during this verification.
