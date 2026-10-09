# Feedback milestone delivery

Base: `49da5651512bf14d910bad34732986c2feacdc2d` (main as cloned).
This section records the initial local-only validation before reconciliation.
See [the reconciliation report](feedback_reconciliation.md) for the current base and checks.

## Delivered

- Immutable, hash-validated DecisionContext and pre-action binding with explicit
  client-attested coverage; feedback endpoint and strict source ownership rules
- Deterministic external reactive demo: failure observation changes next action
- Typed hypothesis/criterion/verifier policy binding, trust labeling and host-injected
  evaluator boundary; unrelated/tampered evidence fails closed
- Truthful captured-input SFT/RL exports, nullable unknowns, timeout/boundary labels,
  replay weighting, unknown legacy provenance exclusion and lineage-grouped splits
- Backwards-compatible entity defaults, legacy execution path, generated public schemas
- Contract/adversarial tests and usage/limitations documentation

## Verification in this cloud workspace

- `OTEL_TRACES_SAMPLER=always_on .venv/bin/pytest -q`: **96 passed, 11 skipped**
- `.venv/bin/ruff check .`: passed
- `.venv/bin/ruff format --check .`: passed
- `.venv/bin/python examples/generate_schemas.py`: succeeded
- `git diff --check`: passed

One upstream FastAPI/Starlette warning reports the deprecated httpx TestClient adapter.
The host sets a 1% OpenTelemetry sampling default; an initial unmodified tracing test
failed because no span was sampled. `always_on` makes tracing assertions deterministic;
application telemetry behavior was not changed to mask that environment setting.

The 11 skips are live Docker/Postgres integration tests. A forced attempt with
`HARNESS_INTEGRATION=1` stopped on the first fixture: Docker CLI is absent and Docker
SDK socket access returned PermissionError. No Docker/Postgres integration behavior,
real-container reactive demo, or end-to-end scientific quality is claimed verified.
The SQLite/memory fake checks exercise interface contracts, not container isolation.

Read [feedback contracts](feedback_milestone.md) and
[verifier trust/usage](verifier_policies.md) for explicit limitations.
