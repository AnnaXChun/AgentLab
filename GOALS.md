# Agent Experiment Infrastructure goals

Execute externally selected actions; isolate, observe, version, verify, snapshot, restore, replay, record and export experiments. AI4S is a domain plugin. Priorities: schema correctness > provenance > reproducibility > replay > feature count.

No planner, generic reasoning agent, autonomous multi-agent system, persona, task decomposition, prompt framework or model training belongs in core.

See [architecture_refactor.md](docs/architecture_refactor.md) for the implemented vertical slice and legacy compatibility decisions.

---

# Scientific Harness V0

A domain-agnostic, agent-agnostic execution harness for evidence-bound scientific work.

The acceptance path is goal → agent action → retrieval → MCP → sandbox → immutable
artifact → claim → independent verifier → canonical trajectory → replay → export.

## Invariants
- Core knows neither model vendors nor scientific domains. DomainPack is the extension boundary.
- Agents return declarative actions; only the harness dispatches external operations.
- Append-only PostgreSQL events are canonical. Projections and telemetry are derived.
- Every operation records intent before execution and records success or failure afterward.
- Claims require evidence, immutable artifacts and successful independent verification.
- SHA-256 artifacts cannot be overwritten. Public schemas carry schema_version.
- Record commands, inputs, environment, versions, seed, output and failures for replay.
- Fork and rollback append history; they never rewrite earlier branches.

## Non-goals
No model training, RL implementation, agent framework, UI, wet lab, multi-tenancy,
billing, large benchmark, or production distributed sandbox scheduler.

## Delivery order
1. Schemas and skeleton 2. Persistence 3. Docker sandbox 4. MCP gateway
5. Retrieval 6. Verification 7. Branch/replay 8. Export 9. Tracing 10. Demo/tests/docs.
Each phase has executable tests before the next phase starts.
