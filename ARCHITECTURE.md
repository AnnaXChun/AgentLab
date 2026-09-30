# Agent Experiment Infrastructure

The active 0.2 architecture is described in [docs/architecture_refactor.md](docs/architecture_refactor.md), [schema.md](docs/schema.md), [provenance.md](docs/provenance.md), and [replay.md](docs/replay.md).

Agent is replaceable. Environment is replaceable. Model and tools are replaceable. Artifacts, evidence, provenance and trajectories are persistent.

The earlier architecture below describes the retained v1 compatibility layer; its agent loop now lives outside Harness and its Claim semantics are not the v2 semantics.

---

# Architecture

`packages.sdk.Harness` owns orchestration. AgentAdapter receives a JSON observation
context and returns an AgentAction. Adapters are trusted host plugins; a Python
interface cannot prevent a malicious in-process plugin from accessing the host.
Untrusted generated code executes only through SandboxBackend.

## Boundaries
- core: versioned entities, events, append-only storage, claims and verifier contracts.
- adapters/storage: local SHA-256 CAS; SQLAlchemy PostgreSQL event persistence.
- adapters/sandbox: Docker implementation behind the sandbox protocol.
- mcp_gateway: JSON Schema validation and explicit tool allowlist before MCP dispatch.
- retrieval: parser/embedding interfaces, chunking, Postgres FTS + pgvector fusion.
- domain_packs: versioned configuration, documents and MCP server commands.
- provenance: deterministic event folding, checkpoints, branches and rollback.
- exporters: JSONL, normalized Parquet, visible SFT, RL records and recovery pairs.
- sdk: orchestration, FastAPI and mock agent driver.

## Canonical truth and durability
Postgres events contain immutable sequence numbers and per-branch hash-linked history state.
A per-episode row lock serializes append operations. Database triggers reject UPDATE
and DELETE. Foreign keys and unique constraints protect event identity and ordering.
Large JSON payloads become immutable artifacts; the event contains a CAS reference.
The small mutable episode row is an index/locking projection, never replay authority.
Evidence, claims, verifications and artifact metadata are persisted inside canonical
events; relational retrieval chunks are a rebuildable search index.

An intent is committed before an external call. An interrupted intent is visible as
unfinished; V0 never silently retries an external side effect after a crash. This is
at-least-once orchestration with explicit recovery, not cross-system exactly-once.
Filesystem writes used to commit the log/CAS are the trusted persistence substrate;
bootstrap migration and administrative exports are host operations. All experiment
retrieval, ingestion, tool, sandbox, artifact and verification effects are logged.

## Scientific contract
Claims begin UNVERIFIED. Resolved evidence plus artifact references allow SUPPORTED.
Only the verifier service may append VERIFIED; failed verification gives CONTRADICTED.
Verifiers consume immutable artifact bytes, not an agent-provided observed value.
A verification records its claim ID, context, criteria and input artifact hashes.

## Replay and experiment DAG
Replay reconstructs state without executing tools. Hash validation detects changed
payloads/order. Checkpoints retain state and sandbox snapshot artifacts. Fork gets a
new branch ID and inherits checkpoint state. Rollback appends a restore event. Edges
encode next, fork, retry, rollback, derived_from and verified_by. Re-execution is an
explicit future mode: recorded Docker image/command/seed support it but do not promise
bitwise reproducibility across architectures or external tool versions.

## Schema evolution
V0 public schemas use `schema_version: 1.0`. Unsupported versions are rejected rather
than silently coerced. `packages.core.models.migrations` provides an explicit migration
registry for future breaking changes; Alembic independently versions SQL storage.

## Trust and deployment
Docker has networking disabled by default, resource limits, non-root execution and
no privileged mode. The API uses a Docker socket to launch siblings; this is a trusted
single-user deployment and the socket grants host-level control. Bind to loopback.
Telemetry (OpenTelemetry + OpenInference attributes) mirrors events only.

The API uses one worker and an episode lock. Direct SDK users must serialize episode
operations. Stdout/stderr and transfer limits are explicit in README; truncation is
recorded. Branch hashes identify event prefixes, not only reduced entity state.
