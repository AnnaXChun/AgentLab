# Architecture convergence: Agent Experiment Infrastructure

## Existing repository analysis

The working system is episode/event centered. There is no separate Task, Experiment,
Hypothesis or Run model. `Episode.goal` acts as the task. `DomainPack.environment`
and `SandboxBackend` describe execution; `DockerSandboxBackend` implements it.
`Harness.run_agent` owns an agent loop and dispatch, while `MockAgentAdapter` owns
a scientific demonstration strategy. Replay currently folds events without executing
commands. Checkpoint/fork/rollback already preserve append-only branch history.

Directly retain:
- PostgreSQL SQLAlchemy event persistence, hash chains, append-only triggers and Alembic.
- Local SHA-256 CAS and artifact integrity checks.
- Docker command execution, timeout, upload/download, snapshot and restore.
- MCP SDK gateway, allowlist and schema validation; Postgres FTS/pgvector retrieval.
- OpenTelemetry/OpenInference and the existing FastAPI service.
- Legacy episode exports and tests as compatibility coverage.

Adjust responsibilities:
- Move the agent-driving loop out of Harness to an external adapter driver. The new
  ExperimentRuntime accepts a single explicit action and never chooses the next one.
- Introduce Experiment/Hypothesis/Run and typed Action/Observation/Trajectory schemas.
- Preserve legacy retrieval Evidence (a document chunk) as context. New experimental
  Evidence binds a verifier outcome to immutable artifacts, observations and runs.
- Introduce new Claim semantics: SUPPORTED requires successful verification;
  REFUTED/INCONCLUSIVE/UNVERIFIED replace the old pre-verification SUPPORTED meaning.
- Wrap DockerSandboxBackend in an EnvironmentBackend providing observe and snapshot
  diff. Do not implement another container runtime.
- Add object-level provenance and actual exact/branch execution replay on the same log.

## Implementation plan and boundaries

Use `src/experiment_infra/` for the converged public layer, keeping proven storage and
integration implementations at their current paths. Experiment IDs map to existing
physical episode partitions; the legacy envelope remains schema 1.0, while new public
objects use schema 2.0. The distinction is explicit, not a silent reinterpretation.

A small additive migration indexes object identity and provenance edges. Entity versions
and canonical trajectory steps remain in append-only events. Registry/index writes
must commit with their canonical event. Old episodes are not rewritten and cannot be
silently upgraded to verified experimental evidence.

Implement the end-to-end slice first: external scripted backend → hypothesis → run →
source/config artifacts → deterministic operation-count experiment → metrics → independent
metric/test verifiers → evidence → claim → trajectory → actual Docker replay → snapshot
branch changing N → provenance → training-data exports. Use operation counts rather than
wall-clock timing as the deterministic hypothesis; report wall time separately.

The system has no planner, persona, generic reasoning, task decomposition, autonomous
multi-agent coordination or training. MCP and RAG are ordinary action integrations.
Codex/Claude/GPT/Qwen are interchangeable clients of get_state/execute, not core logic.

## Compatibility and constraints

Existing /episodes and v1 schema remain legacy compatibility APIs; new work uses
/experiments and /runs. Existing fixtures and the old demo remain runnable through an
external driver. Do not move files solely to fit a directory diagram.

Exact replay reuses initial snapshot, code/config artifacts, seed, environment image
ID, resource spec and recorded executable actions in a fresh run. Compare collected
output hashes and exit codes. External MCP/RAG calls are replayed explicitly and may
vary; report differences rather than claiming determinism. Branch starts from the
selected completed step's snapshot and accepts new actions. No credit assignment is
computed; CounterfactualLink is a versioned contract for declared interventions.

## Implemented outcome

The vertical slice is implemented in `src/experiment_infra/`. The original 24 tests
remain regression coverage. New tests exercise schema round-trips, immutable artifact
versions, object DAG integrity, Claim safety, real exact/branch replay, portable export
contents, HTTP endpoints and both retained MCP/RAG integrations.

Environment-specific pinning now belongs to the backend's observation metadata:
`spec`, `fingerprint`, and optional `snapshot_mime_type`. The executor does not inspect
Docker image internals or assume that snapshot bytes are tar. A test-only memory
backend proves execution and replay without Docker; Docker remains the sole production
implementation. CLI-style exec actions are the implemented V0 operation vocabulary.

Two first-class indexes were added in migration 0003; the existing byte store, event
store, Docker implementation, retrieval SQL and MCP SDK transport were reused. No new
vector database, planner, model client or training subsystem was introduced.
