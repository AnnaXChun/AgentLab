# AgentLab: Agent Experiment Infrastructure

The execution, evidence and data layer underneath autonomous experimentation.
An external agent chooses an action; this runtime executes it and records what happened.
AI4S is the first domain showcase. Python 3.12, Apache-2.0, no LLM API key required.

Version 0.3 adds pinned ScienceAgentBench and tau2 telecom-solo adapters, an
OpenAI-compatible structured-action backend, and a resumable batch runner.
See [benchmark setup and protocols](docs/benchmarks.md) and
[benchmark selection](docs/benchmark_selection.md).
Environment calibration is separate from real-model baseline measurement.
See the [0.3 delivery report](docs/benchmark_delivery.md) for installed environments,
measured calibration results, replay checks and current limits.

```python
state = await runtime.get_state(experiment_id, run_id)
action = await agent_backend.next_action(state)
step = await runtime.execute(experiment_id, action, run_id=run_id)
```

The runtime does not contain a planner, agent persona, task decomposition framework,
prompt framework or model training loop. Codex, Claude Code, GPT, Qwen and custom agents
can submit actions over HTTP or the Python API.

## Run the end-to-end experiment

With Docker/Compose available:

```bash
make demo
```

This builds the API, starts PostgreSQL/pgvector, automatically migrates to Alembic
revision `0003`, and runs `examples/toy_experiment/run.py` inside the service.

For the requested local Python entry point:

```bash
uv sync --python 3.12
docker compose up -d --build --wait
source .venv/bin/activate
python examples/toy_experiment/run.py
pytest
```

The demo is deterministic: algorithm A sums integers using N additions, algorithm B
uses a closed-form sum with one addition. The hypothesis concerns **addition counts**,
not wall-clock speed. Unit tests independently check correctness for several N values.

The external scripted client performs:

1. Create Experiment and Hypothesis; start a Run with `N=2000`.
2. Upload two Python implementations and a benchmark; run in real Docker.
3. Save source/config/metrics/log artifacts in SHA-256 CAS.
4. Run tests separately; MetricVerifier and TestVerifier generate typed Evidence.
5. Create a SUPPORTED Claim and persistent Trajectory.
6. Exact replay in a fresh container using the pinned image and initial snapshot.
7. Branch after step 1, change `N=4000`, run inherited code and compare outcomes.
8. Confirm original trajectory is unchanged; validate provenance and export data.

Expected metrics:

| N | Algorithm A additions | Algorithm B additions | Both sums |
| ---: | ---: | ---: | ---: |
| 2000 | 2000 | 1 | 2,001,000 |
| 4000 | 4000 | 1 | 8,002,000 |

Exports are in `outputs/experiments/<experiment-id>/<export-id>/`:

```text
trajectory.jsonl       One typed Trajectory per run, with canonical steps
artifacts.jsonl        Immutable artifact version records
evidence.jsonl         Verifier-generated experimental evidence
sft.jsonl              state/context → action
rl_transition.jsonl    state/action/observation/reward_signals/next_state
provenance.jsonl       Object graph
events.jsonl           Full underlying append-only event envelopes
manifest.json          File hashes, event cutoff/hash, artifact closure
artifacts/             Portable CAS payloads
summary.json           Demo results and replay comparisons
```

`reward_signals` is empty. No RL/SFT training or hidden chain-of-thought generation runs.
The summary includes exact replay comparisons, claim statuses and the unchanged-original
check. Container paths `/data/exports` map to local `outputs/`.

## Architecture

Read [architecture analysis and refactor plan](docs/architecture_refactor.md),
[schema contracts](docs/schema.md), [provenance](docs/provenance.md), and
[replay semantics](docs/replay.md). The measured results are in
[the refactor delivery report](docs/refactor_delivery.md).

```text
src/experiment_infra/
  core/          Experiment/Hypothesis/Run/Action/Observation/Artifact/Evidence/
                 Claim/Trajectory/Counterfactual schemas, ledger and provenance
  runtime/       Single-action executor, exact/branch replay, FastAPI routes
  environments/  Replaceable protocol + wrapper around the existing Docker adapter
  agents/        External AgentBackend protocol and caller callback bridge
  verifiers/     Independent MetricVerifier and TestVerifier; future protocols
  domains/       science_demo configuration/templates
  exporters/     Canonical entities, SFT and RL transitions
packages/        Retained event/CAS contracts, MCP/RAG, tracing and legacy API
adapters/        Retained Docker/CAS implementations and legacy external agent driver
migrations/      0001, 0002 retained; additive 0003 object/provenance indexes
schemas/v2/      Generated JSON Schemas for the new public layer
examples/toy_experiment/
tests/unit/ tests/contract/ tests/integration/
```

Public experiment objects use `schema_version: "2.0"`. Existing event envelopes and
legacy schemas remain `1.0`; no historical bytes or claim statuses are reinterpreted.
Experiment IDs reuse the existing physical episode partition. Object identity and
provenance indexes commit atomically with their canonical events. The database rejects
UPDATE/DELETE/TRUNCATE on event history and indexes.

`Artifact` versions have separate IDs, SHA-256 contents, parent artifact IDs and producer
run/step IDs. `Evidence` references immutable artifacts, runs, observations and an
independent VerificationResult. A retrieval document chunk is context, never automatically
verified experimental Evidence. New Claim statuses are SUPPORTED, REFUTED, INCONCLUSIVE
and UNVERIFIED. Only verifier-generated Evidence can support or refute a Claim.

## API

The API runs at http://127.0.0.1:8000, reference docs at http://127.0.0.1:8000/docs.

| Method | Path |
| --- | --- |
| POST / GET | `/experiments`, `/experiments/{id}` |
| GET | `/experiments/{id}/state?run_id=...` |
| POST | `/experiments/{id}/hypotheses` |
| POST | `/experiments/{id}/runs` |
| POST | `/experiments/{id}/actions` |
| POST | `/experiments/{id}/claims` |
| POST | `/experiments/{id}/verify` |
| POST | `/runs/{id}/replay` |
| POST | `/runs/{id}/branch` |
| POST | `/runs/{id}/finish`, `/experiments/{id}/finish` |
| GET | `/objects/{id}/provenance` |
| GET | `/trajectories/{id}` |
| POST | `/exports` |

Create an experiment:

```bash
curl -s http://127.0.0.1:8000/experiments \
  -H 'Content-Type: application/json' \
  -d '{"goal":"Compare algorithms","agent_backend":"external","success_criteria":{"metric":{"field":"score","operator":"gt","baseline":0.5}}}'
```

Submit an action:

```json
{
  "schema_version": "2.0",
  "run_id": "<run-id>",
  "action": {
    "tool": "environment",
    "operation": "exec",
    "arguments": {
      "command": {
        "argv": ["python", "benchmark.py"],
        "files": {"benchmark.py": "print(42)"},
        "seed": 0
      }
    },
    "expected_effect": "Execute the supplied experiment code"
  }
}
```

A branch request accepts `{"from_step":1,"config":{"N":4000}}`. Step numbers are
one-based and refer to the state **after** a completed step. Step UUIDs also work.
An export request accepts `{"experiment_id":"..."}`; output roots are server configured.
Callers cannot supply Claim status, Evidence, VerificationResult or raw canonical events.

## Replaceable components

`ExperimentRuntime(environment=custom_environment)` does not require a Docker client.
The only concrete environment in this release is Docker. MCP and RAG are opt-in injected
components using the retained implementations; no new vector database layer was added.
For example, a host can use `storage.components(load_pack("demo"))` to obtain the existing
retriever/gateway and pass them into `ExperimentRuntime(storage, retriever=..., gateway=...)`.
Pass this configured runtime as `create_app(storage, experiment_runtime=...)` for HTTP.

MCP uses `Action(tool="mcp", operation="call", arguments={"tool_name":...,"arguments":...})`;
RAG uses `Action(tool="rag", operation="retrieve", arguments={"query":...,"top_k":...})`.
Both produce the same ActionEvent, ObservationEvent, artifact and TrajectoryStep contracts.
Without an injected integration, these operations record an explicit failed observation.

`Verifier.verify(experiment, run, artifacts, claims)` is independent of the agent.
MetricVerifier uses the experiment's declared criterion. TestVerifier accepts only
runtime-produced logs of the declared test command. LLM/Human/Statistical/Deterministic
extension protocols are present; no such external services or workflows are implemented.

## Development and compatibility

```bash
HARNESS_INTEGRATION=1 uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python examples/generate_schemas.py
```

With infrastructure unavailable, integration tests skip unless `HARNESS_INTEGRATION=1`,
which makes the missing environment a failure. SQLite is used only for isolated tests;
production uses Postgres. The live tests exercise actual Docker, MCP and pgvector.

Configuration remains `DATABASE_URL`, `ARTIFACT_ROOT`, `EXPORT_ROOT`, `DOCKER_HOST` and
`HARNESS_TRACE_CONSOLE`. Defaults are Postgres at `localhost:55432`, `.artifacts/` and
`outputs/`. Python Docker reads the current Docker CLI context when DOCKER_HOST is absent.
On this machine, Colima profile `scientific-harness` provides the Docker daemon.

`make legacy-demo` retains the earlier showcase; [README.v0.md](README.v0.md) describes
its historical model. `Harness.run_agent()` was removed: its loop now resides in
`adapters/agents/driver.py`. Existing `/episodes` routes remain compatibility APIs and
reject writes to new experiment partitions. New work should use `/experiments`.

## Scope and limits

- Trusted single-user deployment, one API worker. HTTP operations are serialized;
  direct SDK callers must serialize lifecycle/verification operations. Event appends
  use database row locks, but there are no distributed execution leases.
- Exact replay executes recorded actions. It pins the initial image, architecture,
  Python, resources, config, seed and snapshot. It compares output hashes/exit codes
  and normal stdout/stderr. Declared test-runner timing text is reported separately;
  test outcome comparison uses exit code and collected artifacts.
- External retrieval/tools and nondeterministic programs can differ on replay; the
  result reports mismatches. No promise of arbitrary hardware/physical-system replay.
- Docker snapshots cover `/work`, not process memory, external services or databases.
  Symlinks/special files are rejected. A snapshot failure remains in the trajectory,
  is unbranchable, and marks the run failed even if the command exited successfully.
- Generic verification checks declared criteria, not arbitrary natural-language claim
  entailment. Test quality and the scientific validity of a metric remain domain policy.
- CounterfactualLink declares six intervention types. This slice implements snapshot
  branches and caller-provided alternate actions; no automatic context ablation or credit
  assignment engine. Rewards/costs/token/compute fields are available but unmeasured values
  remain empty/null. Budgets/constraints are descriptive metadata beyond Docker limits.
- Docker uses the existing 16 MiB transfer/snapshot and stdout/stderr limits, explicit
  truncation flags, 128 MiB work directory, disabled network and bounded execution time.
- Crashes can leave pending intents or containers; automatic recovery is not implemented.
  Projections currently fold the event log, which favors correctness over high-volume speed.
- There is no planner, multi-agent framework, frontend, model training, HPC integration,
  external model adapter, new vector database or autonomous-scientist benchmark.
