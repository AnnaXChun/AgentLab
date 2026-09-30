# scientific-harness

A minimal, agent-agnostic and domain-agnostic harness for evidence-bound scientific
experiments. Python 3.12, Apache-2.0. No LLM account or API key is needed.

An agent returns actions. The harness retrieves evidence, calls MCP tools, executes
code in Docker, stores immutable artifacts, verifies claims, records canonical
Postgres events, and exports replayable trajectories.

## Run the complete demo

Requirements: Docker Engine with Compose, at least 2 CPUs / 4 GiB RAM, and `make`.
The API starts with automatic Alembic migration after Postgres becomes healthy.

```bash
cd scientific-harness
make demo
```

For an existing checkout, start at `cd scientific-harness`. The repository is local;
no remote repository is configured or published by this implementation.

Equivalent commands:

```bash
docker compose up -d --build --wait
docker compose exec -T scientific-harness-api python examples/run_demo.py
```

The API is at http://127.0.0.1:8000; OpenAPI documentation is at
http://127.0.0.1:8000/docs. These are API reference pages, not a product UI.
Postgres is bound to `127.0.0.1:55432` with local development credentials
`harness:harness`, database `harness`. Override `API_PORT` / `POSTGRES_PORT` if needed.

The demo prints its episode ID and writes `outputs/<episode-id>/<export-id>/`.
Paths printed from inside the container use `/data/exports`; that directory maps
to the checkout's `outputs/`. `.artifacts/` maps to `/data/artifacts`.

```text
goal → MockAgent → retrieval → load_dataset MCP → run_mock_simulation MCP
     → immutable experiment.py → Docker upload/execute → result.json artifact
     → UNVERIFIED claim → SUPPORTED → NumericVerifier → VERIFIED
     → checkpoint → branch A: failed command
                  → branch B: restored snapshot → corrected command succeeds
     → main rollback (history retained) → container cleanup → episode completed
     → replay → JSONL / Parquet / SFT / RL-style / failure-recovery exports
```

The mock observations are `[0.18, 0.20, 0.22]`. Their mean is `0.2`, below the
reference threshold `0.5`. This is a synthetic software demonstration.

## Local Python development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Docker.
On macOS, Docker Desktop or Colima works. To create an isolated Colima environment:

```bash
brew install colima docker docker-compose
mkdir -p ~/.docker/cli-plugins
ln -sf "$(brew --prefix)/bin/docker-compose" ~/.docker/cli-plugins/docker-compose
colima start scientific-harness --runtime docker --cpu 2 --memory 4 --disk 20
docker context use colima-scientific-harness
```

The Python Docker adapter reads `DOCKER_HOST` first, then the current Docker CLI
context. It does not assume Docker Desktop's socket location.

```bash
uv sync --python 3.12
docker compose up -d --build --wait
uv run python examples/run_demo.py
uv run pytest -q
HARNESS_INTEGRATION=1 uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Without live Postgres and Docker, integration tests skip with a reason. Setting
`HARNESS_INTEGRATION=1` makes unavailable infrastructure a failure. Unit and contract
tests use SQLite only as a test double for event storage; production storage and
retrieval always use PostgreSQL. MCP contract tests launch real SDK stdio servers.

To run only the local API against the Compose database, stop the container API
first, then run:

```bash
docker compose stop scientific-harness-api
DATABASE_URL=postgresql+psycopg://harness:harness@localhost:55432/harness uv run alembic upgrade head
uv run uvicorn packages.sdk.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Configuration:

| Variable | Local default / meaning |
| --- | --- |
| `DATABASE_URL` | `postgresql+psycopg://harness:harness@localhost:55432/harness` in Runtime |
| `ARTIFACT_ROOT` | `.artifacts` |
| `EXPORT_ROOT` | `outputs` |
| `DOCKER_HOST` | Docker daemon endpoint; otherwise current CLI context |
| `HARNESS_TRACE_CONSOLE=1` | Export OTel spans to stdout |
| `HARNESS_INTEGRATION=1` | Require real integration infrastructure in pytest |

For direct Alembic use, explicitly set the database URL (its ini default uses 5432):

```bash
export DATABASE_URL=postgresql+psycopg://harness:harness@localhost:55432/harness
uv run alembic upgrade head
```

Stop the services with `docker compose down`. The named database volume and local
artifact/export directories are retained. Sandbox containers are removed on normal
episode completion, including failed demo episodes. An abrupt host/process crash
may leave a container labelled `scientific-harness=sandbox`; inspect it before cleanup.

## Architecture and module boundaries

See [GOALS.md](GOALS.md), [ARCHITECTURE.md](ARCHITECTURE.md), and the measured
[V0 delivery report](DELIVERY_REPORT.md).

```text
packages/
  core/
    models/          Pydantic schemas and SQLAlchemy canonical storage models
    events/          EventStore and audited operation recorder
    artifacts/       ArtifactStore protocol
    verification/    Numeric, schema and execution verifiers; claim transitions
    episodes/ evidence/ claims/  Entity boundaries (schemas are centralized in V0)
  mcp_gateway/       Input schema validation, allowlist and MCP SDK client
  retrieval/         Parser/embedding interfaces; Postgres FTS + pgvector RRF
  sandbox/           Command/result schemas and SandboxBackend protocol
  provenance/        Event replay, branch projections and OTel/OpenInference
  exporters/         Five trajectory formats plus edge table and CAS closure
  sdk/               Harness, runtime factory, versioned FastAPI contracts
adapters/
  agents/            AgentAdapter protocol and deterministic MockAgentAdapter
  sandbox/           Docker implementation
  storage/           Atomic local SHA-256 artifact store
domain_packs/
  demo/ materials/ biology/  Manifest, Markdown document and mock capabilities
schemas/             Generated JSON Schemas and OpenAPI
migrations/          Alembic revisions 0001 and 0002
tests/               Unit, contract and real integration tests
examples/            Demo, offline replay and schema generation
```

`AgentAdapter.run_step(context)` has no vendor dependency. `DomainPack` carries
configuration; no scientific-domain branch exists in core. The three shipped packs
use the same synthetic MCP server and generic verifiers. Production domain software
is intentionally outside V0.

`ArtifactStore`, `SandboxBackend`, `Retriever`, `EmbeddingProvider`, `DocumentParser`,
`Verifier` and `TrajectoryExporter` are replaceable interfaces. Inject a custom
verifier registry into `VerifierService`; future Docling support implements
`DocumentParser`. Adding S3 or another sandbox does not require changing core models.

## Persistence and scientific contracts

| Table | Role |
| --- | --- |
| `episodes` | Episode metadata index and row used to serialize event appends |
| `events` | Canonical JSONB envelope, globally ordered within each episode |
| `event_edges` | Typed DAG links, including next/fork/retry/rollback/derived_from/verified_by |
| `retrieval_chunks` | Rebuildable search index: tsvector + vector(64), namespace scoped |
| `alembic_version` | SQL migration revision |

Evidence, Artifact, Claim and Verification records are embedded in immutable events.
Replay derives their current branch state; there are no competing mutable claim
rows. `episode.started` is authoritative for the initial episode; final status comes
from terminal events. Database triggers reject UPDATE, DELETE and TRUNCATE on the
canonical events and edges. An episode row lock serializes concurrent log appends.

Every external experiment operation commits intent before execution and commits a
result or failure afterwards. Payloads over 16 KiB become artifacts. Artifact files
use `.artifacts/sha256/<first-two-hex>/<sha256>`, with atomic no-clobber publication
and hash verification on reads. Commands record argv, uploaded inputs, environment,
seed, image digest/architecture, outputs and errors. Episode metadata includes
Python/dependency versions and a source-tree hash.

`state_before_hash` / `state_after_hash` are branch history accumulator hashes: a
replay position can be recovered from the hashed event prefix. They are not hashes
of only the reduced claim/evidence projection. Export manifests anchor event count
and final hash; offline replay checks that anchor, the event chains and CAS contents.

Claim creation does not accept `status`. Resolved evidence **and** artifact references
are required for SUPPORTED. Verification reads immutable artifact bytes and records
criteria, observed values and claim identity. Only the service emits VERIFIED; failed
criteria emit CONTRADICTED. All transitions are retained. Generic verifiers check the
selected formal criterion, not arbitrary natural-language entailment of claim text.

Public schemas use `schema_version: "1.0"`; unknown versions fail validation.
`packages/core/models/migrations.py` is the explicit future event-migration registry.
SQL evolution is separate and uses Alembic.

## API

All JSON request/response schemas are Pydantic v2. `branch_id` is optional and defaults
to the root branch. No endpoint accepts a caller-authored Event or claim status.

| Method | Path |
| --- | --- |
| POST | `/episodes` |
| GET | `/episodes/{id}` |
| POST | `/episodes/{id}/ingest` |
| POST | `/episodes/{id}/retrieve` |
| POST | `/episodes/{id}/tools/{tool_name}` |
| POST | `/episodes/{id}/sandbox/exec` |
| POST | `/episodes/{id}/claims` |
| POST | `/episodes/{id}/claims/{claim_id}/verify` |
| POST | `/episodes/{id}/checkpoint` |
| POST | `/episodes/{id}/fork` |
| POST | `/episodes/{id}/rollback` |
| POST | `/episodes/{id}/complete` |
| GET | `/episodes/{id}/events` |
| GET | `/episodes/{id}/replay` |
| POST | `/episodes/{id}/export` |

Example creation:

```bash
curl -s http://127.0.0.1:8000/episodes \
  -H 'Content-Type: application/json' \
  -d '{"schema_version":"1.0","goal":"Determine whether candidate A satisfies property X.","domain_pack":"demo","agent_id":"external"}'
```

For sandbox execution, submit `{"command":{"argv":["python","experiment.py"],
"files":{"experiment.py":"print(42)"},"collect":[],"timeout_seconds":30,"seed":0}}`.
Uploaded text files are stored in CAS before upload. Shell behavior is explicit:
use `argv: ["sh", "-c", "..."]` if needed, inside the sandbox.

## Replay and export

```bash
uv run python examples/replay.py outputs/<episode-id>/<export-id>
```

Offline replay requires neither Docker nor Postgres. It reconstructs branches and
claim/evidence/artifact state without re-executing tools. The export directory contains:

| File | Contents |
| --- | --- |
| `raw.jsonl` | One canonical Event per line, preserving artifact pointers |
| `events.parquet` | Flattened event/usage columns, JSON payload columns, typed ref lists |
| `sft.jsonl` | Per-branch visible assistant messages, tool calls and observations |
| `rl.jsonl` | State reference, action, observation, terminal flag, null optional reward |
| `failure_recovery.jsonl` | Failed action → explicit retry edge → successful corrected action |
| `edges.jsonl` | Complete DAG edge table |
| `manifest.json` | Cutoff count/hash and artifact closure |
| `artifacts/` | Portable immutable CAS contents required by the export |
| `demo_summary.json` | Convenience report written by the demo |

Exports include the `export.started` cutoff; `export.completed` is appended to the
live log after the files are written. A later export can include that completion.
SFT does not synthesize hidden chain-of-thought. RL records are data only; no training,
policy optimization or reward model runs. Recovery pairs require an explicit causal
retry edge, not merely adjacent failures and successes.

Inspect Parquet with DuckDB:

```python
import duckdb

with duckdb.connect() as db:
    print(
        db.execute(
            "SELECT event_type, count(*) FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
            ["outputs/<episode>/<export>/events.parquet"],
        ).fetchall()
    )
```

## V0 limits and next steps

- Trusted single-user deployment: the API can reach the Docker daemon through its
  socket. It is bound to loopback and has no multi-tenant authentication. MCP pack
  processes and Python adapters are trusted host plugins; in-process Python cannot
  enforce confinement against a malicious plugin. Pack networking policy applies to
  Docker experiment containers, which have networking disabled.
- One API worker serializes each episode's operations. The SDK expects callers to
  serialize operations on the same episode. Postgres serializes event appends, but
  V0 has no distributed execution leases or atomic transaction across tools and DB.
  A crash can leave an unfinished intent; replay exposes it and never silently
  repeats the external operation.
- Replay reconstructs recorded state. Deterministic re-execution across different
  hardware, image tags, dependency versions or external services is not promised.
- Docker images must provide Python 3.12 at `python`. Workspace 128 MiB, temporary
  storage 64 MiB, individual uploaded/collected files and snapshots 16 MiB; stdout
  and stderr each retain up to 16 MiB with explicit truncation flags. Timeouts are
  0–300 seconds. Snapshot restore rejects symlinks and special files.
- Retrieval uses deterministic 64-dimensional feature hashing, not a pretrained
  semantic embedding model. Ingestion supports UTF-8 text/Markdown, not PDF/Docling.
- Claim verification checks explicit numeric/schema/execution criteria. Scientific
  relevance of evidence and natural-language claim entailment need domain validators.
  The three packs are synthetic; materials and biology contain no real lab software.
- Budget event types and usage fields exist; token/cost/CPU/GPU accounting and budget
  enforcement beyond the mock-agent step limit are not implemented.
- Observability emits SDK spans with OpenInference attributes; a collector/exporter
  can be configured later. No Phoenix UI is included.

V1 priorities: (1) execution leases, crash recovery and idempotency; (2) stronger
criterion/evidence binding and declarative verifier policies; (3) explicit hermetic
re-execution with pinned environments; (4) external AgentAdapter and sandbox/storage
conformance suites; (5) semantic embeddings/Docling and projection indexing;
(6) preference datasets and richer reward provenance, without adding training to core.
