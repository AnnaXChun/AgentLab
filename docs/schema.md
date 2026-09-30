# Public schema 2.0 and persistence compatibility

Schema definitions are in `src/experiment_infra/core/schema.py`; machine-readable
contracts are generated in `schemas/v2/`. Every object rejects unknown fields and
versions, uses JSON-serializable data and timezone-aware timestamps by default.

| Object | Main fields |
| --- | --- |
| Experiment | experiment_id, goal, hypothesis_ids, status, constraints, success_criteria, budget, environment_spec, agent_backend, parent_experiment_id, created_at, finished_at |
| Hypothesis | hypothesis_id, experiment_id, statement, rationale, assumptions, expected_observation, status, parent_hypothesis_id |
| Run | run_id, experiment_id, hypothesis_ids, environment_snapshot_id, environment_handle, environment_spec, code_version, config, seed, resource_spec, status, started_at, finished_at, parent_run_id, parent_step_id, trajectory_id |
| Action | tool, operation, arguments, expected_effect, context_refs, optional reasoning_summary |
| ActionEvent | event_id, experiment_id, run_id, step_id, tool, operation, arguments, expected_effect, agent_backend, timestamp, context_refs, optional reasoning_summary |
| ObservationEvent | event_id, action_event_id, stdout, stderr, structured_output, exit_code, environment_diff, created_artifacts, timestamp |
| Artifact | artifact_id, type, uri, sha256, size, mime_type, parent_artifact_ids, producer_run_id, producer_step_id, metadata, created_at |
| Evidence | evidence_id, experiment_id, hypothesis_id, source_artifact_ids, source_run_ids, source_observation_ids, metric, value, uncertainty, verifier, verification_id, verification_status, metadata |
| Claim | claim_id, experiment_id, statement, evidence_ids, confidence, status |
| VerificationResult | verification_id, experiment_id, run_id, verifier, status, source_artifact_ids, source_observation_ids, metric, value, uncertainty, details, created_at |
| Snapshot | snapshot_id, experiment_id, run_id, step_id, artifact_id, environment, created_at |
| TrajectoryStep | trajectory_id, step_id, state_before, context_refs, action, observation, artifact_ids, evidence_ids, state_after, verifier_feedback, reward_signals, latency, token_usage, compute_usage, monetary_cost, parent_step_id, branch_id |
| Trajectory | trajectory_id, experiment_id, run_id, agent_backend, steps, final_status, final_claims, total_cost |
| CounterfactualLink | link_id, original_step_id, counterfactual_step_id, intervention_type, removed_context_refs, modified_context_refs, original_outcome, counterfactual_outcome, outcome_delta |
| DomainPack | name, tools, environment, artifact_types, verifier, templates |

Artifact IDs identify immutable versions; SHA-256 addresses the bytes. Equal bytes
may have different producer/version IDs with the same hash. Changed bytes produce a
new hash and a new object with parent_artifact_ids. Existing identities cannot be
rewritten through the experiment ledger. The existing LocalArtifactStore remains the
only byte store. Artifact types include source_code, config, dataset_reference,
checkpoint, log, metric, table, plot, report and generated_file.

A Claim with no evidence is UNVERIFIED. All referenced Evidence must resolve in the
same Experiment to real VerificationResult and Artifact records. All PASSED supports
a Claim; all FAILED refutes it; mixed/error/inconclusive results give INCONCLUSIVE.
The new model deliberately has no VERIFIED status. Claim status cannot be supplied
through HTTP. Retrieval chunk IDs do not resolve as experimental evidence.

## SQL migration

Alembic `0003` adds `experiment_objects` (object_id PK, experiment_id FK, kind,
created_event_id FK) and `provenance_edges` (source_id/target_id/relation composite PK,
created_event_id FK). Both receive the existing append-only mutation-rejection trigger.
Object identity and DAG edges commit in the same SQL transaction as their canonical
Event. Cycle or scope validation failure rolls back the event and index changes.

Canonical entity versions live in `events.payload`, spilling to CAS when large.
Indexes locate objects; they do not replace event-sourced truth. An Experiment uses
an existing episode partition so old storage, event hash chains and SQL constraints
are reused. The event envelope remains schema_version 1.0; nested experiment objects
use schema_version 2.0. EventType adds explicit experiment/run/action/observation/
trajectory/evidence names; the SQL column is text and requires no enum rewrite.

Existing data is left intact. There is no implicit 1.0 Claim→2.0 Claim migration:
legacy SUPPORTED did not require successful verification. A future data upgrade must
explicitly establish artifact/run/observation evidence, not promote legacy labels.
Runtime rejects using legacy episode write APIs against a 2.0 Experiment partition.

The hypotheses/run statuses are projections over immutable versions. Trajectory
queries join separately recorded verifier feedback and final claims into steps;
verification does not mutate the original action or observation event history.

EnvironmentSpec uses a provider name rather than a Docker-only enum, an optional
image, settings, resource limits and dependencies. EnvironmentBackend.observe returns
its pinned spec and fingerprint. Docker derives these from the image digest,
architecture and Python; another backend may use different metadata and snapshot
encoding without changing the experiment schemas or ledger.
