import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiment_infra.verifiers.policy import SuccessCriterion, criteria_by_id
from packages.core.models.schemas import now, uid


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["2.0"] = "2.0"


class EnvironmentSpec(Model):
    backend: str = Field(default="docker", min_length=1)
    image: str | None = "python:3.12-slim"
    settings: dict[str, Any] = Field(default_factory=dict)
    cpu: float = Field(default=1, gt=0)
    memory_gb: float = Field(default=1, gt=0)
    dependencies: dict[str, str] = Field(default_factory=dict)
    network: Literal["disabled"] = "disabled"


class Experiment(Model):
    criteria: list[SuccessCriterion] = Field(default_factory=list)
    experiment_id: str = Field(default_factory=uid)
    goal: str = Field(min_length=1, max_length=8000)
    hypothesis_ids: list[str] = Field(default_factory=list)
    status: Literal["CREATED", "RUNNING", "COMPLETED", "FAILED"] = "CREATED"
    constraints: dict[str, Any] = Field(default_factory=dict)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    environment_spec: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    agent_backend: str = "external"
    parent_experiment_id: str | None = None
    created_at: datetime = Field(default_factory=now)
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def unique_criteria(self):
        criteria_by_id(self)
        return self


class Hypothesis(Model):
    criterion_ids: list[str] = Field(default_factory=list)
    hypothesis_id: str = Field(default_factory=uid)
    experiment_id: str
    statement: str = Field(min_length=1)
    rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
    expected_observation: dict[str, Any] = Field(default_factory=dict)
    status: Literal["PROPOSED", "TESTED"] = "PROPOSED"
    parent_hypothesis_id: str | None = None


RunPurpose = Literal["experiment", "exact_replay", "branch", "unknown"]


class Run(Model):
    purpose: RunPurpose = "unknown"
    run_id: str = Field(default_factory=uid)
    experiment_id: str
    hypothesis_ids: list[str] = Field(default_factory=list)
    environment_snapshot_id: str | None = None
    environment_handle: str | None = None
    environment_spec: EnvironmentSpec
    code_version: str
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    resource_spec: dict[str, Any] = Field(default_factory=dict)
    status: Literal["RUNNING", "COMPLETED", "FAILED"] = "RUNNING"
    started_at: datetime = Field(default_factory=now)
    finished_at: datetime | None = None
    parent_run_id: str | None = None
    parent_step_id: str | None = None
    trajectory_id: str = Field(default_factory=uid)


ArtifactType = Literal[
    "source_code",
    "config",
    "dataset_reference",
    "checkpoint",
    "log",
    "metric",
    "table",
    "plot",
    "report",
    "generated_file",
]


class Artifact(Model):
    artifact_id: str = Field(default_factory=uid)
    type: ArtifactType
    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    mime_type: str
    parent_artifact_ids: list[str] = Field(default_factory=list)
    producer_run_id: str
    producer_step_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now)


class DecisionContext(Model):
    """Immutable serialized input. Coverage is client-attested, not model-use proof."""

    context_id: str = Field(default_factory=uid)
    experiment_id: str
    run_id: str
    payload_json: str
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage: Literal["complete", "partial", "unknown"] = "unknown"
    source_refs: tuple[str, ...] = ()
    last_step_id: str | None = None
    created_at: datetime = Field(default_factory=now)

    @model_validator(mode="after")
    def valid_payload(self):
        data = json.loads(self.payload_json)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical != self.payload_json:
            raise ValueError("Decision payload must be canonical JSON")
        if hashlib.sha256(canonical.encode()).hexdigest() != self.payload_sha256:
            raise ValueError("Decision payload hash mismatch")
        return self

    @property
    def payload(self):
        return json.loads(self.payload_json)


class Action(Model):
    decision_context_id: str | None = None
    tool: Literal["environment", "mcp", "rag"]
    operation: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_effect: str = ""
    context_refs: list[str] = Field(default_factory=list)
    reasoning_summary: str | None = None


class ActionEvent(Model):
    decision_context_id: str | None = None
    event_id: str = Field(default_factory=uid)
    experiment_id: str
    run_id: str
    step_id: str
    tool: str
    operation: str
    arguments: dict[str, Any]
    expected_effect: str
    agent_backend: str
    timestamp: datetime = Field(default_factory=now)
    context_refs: list[str] = Field(default_factory=list)
    reasoning_summary: str | None = None


class ObservationEvent(Model):
    event_id: str = Field(default_factory=uid)
    action_event_id: str
    stdout: str = ""
    stderr: str = ""
    structured_output: dict[str, Any] = Field(default_factory=dict)
    exit_code: int | None = None
    environment_diff: dict[str, Any] = Field(default_factory=dict)
    created_artifacts: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=now)


class Snapshot(Model):
    snapshot_id: str = Field(default_factory=uid)
    experiment_id: str
    run_id: str
    step_id: str | None = None
    artifact_id: str
    environment: dict[str, Any]
    created_at: datetime = Field(default_factory=now)


class VerificationResult(Model):
    hypothesis_id: str | None = None
    criterion_id: str | None = None
    policy_id: str | None = None
    policy_sha256: str | None = None
    trust_level: Literal["agent_reported", "runtime_observed", "host_evaluated"] = "agent_reported"

    verification_id: str = Field(default_factory=uid)
    experiment_id: str
    run_id: str
    verifier: str
    status: Literal["PASSED", "FAILED", "INCONCLUSIVE", "ERROR"]
    source_artifact_ids: list[str]
    source_observation_ids: list[str] = Field(default_factory=list)
    metric: str
    value: Any = None
    uncertainty: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now)


class Evidence(Model):
    criterion_id: str | None = None
    policy_id: str | None = None
    policy_sha256: str | None = None
    trust_level: Literal["agent_reported", "runtime_observed", "host_evaluated"] = "agent_reported"

    evidence_id: str = Field(default_factory=uid)
    experiment_id: str
    hypothesis_id: str
    source_artifact_ids: list[str] = Field(min_length=1)
    source_run_ids: list[str] = Field(min_length=1)
    source_observation_ids: list[str] = Field(default_factory=list)
    metric: str
    value: Any
    uncertainty: float | None = None
    verifier: str
    verification_id: str
    verification_status: Literal["PASSED", "FAILED", "INCONCLUSIVE", "ERROR"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(Model):
    hypothesis_id: str | None = None
    criterion_ids: list[str] = Field(default_factory=list)
    assessment_scope: Literal["unbound_legacy", "criterion_bound"] = "unbound_legacy"
    claim_id: str = Field(default_factory=uid)
    experiment_id: str
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: Literal["SUPPORTED", "REFUTED", "INCONCLUSIVE", "UNVERIFIED"] = "UNVERIFIED"

    @model_validator(mode="after")
    def evidence_required(self):
        if self.status != "UNVERIFIED" and not self.evidence_ids:
            raise ValueError("Claim status requires verifier-generated evidence")
        return self


class TrajectoryStep(Model):
    decision_context: DecisionContext | None = None
    trajectory_id: str
    step_id: str = Field(default_factory=uid)
    state_before: dict[str, Any]
    context_refs: list[str] = Field(default_factory=list)
    action: ActionEvent
    observation: ObservationEvent
    artifact_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    state_after: dict[str, Any]
    verifier_feedback: list[VerificationResult] = Field(default_factory=list)
    reward_signals: dict[str, Any] = Field(default_factory=dict)
    latency: float
    token_usage: dict[str, Any] | None = None
    compute_usage: dict[str, Any] | None = None
    monetary_cost: float | None = None
    parent_step_id: str | None = None
    branch_id: str


class Trajectory(Model):
    purpose: RunPurpose = "unknown"
    lineage_root_experiment_id: str | None = None
    trajectory_id: str
    experiment_id: str
    run_id: str
    agent_backend: str
    steps: list[TrajectoryStep]
    final_status: str
    final_claims: list[Claim] = Field(default_factory=list)
    total_cost: float | None = None


class CounterfactualLink(Model):
    link_id: str = Field(default_factory=uid)
    original_step_id: str
    counterfactual_step_id: str
    intervention_type: Literal[
        "context_remove",
        "context_mask",
        "artifact_remove",
        "tool_result_replace",
        "action_replace",
        "state_restore",
    ]
    removed_context_refs: list[str] = Field(default_factory=list)
    modified_context_refs: list[str] = Field(default_factory=list)
    original_outcome: dict[str, Any] = Field(default_factory=dict)
    counterfactual_outcome: dict[str, Any] = Field(default_factory=dict)
    outcome_delta: dict[str, Any] = Field(default_factory=dict)


class DomainPack(Model):
    name: str
    tools: list[dict[str, Any]] = Field(default_factory=list)
    environment: EnvironmentSpec = Field(default_factory=EnvironmentSpec)
    artifact_types: list[ArtifactType] = Field(default_factory=list)
    verifier: list[str] = Field(default_factory=list)
    templates: dict[str, Any] = Field(default_factory=dict)


class ReplayResult(Model):
    original_run_id: str
    replay_run_id: str
    matched: bool
    comparisons: list[dict[str, Any]]


OBJECT_IDS = {
    "DecisionContext": "context_id",
    "Experiment": "experiment_id",
    "Hypothesis": "hypothesis_id",
    "Run": "run_id",
    "ActionEvent": "event_id",
    "ObservationEvent": "event_id",
    "Artifact": "artifact_id",
    "Snapshot": "snapshot_id",
    "VerificationResult": "verification_id",
    "Evidence": "evidence_id",
    "Claim": "claim_id",
    "TrajectoryStep": "step_id",
    "Trajectory": "trajectory_id",
    "CounterfactualLink": "link_id",
}
OBJECT_TYPES = {name: globals()[name] for name in OBJECT_IDS}
