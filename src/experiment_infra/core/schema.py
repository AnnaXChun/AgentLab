from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

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


class Hypothesis(Model):
    hypothesis_id: str = Field(default_factory=uid)
    experiment_id: str
    statement: str = Field(min_length=1)
    rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
    expected_observation: dict[str, Any] = Field(default_factory=dict)
    status: Literal["PROPOSED", "TESTED"] = "PROPOSED"
    parent_hypothesis_id: str | None = None


TrajectorySource = Literal["OFFLINE", "ONLINE_RL", "REPLAY", "BRANCH"]


class PolicyIdentity(Model):
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    training_step: int = Field(ge=0, strict=True)
    rollout_batch_id: str = Field(min_length=1)


class Run(Model):
    trajectory_source: TrajectorySource = "OFFLINE"
    policy_identity: PolicyIdentity | None = None
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

    @model_validator(mode="after")
    def online_identity(self):
        if self.trajectory_source == "ONLINE_RL" and self.policy_identity is None:
            raise ValueError("Online run requires policy identity")
        return self


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


class Action(Model):
    tool: Literal["environment", "mcp", "rag"]
    operation: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_effect: str = ""
    context_refs: list[str] = Field(default_factory=list)
    reasoning_summary: str | None = None


class ActionEvent(Model):
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
    trajectory_source: TrajectorySource = "OFFLINE"
    policy_identity: PolicyIdentity | None = None
    trajectory_id: str
    experiment_id: str
    run_id: str
    agent_backend: str
    steps: list[TrajectoryStep]
    final_status: str
    final_claims: list[Claim] = Field(default_factory=list)
    total_cost: float | None = None

    @model_validator(mode="after")
    def online_identity(self):
        if self.trajectory_source == "ONLINE_RL" and self.policy_identity is None:
            raise ValueError("Online trajectory requires policy identity")
        return self


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


class PolicyStepSpan(Model):
    trajectory_step_id: str | None = None
    response_token_start: StrictInt
    response_token_end: StrictInt
    action_token_start: StrictInt | None = None
    action_token_end: StrictInt | None = None
    tool_response_token_start: StrictInt | None = None
    tool_response_token_end: StrictInt | None = None


class PolicyTrace(PolicyIdentity):
    policy_trace_id: str = Field(default_factory=uid)
    experiment_id: str
    run_id: str
    trajectory_id: str
    trajectory_source: TrajectorySource = "ONLINE_RL"
    prompt_token_ids: list[StrictInt]
    response_token_ids: list[StrictInt]
    response_mask: list[Literal[0, 1]]
    response_logprobs: list[float] | None = None
    step_spans: list[PolicyStepSpan]
    sampling_params: dict[str, Any]
    created_at: datetime = Field(default_factory=now)

    @model_validator(mode="after")
    def integrity(self):
        length = len(self.response_token_ids)
        if len(self.response_mask) != length:
            raise ValueError("Response IDs and mask must align exactly")
        if self.response_logprobs is not None and len(self.response_logprobs) != length:
            raise ValueError("Response logprobs must align exactly")
        if any(x < 0 for x in self.prompt_token_ids + self.response_token_ids):
            raise ValueError("Token IDs must be nonnegative")
        last = 0
        covered = set()
        seen = set()
        for span in self.step_spans:
            start, end = span.response_token_start, span.response_token_end
            if not last <= start < end <= length:
                raise ValueError("Generation spans must be ordered and nonoverlapping")
            if any(x != 1 for x in self.response_mask[start:end]):
                raise ValueError("Generation span covers non-policy tokens")
            covered.update(range(start, end))
            last = end
            for prefix, mask in (("action", 1), ("tool_response", 0)):
                a, b = getattr(span, prefix + "_token_start"), getattr(span, prefix + "_token_end")
                if (a is None) != (b is None):
                    raise ValueError("Span endpoints must both be present or absent")
                if a is not None:
                    if not 0 <= a < b <= length or any(x != mask for x in self.response_mask[a:b]):
                        raise ValueError("Invalid token span or response mask")
                    if prefix == "action" and not start <= a < b <= end:
                        raise ValueError("Action must belong to its generation")
                    if prefix == "tool_response":
                        if a < end:
                            raise ValueError("Tool response precedes its action")
                        last = b
            if span.trajectory_step_id:
                if span.trajectory_step_id in seen or span.action_token_start is None:
                    raise ValueError("Each environment step needs one known action span")
                seen.add(span.trajectory_step_id)
        if covered != {i for i, value in enumerate(self.response_mask) if value == 1}:
            raise ValueError("Every policy token must belong to a recorded generation span")
        return self


class RewardResult(Model):
    reward_id: str = Field(default_factory=uid)
    experiment_id: str
    run_id: str
    scalar_reward: float = Field(ge=0, le=1)
    reward_signals: dict[str, float]
    verification_ids: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    reward_source: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


OBJECT_IDS = {
    "PolicyTrace": "policy_trace_id",
    "RewardResult": "reward_id",
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
