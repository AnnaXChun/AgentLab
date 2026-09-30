from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def uid() -> str:
    return str(uuid4())


def now() -> datetime:
    return datetime.now(timezone.utc)


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"


class EventType(StrEnum):
    EXPERIMENT_CREATED = "experiment.created"
    EXPERIMENT_UPDATED = "experiment.updated"
    HYPOTHESIS_REGISTERED = "hypothesis.registered"
    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    ACTION_REQUESTED = "action.requested"
    OBSERVATION_RECORDED = "observation.recorded"
    TRAJECTORY_RECORDED = "trajectory.recorded"
    OBJECT_RECORDED = "object.recorded"
    ENVIRONMENT_OPERATION = "environment.operation"
    EVIDENCE_GENERATED = "evidence.generated"
    CLAIM_ASSESSED = "claim.assessed"
    COUNTERFACTUAL_LINKED = "counterfactual.linked"
    REPLAY_RESULT = "replay.result"
    EPISODE_STARTED = "episode.started"
    EPISODE_COMPLETED = "episode.completed"
    EPISODE_FAILED = "episode.failed"
    AGENT_MESSAGE = "agent.message"
    AGENT_ACTION = "agent.action"
    RETRIEVAL_INGEST = "retrieval.ingest"
    RETRIEVAL_INGESTED = "retrieval.ingested"
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_RESULT = "retrieval.result"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    SANDBOX_COMMAND = "sandbox.command"
    SANDBOX_RESULT = "sandbox.result"
    ARTIFACT_CREATED = "artifact.created"
    CLAIM_CREATED = "claim.created"
    CLAIM_UPDATED = "claim.updated"
    VERIFICATION_STARTED = "verification.started"
    VERIFICATION_RESULT = "verification.result"
    CHECKPOINT_CREATED = "checkpoint.created"
    BRANCH_CREATED = "branch.created"
    ROLLBACK_CREATED = "rollback.created"
    BUDGET_UPDATED = "budget.updated"
    EXPORT_STARTED = "export.started"
    EXPORT_COMPLETED = "export.completed"


class ClaimStatus(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"


class Usage(Schema):
    wall_time_ms: float = 0
    cpu_seconds: float | None = None
    gpu_seconds: float | None = None
    tokens: int | None = None
    estimated_cost_usd: float | None = None


class Episode(Schema):
    episode_id: str = Field(default_factory=uid)
    goal: str = Field(min_length=1, max_length=8000)
    domain_pack: str
    agent_id: str
    status: Literal["RUNNING", "COMPLETED", "FAILED"] = "RUNNING"
    created_at: datetime = Field(default_factory=now)
    completed_at: datetime | None = None
    root_branch_id: str = Field(default_factory=uid)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Event(Schema):
    episode_id: str
    event_id: str = Field(default_factory=uid)
    sequence: int = 0
    timestamp: datetime = Field(default_factory=now)
    trace_id: str = ""
    span_id: str = ""
    branch_id: str
    parent_event_id: str | None = None
    actor_type: str = "harness"
    actor_id: str = "scientific-harness"
    event_type: EventType
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    state_before_hash: str = ""
    state_after_hash: str = ""
    status: Literal["PENDING", "SUCCESS", "FAILED"] = "SUCCESS"
    usage: Usage = Field(default_factory=Usage)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Artifact(Schema):
    artifact_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str
    mime_type: str
    size_bytes: int = Field(ge=0)
    created_by_event_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(Schema):
    evidence_id: str
    document_id: str
    document_hash: str
    source_uri: str
    title: str
    section: str | None = None
    page: int | None = None
    chunk_id: str
    content: str
    retrieval_score: float
    retrieved_at: datetime = Field(default_factory=now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(Schema):
    claim_id: str = Field(default_factory=uid)
    episode_id: str
    text: str
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    evidence_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    verification_refs: list[str] = Field(default_factory=list)
    created_by_event_id: str

    @model_validator(mode="after")
    def valid_support(self):
        if self.status != ClaimStatus.UNVERIFIED and (
            not self.evidence_refs or not self.artifact_refs
        ):
            raise ValueError("Supported scientific claims require evidence and artifacts")
        if self.status == ClaimStatus.VERIFIED and not self.verification_refs:
            raise ValueError("Verified claims require verification references")
        return self


class Verification(Schema):
    verification_id: str = Field(default_factory=uid)
    claim_id: str
    verifier_type: str
    status: Literal["PASSED", "FAILED", "ERROR"]
    score: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now)


class EventEdge(Schema):
    source_event_id: str
    target_event_id: str
    relation: Literal["next", "fork", "retry", "rollback", "derived_from", "verified_by"]


class AgentAction(Schema):
    kind: Literal["retrieve", "tool", "sandbox", "claim", "verify", "message", "finish"]
    arguments: dict[str, Any] = Field(default_factory=dict)


class DomainPack(Schema):
    api_version: Literal["sciharness/v1"] = "sciharness/v1"
    kind: Literal["DomainPack"] = "DomainPack"
    metadata: dict[str, str]
    environment: dict[str, str]
    mcp_servers: list[dict[str, Any]]
    retrieval: dict[str, Any]
    validators: list[str]
    resources: dict[str, float]
    network: dict[str, str]
