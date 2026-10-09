"""Structured evidence binding and an explicit host-owned evaluation boundary.

Policy documents are declarations, not authority to trust a measurement. Only a host
registering a HostEvaluator can produce host_evaluated results; no action payload,
artifact metadata, or success criterion can promote submitted metrics to that level.
"""

import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from experiment_infra.core.schema import Artifact, Experiment, Hypothesis, Run

EvidenceTrust = Literal["agent_reported", "runtime_observed", "host_evaluated"]
_TRUST = {"agent_reported": 0, "runtime_observed": 1, "host_evaluated": 2}


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class VerifierPolicy(PolicyModel):
    policy_id: str = Field(min_length=1)
    verifier: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    trust_requirement: EvidenceTrust = "host_evaluated"

    @model_validator(mode="after")
    def validate_builtin_parameters(self):
        values = self.parameters
        if self.verifier == "metric":
            if not isinstance(values.get("field"), str) or not values["field"]:
                raise ValueError("Metric policy requires a field")
            if values.get("operator") not in {"lt", "gt"}:
                raise ValueError("Metric policy requires operator lt or gt")
            if ("baseline" in values) == ("baseline_field" in values):
                raise ValueError("Metric policy requires exactly one baseline or baseline_field")
            if "baseline_field" in values:
                if not isinstance(values["baseline_field"], str) or not values["baseline_field"]:
                    raise ValueError("Metric policy baseline_field must be a nonempty string")
            else:
                baseline = values["baseline"]
                if (
                    isinstance(baseline, bool)
                    or not isinstance(baseline, (int, float))
                    or not math.isfinite(baseline)
                ):
                    raise ValueError("Metric policy baseline must be a finite number")
        if self.verifier == "test":
            argv = values.get("argv")
            if (
                not isinstance(argv, list)
                or not argv
                or any(not isinstance(arg, str) or not arg for arg in argv)
            ):
                raise ValueError("Test policy requires a nonempty argv list")
        # Fail early for non-JSON values, including NaN nested in arbitrary parameters.
        json.dumps(values, allow_nan=False)
        return self


class SuccessCriterion(PolicyModel):
    criterion_id: str = Field(min_length=1)
    description: str = ""
    policy: VerifierPolicy


class HostEvaluation(PolicyModel):
    """Result returned by host code after independent evaluation of supplied artifacts."""

    status: Literal["PASSED", "FAILED", "INCONCLUSIVE", "ERROR"]
    metric: str = Field(min_length=1)
    value: Any = None
    uncertainty: float | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationContext:
    experiment: "Experiment"
    hypothesis: "Hypothesis"
    run: "Run"
    criterion: SuccessCriterion
    artifacts: Mapping[str, tuple["Artifact", bytes]]


@dataclass(frozen=True)
class HostEvaluator:
    """Host-injected evaluator, never constructed from public request data.

    Register as runtime.verifiers[name] = HostEvaluator(callback). The callback must
    independently evaluate outputs against host-owned fixtures/logic. Executing an
    agent's test file or re-reading its metrics is not an independent evaluation.
    This adapter is a trust declaration by the host, not a sandbox or attestation
    system; malicious host code or a compromised evaluator can still lie.
    """

    callback: Callable[[EvaluationContext], Awaitable[HostEvaluation]]

    async def evaluate(self, context: EvaluationContext) -> HostEvaluation:
        return HostEvaluation.model_validate(await self.callback(context))


def policy_digest(criterion: SuccessCriterion) -> str:
    canonical = json.dumps(
        criterion.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def criteria_by_id(experiment) -> dict[str, SuccessCriterion]:
    criteria = [SuccessCriterion.model_validate(item) for item in experiment.criteria]
    result = {item.criterion_id: item for item in criteria}
    if len(result) != len(criteria):
        raise ValueError("Experiment criterion IDs must be unique")
    policy_ids = [item.policy.policy_id for item in criteria]
    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("Experiment verifier policy IDs must be unique")
    return result


def resolve_criterion(experiment, hypothesis, verifier, criterion_id=None):
    available = criteria_by_id(experiment)
    if not available:
        if criterion_id is not None or hypothesis.criterion_ids:
            raise ValueError("No structured criteria are declared for this experiment")
        return None
    declared = hypothesis.criterion_ids
    if not declared or len(set(declared)) != len(declared):
        raise ValueError("Structured hypotheses require unique declared criterion IDs")
    if not set(declared).issubset(available):
        raise ValueError("Hypothesis references an unknown criterion")
    if criterion_id is None:
        matching = [key for key in declared if available[key].policy.verifier == verifier]
        if len(matching) != 1:
            raise ValueError("Select a criterion_id declared by this hypothesis")
        criterion_id = matching[0]
    if criterion_id not in declared:
        raise ValueError("Criterion is not declared by the selected hypothesis")
    criterion = available[criterion_id]
    if criterion.policy.verifier != verifier:
        raise ValueError("Verifier does not match the criterion policy")
    return criterion


def meets_trust(actual: EvidenceTrust, required: EvidenceTrust) -> bool:
    return _TRUST[actual] >= _TRUST[required]


def verifier_trust(verifier) -> EvidenceTrust:
    # Do not read a user-settable string or duck-typed trust attribute. These
    # concrete adapters define the three audited boundaries.
    from experiment_infra.verifiers.test import TestVerifier

    if isinstance(verifier, HostEvaluator):
        return "host_evaluated"
    if type(verifier) is TestVerifier:
        return "runtime_observed"
    return "agent_reported"
