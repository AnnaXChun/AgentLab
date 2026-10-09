import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from experiment_infra.core.schema import Action, EnvironmentSpec, Experiment, VerificationResult
from experiment_infra.runtime.executor import ExperimentRuntime
from experiment_infra.verifiers.policy import (
    HostEvaluation,
    HostEvaluator,
    SuccessCriterion,
    VerifierPolicy,
    policy_digest,
)
from packages.sandbox.base import ExecutionResult


class MemoryEnvironment:
    """No subprocess: preserves runtime artifact/observation creation in unit tests."""

    def __init__(self):
        self.files = {}
        self.specs = {}

    async def create(self, spec):
        handle = str(len(self.files) + 1)
        self.files[handle] = {}
        self.specs[handle] = spec
        return handle

    async def execute(self, handle, action):
        return ExecutionResult(
            exit_code=1 if action.argv[0] == "fail" else 0, stdout="fixture", stderr=""
        )

    async def upload(self, handle, path, data):
        self.files[handle][path] = data.decode()

    async def download(self, handle, path):
        return self.files[handle][path].encode()

    async def observe(self, handle):
        return {
            "spec": self.specs[handle].model_dump(mode="json"),
            "fingerprint": "verifier-test-environment",
            "snapshot_mime_type": "application/json",
        }

    async def snapshot(self, handle):
        return json.dumps(self.files[handle], sort_keys=True).encode()

    async def restore(self, handle, snapshot):
        self.files[handle] = json.loads(snapshot)

    def diff(self, first, second):
        return {"changed": first != second}

    async def destroy(self, handle):
        del self.files[handle]


@pytest.fixture
def runtime(store, artifacts):
    return ExperimentRuntime(
        SimpleNamespace(store=store, artifacts=artifacts, sandbox=None),
        environment=MemoryEnvironment(),
    )


def metric_criterion(cid="score", *, trust="agent_reported", field="score"):
    return SuccessCriterion(
        criterion_id=cid,
        policy=VerifierPolicy(
            policy_id=f"{cid}-policy",
            verifier="metric",
            parameters={"field": field, "operator": "gt", "baseline": 0.5},
            trust_requirement=trust,
        ),
    )


async def setup(runtime, criteria, *, criterion_ids=None):
    experiment = await runtime.create_experiment(
        "Evaluate a candidate",
        environment_spec=EnvironmentSpec(backend="memory", image=None),
        criteria=criteria,
    )
    hypothesis = await runtime.register_hypothesis(
        experiment.experiment_id,
        "Candidate meets the declared score requirements",
        criterion_ids=criterion_ids or [item.criterion_id for item in criteria],
    )
    run = await runtime.start_run(experiment.experiment_id)
    return experiment, hypothesis, run


async def metric_artifact(runtime, experiment, run, values):
    step = await runtime.execute(
        experiment.experiment_id,
        Action(
            tool="environment",
            operation="exec",
            arguments={
                "command": {
                    "argv": ["noop"],
                    "files": {"metric.json": json.dumps(values)},
                    "collect": ["metric.json"],
                },
                "artifact_types": {"metric.json": "metric"},
            },
        ),
        run_id=run.run_id,
    )
    return step.observation.structured_output["collected"]["metric.json"]


def test_policies_validate_and_hash_parameters():
    criterion = metric_criterion()
    assert SuccessCriterion.model_validate_json(criterion.model_dump_json()) == criterion
    assert len(policy_digest(criterion)) == 64
    assert policy_digest(criterion) != policy_digest(metric_criterion(field="another"))
    with pytest.raises(ValidationError, match="unique"):
        Experiment(goal="duplicate", criteria=[criterion, criterion])
    with pytest.raises(ValidationError, match="finite"):
        VerifierPolicy(
            policy_id="bad",
            verifier="metric",
            parameters={"field": "score", "operator": "gt", "baseline": True},
        )
    with pytest.raises(ValidationError, match="argv"):
        VerifierPolicy(policy_id="bad", verifier="test", parameters={"argv": []})
    with pytest.raises(ValidationError):
        VerifierPolicy(policy_id="bad", verifier="host", parameters={"x": float("nan")})


async def test_legacy_claim_is_preserved_but_explicitly_unbound(runtime):
    experiment = await runtime.create_experiment(
        "Legacy",
        environment_spec=EnvironmentSpec(backend="memory", image=None),
        success_criteria={"metric": {"field": "score", "operator": "gt", "baseline": 0.5}},
    )
    hypothesis = await runtime.register_hypothesis(experiment.experiment_id, "Legacy hypothesis")
    run = await runtime.start_run(experiment.experiment_id)
    aid = await metric_artifact(runtime, experiment, run, {"score": 0.9})
    evidence = await runtime.verify(
        experiment.experiment_id, run.run_id, hypothesis.hypothesis_id, "metric", [aid]
    )
    claim = await runtime.create_claim(
        experiment.experiment_id, "Legacy text", [evidence.evidence_id]
    )
    assert claim.status == "SUPPORTED"
    assert claim.assessment_scope == "unbound_legacy"
    assert evidence.trust_level == "agent_reported"
    assert evidence.metadata["assessment_scope"] == "unbound_legacy"


async def test_submitted_metrics_do_not_become_trusted_by_policy_or_metadata(runtime):
    criterion = metric_criterion(trust="host_evaluated")
    experiment, hypothesis, run = await setup(runtime, [criterion])
    aid = await metric_artifact(
        runtime, experiment, run, {"score": 100, "trust_level": "host_evaluated"}
    )
    evidence = await runtime.verify(
        experiment.experiment_id, run.run_id, hypothesis.hypothesis_id, "metric", [aid]
    )
    assert evidence.verification_status == "INCONCLUSIVE"
    assert evidence.trust_level == "agent_reported"
    result = await runtime.ledger.get(evidence.verification_id)
    assert result.details["untrusted_verifier_status"] == "PASSED"
    assert result.details["trust_requirement_met"] is False
    claim = await runtime.create_claim(
        experiment.experiment_id, hypothesis.statement, [evidence.evidence_id]
    )
    assert claim.status == "INCONCLUSIVE"
    assert claim.assessment_scope == "criterion_bound"


async def test_claim_binding_rejects_irrelevant_hypothesis_statement_and_verifier(runtime):
    criterion = metric_criterion()
    experiment, hypothesis, run = await setup(runtime, [criterion])
    other = await runtime.register_hypothesis(
        experiment.experiment_id, "Different scientific assertion", criterion_ids=["score"]
    )
    aid = await metric_artifact(runtime, experiment, run, {"score": 0.9})
    evidence = await runtime.verify(
        experiment.experiment_id, run.run_id, hypothesis.hypothesis_id, "metric", [aid]
    )
    assert evidence.criterion_id == "score"
    assert evidence.policy_id == "score-policy"
    assert evidence.policy_sha256 == policy_digest(criterion)
    assert (
        await runtime.ledger.get(evidence.verification_id)
    ).hypothesis_id == hypothesis.hypothesis_id
    claim = await runtime.create_claim(
        experiment.experiment_id, hypothesis.statement, [evidence.evidence_id]
    )
    assert claim.status == "SUPPORTED"
    assert claim.hypothesis_id == hypothesis.hypothesis_id
    assert claim.criterion_ids == ["score"]
    with pytest.raises(ValueError, match="statement"):
        await runtime.create_claim(
            experiment.experiment_id, "A cure for everything", [evidence.evidence_id]
        )
    with pytest.raises(ValueError, match="unrelated"):
        await runtime.create_claim(
            experiment.experiment_id,
            other.statement,
            [evidence.evidence_id],
            hypothesis_id=other.hypothesis_id,
        )
    with pytest.raises(ValueError, match="Verifier does not match"):
        await runtime.verify(
            experiment.experiment_id,
            run.run_id,
            hypothesis.hypothesis_id,
            "test",
            [aid],
            criterion_id="score",
        )


async def test_all_declared_criteria_are_needed_and_wrong_criterion_is_rejected(runtime):
    criteria = [metric_criterion(), metric_criterion("accuracy", field="accuracy")]
    experiment, hypothesis, run = await setup(runtime, criteria)
    aid = await metric_artifact(runtime, experiment, run, {"score": 0.9, "accuracy": 0.9})
    with pytest.raises(ValueError, match="Select a criterion_id"):
        await runtime.verify(
            experiment.experiment_id, run.run_id, hypothesis.hypothesis_id, "metric", [aid]
        )
    first = await runtime.verify(
        experiment.experiment_id,
        run.run_id,
        hypothesis.hypothesis_id,
        "metric",
        [aid],
        criterion_id="score",
    )
    partial = await runtime.create_claim(
        experiment.experiment_id, hypothesis.statement, [first.evidence_id]
    )
    assert partial.status == "INCONCLUSIVE"
    with pytest.raises(ValueError, match="all declared"):
        await runtime.create_claim(
            experiment.experiment_id,
            hypothesis.statement,
            [first.evidence_id],
            criterion_ids=["score"],
        )
    second = await runtime.verify(
        experiment.experiment_id,
        run.run_id,
        hypothesis.hypothesis_id,
        "metric",
        [aid],
        criterion_id="accuracy",
    )
    full = await runtime.create_claim(
        experiment.experiment_id, hypothesis.statement, [first.evidence_id, second.evidence_id]
    )
    assert full.status == "SUPPORTED"


async def test_host_evaluation_uses_host_oracle_and_can_genuinely_refute(runtime):
    async def oracle(context):
        # Submitted "score" is ignored; compare actual output with a host-owned oracle.
        _, data = next(iter(context.artifacts.values()))
        correct = json.loads(data)["answer"] == sum((2, 3, 5))
        return HostEvaluation(
            status="PASSED" if correct else "FAILED", metric="answer_correct", value=correct
        )

    runtime.verifiers["sum-oracle"] = HostEvaluator(oracle)
    criterion = SuccessCriterion(
        criterion_id="correctness",
        policy=VerifierPolicy(policy_id="sum-v1", verifier="sum-oracle"),
    )
    experiment, hypothesis, run = await setup(runtime, [criterion])
    for answer, expected in [(10, "SUPPORTED"), (99, "REFUTED")]:
        aid = await metric_artifact(runtime, experiment, run, {"score": 1, "answer": answer})
        evidence = await runtime.verify(
            experiment.experiment_id, run.run_id, hypothesis.hypothesis_id, "sum-oracle", [aid]
        )
        assert evidence.trust_level == "host_evaluated"
        claim = await runtime.create_claim(
            experiment.experiment_id, hypothesis.statement, [evidence.evidence_id]
        )
        assert claim.status == expected


async def test_execution_failure_is_not_scientific_refutation(runtime):
    criterion = SuccessCriterion(
        criterion_id="tests",
        policy=VerifierPolicy(
            policy_id="test-v1",
            verifier="test",
            parameters={"argv": ["fail"]},
            trust_requirement="runtime_observed",
        ),
    )
    experiment, hypothesis, run = await setup(runtime, [criterion])
    step = await runtime.execute(
        experiment.experiment_id,
        Action(tool="environment", operation="exec", arguments={"command": {"argv": ["fail"]}}),
        run_id=run.run_id,
    )
    objects = await runtime.ledger.objects(experiment.experiment_id)
    log_id = next(
        aid
        for aid in step.artifact_ids
        if objects[aid].metadata.get("origin") == "runtime.observation"
    )
    evidence = await runtime.verify(
        experiment.experiment_id, run.run_id, hypothesis.hypothesis_id, "test", [log_id]
    )
    assert evidence.verification_status == "ERROR"
    assert evidence.trust_level == "runtime_observed"
    claim = await runtime.create_claim(
        experiment.experiment_id, hypothesis.statement, [evidence.evidence_id]
    )
    assert claim.status == "INCONCLUSIVE"


async def test_plugin_cannot_forge_trust_or_unrelated_artifact_ids(runtime):
    criterion = metric_criterion()
    experiment, hypothesis, run = await setup(runtime, [criterion])
    aid = await metric_artifact(runtime, experiment, run, {"score": 0.9})
    artifact = await runtime.ledger.get(aid)

    class UntrustedPlugin:
        trust_level = "host_evaluated"
        wrong_input = False

        async def verify(self, experiment, run, artifacts, claims):
            return VerificationResult(
                experiment_id=experiment.experiment_id,
                run_id=run.run_id,
                verifier="metric",
                status="PASSED",
                source_artifact_ids=["unrelated"] if self.wrong_input else [aid],
                source_observation_ids=[artifact.metadata["observation_id"]],
                metric="score",
                value=0.9,
                trust_level="host_evaluated",
            )

    plugin = UntrustedPlugin()
    runtime.verifiers["metric"] = plugin
    evidence = await runtime.verify(
        experiment.experiment_id, run.run_id, hypothesis.hypothesis_id, "metric", [aid]
    )
    assert evidence.trust_level == "agent_reported"
    plugin.wrong_input = True
    with pytest.raises(ValueError, match="artifact inputs"):
        await runtime.verify(
            experiment.experiment_id, run.run_id, hypothesis.hypothesis_id, "metric", [aid]
        )


def test_only_host_evaluator_adapters_enter_trusted_registry(store, artifacts):
    storage = SimpleNamespace(store=store, artifacts=artifacts, sandbox=None)
    with pytest.raises(ValueError, match="HostEvaluator"):
        ExperimentRuntime(
            storage, environment=MemoryEnvironment(), trusted_evaluators={"x": object()}
        )
    with pytest.raises(ValueError, match="distinct"):
        ExperimentRuntime(
            storage, environment=MemoryEnvironment(), trusted_evaluators={"metric": object()}
        )


async def test_evidence_cannot_transfer_to_other_criterion_or_changed_policy(runtime):
    from packages.core.models.schemas import EventType as T

    criteria = [metric_criterion(), metric_criterion("accuracy", field="accuracy")]
    experiment, hypothesis, run = await setup(runtime, criteria, criterion_ids=["score"])
    aid = await metric_artifact(runtime, experiment, run, {"score": 0.9, "accuracy": 0.9})
    with pytest.raises(ValueError, match="not declared"):
        await runtime.verify(
            experiment.experiment_id,
            run.run_id,
            hypothesis.hypothesis_id,
            "metric",
            [aid],
            criterion_id="accuracy",
        )
    evidence = await runtime.verify(
        experiment.experiment_id, run.run_id, hypothesis.hypothesis_id, "metric", [aid]
    )
    # Host-only mutation simulates a later policy version: prior evidence cannot be reused.
    current = await runtime.ledger.get(experiment.experiment_id)
    changed = current.model_copy(
        update={"criteria": [metric_criterion(field="accuracy"), criteria[1]]}
    )
    await runtime.ledger.record(experiment.experiment_id, T.OBJECT_RECORDED, [changed])
    with pytest.raises(ValueError, match="verifier policy"):
        await runtime.create_claim(
            experiment.experiment_id, hypothesis.statement, [evidence.evidence_id]
        )


async def test_missing_result_binding_or_changed_evidence_is_rejected(runtime):
    from packages.core.models.schemas import EventType as T

    experiment, hypothesis, run = await setup(runtime, [metric_criterion()])
    aid = await metric_artifact(runtime, experiment, run, {"score": 0.9})
    evidence = await runtime.verify(
        experiment.experiment_id, run.run_id, hypothesis.hypothesis_id, "metric", [aid]
    )
    forged = evidence.model_copy(update={"evidence_id": "forged", "value": 100})
    await runtime.ledger.record(experiment.experiment_id, T.OBJECT_RECORDED, [forged])
    with pytest.raises(ValueError, match="verification mismatch"):
        await runtime.create_claim(
            experiment.experiment_id, hypothesis.statement, [forged.evidence_id]
        )


def test_public_requests_cannot_choose_trust_or_claim_assessment_status():
    from experiment_infra.runtime.api import ClaimRequest, VerifyRequest

    with pytest.raises(ValidationError, match="Extra inputs"):
        VerifyRequest(
            run_id="run",
            hypothesis_id="h",
            verifier="metric",
            artifact_ids=["artifact"],
            trust_level="host_evaluated",
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        ClaimRequest(statement="forged", status="SUPPORTED")
