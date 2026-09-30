from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from experiment_infra.core import schema as m
from experiment_infra.runtime.executor import ExperimentRuntime
from packages.core.models.schemas import EventType as T


def samples():
    experiment = m.Experiment(goal="test")
    eid = experiment.experiment_id
    hypothesis = m.Hypothesis(experiment_id=eid, statement="A testable hypothesis")
    run = m.Run(
        experiment_id=eid, environment_spec=m.EnvironmentSpec(), code_version="sha256:source"
    )
    action = m.ActionEvent(
        experiment_id=eid,
        run_id=run.run_id,
        step_id="step",
        tool="environment",
        operation="exec",
        arguments={},
        expected_effect="result",
        agent_backend="external",
    )
    observation = m.ObservationEvent(action_event_id=action.event_id)
    artifact = m.Artifact(
        type="metric",
        uri="cas://example",
        sha256="a" * 64,
        size=1,
        mime_type="application/json",
        producer_run_id=run.run_id,
    )
    verification = m.VerificationResult(
        experiment_id=eid,
        run_id=run.run_id,
        verifier="metric",
        status="PASSED",
        source_artifact_ids=[artifact.artifact_id],
        metric="x",
        value=2,
    )
    evidence = m.Evidence(
        experiment_id=eid,
        hypothesis_id=hypothesis.hypothesis_id,
        source_artifact_ids=[artifact.artifact_id],
        source_run_ids=[run.run_id],
        metric="x",
        value=2,
        verifier="metric",
        verification_id=verification.verification_id,
        verification_status="PASSED",
    )
    step = m.TrajectoryStep(
        trajectory_id=run.trajectory_id,
        state_before={},
        action=action,
        observation=observation,
        state_after={},
        latency=1,
        branch_id=run.run_id,
    )
    return [
        experiment,
        hypothesis,
        run,
        action,
        observation,
        artifact,
        verification,
        evidence,
        m.Claim(experiment_id=eid, statement="unverified"),
        step,
        m.Trajectory(
            trajectory_id=run.trajectory_id,
            experiment_id=eid,
            run_id=run.run_id,
            agent_backend="external",
            steps=[step],
            final_status="RUNNING",
        ),
        m.Snapshot(
            experiment_id=eid, run_id=run.run_id, artifact_id=artifact.artifact_id, environment={}
        ),
        m.CounterfactualLink(
            original_step_id="a", counterfactual_step_id="b", intervention_type="state_restore"
        ),
        m.DomainPack(name="any-domain"),
        m.Action(tool="rag", operation="retrieve"),
        m.EnvironmentSpec(),
    ]


@pytest.mark.parametrize("model", samples(), ids=lambda obj: type(obj).__name__)
def test_all_public_models_roundtrip(model):
    assert type(model).model_validate_json(model.model_dump_json()) == model
    assert model.schema_version == "2.0"
    with pytest.raises(ValidationError):
        type(model).model_validate({**model.model_dump(), "schema_version": "1.0"})


def test_claim_requires_evidence_and_counterfactual_enum():
    with pytest.raises(ValidationError):
        m.Claim(experiment_id="e", statement="agent says yes", status="SUPPORTED")
    with pytest.raises(ValidationError):
        m.Claim(experiment_id="e", statement="old status", status="VERIFIED", evidence_ids=["x"])
    with pytest.raises(ValidationError):
        m.CounterfactualLink(
            original_step_id="a", counterfactual_step_id="b", intervention_type="train_rl"
        )


async def test_provenance_atomicity_and_claim_scope(store, artifacts):
    rt = ExperimentRuntime(
        SimpleNamespace(store=store, artifacts=artifacts, sandbox=None), environment=object()
    )
    exp = await rt.create_experiment("test")
    hypothesis = await rt.register_hypothesis(exp.experiment_id, "question")
    prior_count = len(store.events(exp.experiment_id))
    with pytest.raises(ValueError, match="acyclic"):
        await rt.ledger.record(
            exp.experiment_id,
            T.OBJECT_RECORDED,
            links=[(hypothesis.hypothesis_id, exp.experiment_id, "cycle")],
        )
    assert len(store.events(exp.experiment_id)) == prior_count
    claim = await rt.create_claim(exp.experiment_id, "unsupported assertion")
    assert claim.status == "UNVERIFIED"
    with pytest.raises(KeyError):
        await rt.create_claim(exp.experiment_id, "fake evidence", ["nonexistent"])
    assert hypothesis.hypothesis_id in await rt.get_descendants(exp.experiment_id)
    assert exp.experiment_id in await rt.get_ancestors(hypothesis.hypothesis_id)


def test_custom_environment_does_not_initialize_docker(monkeypatch):
    import packages.sdk.runtime as bootstrap

    def forbidden():
        raise AssertionError("Docker must not be required by a custom environment")

    monkeypatch.setattr(bootstrap, "DockerSandboxBackend", forbidden)
    env = object()
    rt = ExperimentRuntime(environment=env)
    assert rt.environment is env
    assert rt.storage.sandbox is None
    rt.storage.engine.dispose()


async def test_alternate_environment_can_execute_and_replay_without_docker(store, artifacts):
    import json

    from packages.sandbox.base import ExecutionResult

    class MemoryEnvironment:
        """Test double proving the runtime does not interpret Docker metadata or tar."""

        def __init__(self):
            self.workspaces = {}
            self.specs = {}
            self.serial = 0

        async def create(self, spec):
            self.serial += 1
            handle = str(self.serial)
            self.workspaces[handle] = {}
            self.specs[handle] = spec
            return handle

        async def execute(self, handle, action):
            return ExecutionResult(exit_code=0, stdout="memory environment", stderr="")

        async def upload(self, handle, path, data):
            self.workspaces[handle][path] = data.decode()

        async def download(self, handle, path):
            return self.workspaces[handle][path].encode()

        async def observe(self, handle):
            return {
                "spec": self.specs[handle].model_dump(mode="json"),
                "fingerprint": "memory-v1",
                "snapshot_mime_type": "application/json",
            }

        async def snapshot(self, handle):
            return json.dumps(self.workspaces[handle], sort_keys=True).encode()

        async def restore(self, handle, snapshot):
            self.workspaces[handle] = json.loads(snapshot)

        def diff(self, first, second):
            return {"changed": first != second}

        async def destroy(self, handle):
            del self.workspaces[handle]

    environment = MemoryEnvironment()
    rt = ExperimentRuntime(
        SimpleNamespace(store=store, artifacts=artifacts, sandbox=None), environment=environment
    )
    exp = await rt.create_experiment(
        "backend contract", environment_spec=m.EnvironmentSpec(backend="memory-test", image=None)
    )
    run = await rt.start_run(exp.experiment_id)
    step = await rt.execute(
        exp.experiment_id,
        m.Action(tool="environment", operation="exec", arguments={"command": {"argv": ["noop"]}}),
        run_id=run.run_id,
    )
    assert step.observation.stdout == "memory environment"
    await rt.finish_run(run.run_id)
    assert (await rt.replay(run.run_id)).matched
    assert not environment.workspaces
