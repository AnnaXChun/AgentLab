"""Decision inputs and observation-reactive clients, with no Docker/host exec."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from examples.reactive_experiment import (
    CORRECTED_ARGV,
    INITIAL_ARGV,
    INSTRUCTIONS,
    drive_one_decision,
    next_action,
    run_demo,
)
from experiment_infra.agents.external import ExternalAgentBackend
from experiment_infra.core.schema import Action, DecisionContext, EnvironmentSpec
from experiment_infra.runtime.executor import ExperimentRuntime
from packages.core.models.schemas import EventType as T
from packages.sandbox.base import ExecutionResult
from packages.sdk.api import create_app


class MemoryEnvironment:
    """A deterministic protocol fake, never an interpreter or subprocess runner."""

    def __init__(self):
        self.workspaces = {}
        self.specs = {}
        self.commands = []
        self.serial = 0

    async def create(self, spec):
        self.serial += 1
        handle = str(self.serial)
        self.workspaces[handle] = {}
        self.specs[handle] = spec
        return handle

    async def execute(self, handle, command):
        assert handle in self.workspaces
        self.commands.append(command.model_dump(mode="json"))
        if command.argv == INITIAL_ARGV:
            return ExecutionResult(
                exit_code=1, stdout="", stderr="NameError: name 'total' is not defined\n"
            )
        if command.argv == CORRECTED_ARGV:
            return ExecutionResult(exit_code=0, stdout="6\n", stderr="")
        raise ValueError("The memory fake only supports the two demo commands")

    async def upload(self, handle, path, data):
        self.workspaces[handle][path] = bytes(data)

    async def download(self, handle, path):
        return self.workspaces[handle][path]

    async def observe(self, handle):
        return {
            "spec": self.specs[handle].model_dump(mode="json"),
            "fingerprint": "decision-contract-memory-v1",
            "snapshot_mime_type": "application/json",
        }

    async def snapshot(self, handle):
        return json.dumps(
            {path: data.hex() for path, data in self.workspaces[handle].items()}, sort_keys=True
        ).encode()

    async def restore(self, handle, snapshot):
        self.workspaces[handle] = {
            path: bytes.fromhex(data) for path, data in json.loads(snapshot).items()
        }

    def diff(self, first, second):
        return {"changed": first != second}

    async def destroy(self, handle):
        del self.workspaces[handle]
        del self.specs[handle]


MEMORY_SPEC = EnvironmentSpec(backend="decision-contract-memory", image=None)


@pytest.fixture
async def feedback_runtime(store, artifacts):
    # Reuse the SQLite StaticPool and local CAS fixtures from tests/conftest.py.
    environment = MemoryEnvironment()
    runtime = ExperimentRuntime(
        SimpleNamespace(store=store, artifacts=artifacts, sandbox=None), environment=environment
    )
    yield runtime
    for handle in list(environment.workspaces):
        await environment.destroy(handle)


async def start(runtime):
    experiment = await runtime.create_experiment(
        "Print 6 after correcting a missing variable",
        environment_spec=MEMORY_SPEC,
        success_criteria={"test": {"argv": INITIAL_ARGV}},
    )
    hypothesis = await runtime.register_hypothesis(experiment.experiment_id, "Command prints 6")
    run = await runtime.start_run(experiment.experiment_id)
    return experiment, hypothesis, run


def action(argv=INITIAL_ARGV, *, context_id=None):
    return Action(
        tool="environment",
        operation="exec",
        arguments={"command": {"argv": list(argv)}},
        decision_context_id=context_id,
    )


async def test_agent_reacts_to_observation_and_repeats_identical_initial_choice():
    initial = {
        "instructions": INSTRUCTIONS,
        "feedback": {
            "state": {"goal": "print 6", "config": {}, "last_step_id": None},
            "observations": [],
            "verifier_feedback": [],
        },
    }
    unchanged = deepcopy(initial)
    first = await next_action(initial)
    failure = {
        "instructions": INSTRUCTIONS,
        "feedback": {
            "observations": [{"exit_code": 1, "stderr": "NameError: name 'total' is not defined"}]
        },
    }
    corrected = await next_action(failure)
    assert first.arguments["command"]["argv"] == INITIAL_ARGV
    assert corrected.arguments["command"]["argv"] == CORRECTED_ARGV
    assert corrected != first
    # Calling after the correction still gives the same initial choice. There is
    # no prewritten action queue or hidden progression in the external client.
    assert await next_action(initial) == first
    assert initial == unchanged
    for exit_code, stderr in [(0, failure["feedback"]["observations"][0]["stderr"]), (1, "other")]:
        altered = deepcopy(failure)
        altered["feedback"]["observations"][0].update(exit_code=exit_code, stderr=stderr)
        with pytest.raises(ValueError, match="No correction"):
            await next_action(altered)


async def test_reactive_demo_records_feedback_selected_actions(feedback_runtime, tmp_path):
    runtime = feedback_runtime
    result = await run_demo(runtime, environment_spec=MEMORY_SPEC)
    assert result["recovered"]
    assert result["actions"] == [INITIAL_ARGV, CORRECTED_ARGV]
    assert result["exit_codes"] == [1, 0]
    assert result["stdout"] == "6\n"
    assert result["run_status"] == "FAILED"  # Preserve the original failed action.
    assert not runtime.environment.workspaces
    trajectory = await runtime.get_trajectory(result["trajectory_id"])
    first, second = trajectory.steps
    assert first.decision_context.payload["feedback"]["observations"] == []
    assert second.decision_context.payload["feedback"]["observations"] == [
        first.observation.model_dump(mode="json")
    ]
    assert second.decision_context.source_refs == (first.observation.event_id,)
    assert first.decision_context.last_step_id is None
    assert second.decision_context.last_step_id == first.step_id
    for step in trajectory.steps:
        assert step.action.decision_context_id == step.decision_context.context_id
        assert step.decision_context.coverage == "complete"
    graph = await runtime.get_provenance(second.action.event_id)
    assert {
        "source": second.decision_context.context_id,
        "target": second.action.event_id,
        "relation": "decision_input",
    } in graph["edges"]
    exported = await runtime.export(result["experiment_id"], tmp_path / "reactive-export")
    rows = [
        json.loads(line)
        for line in (Path(exported["directory"]) / "sft.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert [row["input"] for row in rows] == [
        first.decision_context.payload,
        second.decision_context.payload,
    ]
    assert exported["manifest"]["context_coverage"] == {"complete": 2, "partial": 0, "unknown": 0}


async def test_driver_records_exact_payload_before_agent_call_without_aliases(feedback_runtime):
    runtime = feedback_runtime
    experiment, _, run = await start(runtime)
    delivered = []

    async def inspecting_agent(payload):
        contexts = [
            obj
            for obj in (await runtime.ledger.objects(experiment.experiment_id)).values()
            if isinstance(obj, DecisionContext)
        ]
        assert len(contexts) == 1  # It was persisted before invoking this callback.
        assert payload == contexts[0].payload
        delivered.append(deepcopy(payload))
        chosen = await next_action(payload)
        payload["feedback"]["observations"].append({"stderr": "agent-local mutation"})
        payload["instructions"] = "mutated after delivery"
        return chosen

    step = await drive_one_decision(
        runtime, experiment.experiment_id, run.run_id, ExternalAgentBackend(inspecting_agent)
    )
    assert step.decision_context.payload == delivered[0]
    assert step.decision_context.payload["feedback"]["observations"] == []
    assert step.action.arguments["command"]["argv"] == INITIAL_ARGV


async def test_payload_snapshot_hash_and_frozen_ledger_identity(feedback_runtime):
    runtime = feedback_runtime
    experiment, _, run = await start(runtime)
    payload = {"messages": [{"role": "user", "content": "μ input"}], "options": [1, {"a": True}]}
    expected = deepcopy(payload)
    context = await runtime.capture_decision_context(experiment.experiment_id, run.run_id, payload)
    assert context.coverage == "unknown"
    canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert context.payload_json == canonical
    assert context.payload_sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    payload["messages"][0]["content"] = "caller mutation"
    decoded = context.payload
    decoded["options"][1]["a"] = False
    assert context.payload == expected
    assert context.payload is not context.payload
    with pytest.raises(ValidationError, match="frozen"):
        context.payload_json = "{}"
    before = [
        event.model_dump_json() for event in runtime.storage.store.events(experiment.experiment_id)
    ]
    with pytest.raises(ValueError, match="Immutable"):
        await runtime.ledger.record(
            experiment.experiment_id,
            T.OBJECT_RECORDED,
            [context.model_copy(update={"coverage": "complete"})],
        )
    assert before == [
        event.model_dump_json() for event in runtime.storage.store.events(experiment.experiment_id)
    ]
    stored = await runtime.ledger.get(context.context_id, "DecisionContext")
    assert stored == context
    assert stored.payload == expected


@pytest.mark.parametrize("coverage", ["complete", "partial", "unknown"])
async def test_coverage_is_explicit_caller_attestation(feedback_runtime, coverage):
    experiment, _, run = await start(feedback_runtime)
    context = await feedback_runtime.capture_decision_context(
        experiment.experiment_id, run.run_id, {"input": "caller supplied"}, coverage=coverage
    )
    assert context.coverage == coverage
    assert DecisionContext.model_validate_json(context.model_dump_json()) == context


async def test_payload_tampering_and_invalid_capture_rejected(feedback_runtime):
    runtime = feedback_runtime
    experiment, _, run = await start(runtime)
    context = await runtime.capture_decision_context(experiment.experiment_id, run.run_id, {"a": 1})
    with pytest.raises(ValidationError, match="hash mismatch"):
        DecisionContext.model_validate({**context.model_dump(), "payload_json": '{"a":2}'})
    noncanonical = '{"a": 1}'
    with pytest.raises(ValidationError, match="canonical JSON"):
        DecisionContext.model_validate(
            {
                **context.model_dump(),
                "payload_json": noncanonical,
                "payload_sha256": hashlib.sha256(noncanonical.encode()).hexdigest(),
            }
        )
    with pytest.raises(ValidationError):
        await runtime.capture_decision_context(
            experiment.experiment_id, run.run_id, {}, coverage="model-proven"
        )
    with pytest.raises(ValueError):
        await runtime.capture_decision_context(
            experiment.experiment_id, run.run_id, {"x": float("nan")}
        )


async def test_stale_and_cross_run_contexts_rejected_before_execution(feedback_runtime):
    runtime = feedback_runtime
    experiment, _, run = await start(runtime)
    context = await runtime.capture_decision_context(experiment.experiment_id, run.run_id, {})
    sibling = await runtime.start_run(experiment.experiment_id)
    before = len(runtime.environment.commands)
    with pytest.raises(ValueError, match="another run"):
        await runtime.execute(
            experiment.experiment_id, action(context_id=context.context_id), run_id=sibling.run_id
        )
    assert len(runtime.environment.commands) == before
    first = await runtime.execute(
        experiment.experiment_id, action(context_id=context.context_id), run_id=run.run_id
    )
    with pytest.raises(ValueError, match="stale"):
        await runtime.execute(
            experiment.experiment_id,
            action(CORRECTED_ARGV, context_id=context.context_id),
            run_id=run.run_id,
        )
    assert len(runtime.environment.commands) == before + 1
    assert len((await runtime.get_trajectory(run.trajectory_id)).steps) == 1
    for foreign_ref in (first.observation.event_id, first.step_id, first.action.event_id):
        with pytest.raises(ValueError, match="another run"):
            await runtime.capture_decision_context(
                experiment.experiment_id, sibling.run_id, {}, source_refs=[foreign_ref]
            )
    other = await runtime.create_experiment("other", environment_spec=MEMORY_SPEC)
    other_run = await runtime.start_run(other.experiment_id)
    with pytest.raises(ValueError, match="Cross-experiment"):
        await runtime.execute(
            other.experiment_id, action(context_id=context.context_id), run_id=other_run.run_id
        )


async def test_feedback_contains_only_produced_results_and_never_mutates_prior_input(
    feedback_runtime,
):
    runtime = feedback_runtime
    experiment, hypothesis, run = await start(runtime)
    agent = ExternalAgentBackend(next_action)
    first = await drive_one_decision(runtime, experiment.experiment_id, run.run_id, agent)
    second = await drive_one_decision(runtime, experiment.experiment_id, run.run_id, agent)
    before_verification = second.decision_context.model_dump_json()
    feedback = await runtime.get_feedback(experiment.experiment_id, run.run_id)
    assert [item["event_id"] for item in feedback["observations"]] == [
        first.observation.event_id,
        second.observation.event_id,
    ]
    assert feedback["verifier_feedback"] == []
    evidence = await runtime.verify(
        experiment.experiment_id,
        run.run_id,
        hypothesis.hypothesis_id,
        "test",
        [first.observation.structured_output["log_artifact_id"]],
    )
    feedback = await runtime.get_feedback(experiment.experiment_id, run.run_id)
    assert [item["verification_id"] for item in feedback["verifier_feedback"]] == [
        evidence.verification_id
    ]
    assert feedback["verifier_feedback"][0]["status"] == "FAILED"
    # Verification may annotate the trajectory later, but the recorded input to
    # its earlier decision is immutable and never backfilled with that result.
    trajectory = await runtime.get_trajectory(run.trajectory_id)
    assert trajectory.steps[0].verifier_feedback
    assert trajectory.steps[1].decision_context.model_dump_json() == before_verification
    assert first.decision_context.payload["feedback"]["observations"] == []
    assert second.decision_context.payload["feedback"]["verifier_feedback"] == []
    assert len(second.decision_context.payload["feedback"]["observations"]) == 1
    sibling = await runtime.start_run(experiment.experiment_id)
    sibling_feedback = await runtime.get_feedback(experiment.experiment_id, sibling.run_id)
    assert sibling_feedback["observations"] == sibling_feedback["verifier_feedback"] == []
    claim = await runtime.create_claim(
        experiment.experiment_id, "Initial command printed 6", [evidence.evidence_id]
    )
    for foreign_ref in (evidence.evidence_id, evidence.verification_id, claim.claim_id):
        with pytest.raises(ValueError, match="another run"):
            await runtime.capture_decision_context(
                experiment.experiment_id, sibling.run_id, {}, source_refs=[foreign_ref]
            )


async def test_uncaptured_action_is_explicitly_missing_context(feedback_runtime):
    runtime = feedback_runtime
    experiment, _, run = await start(runtime)
    step = await runtime.execute(experiment.experiment_id, action(), run_id=run.run_id)
    assert step.decision_context is None
    assert step.action.decision_context_id is None
    await runtime.finish_run(run.run_id)
    with pytest.raises(ValueError, match="not executable"):
        await runtime.capture_decision_context(experiment.experiment_id, run.run_id, {})


async def test_persisted_context_survives_reload_and_detects_event_tampering(
    feedback_runtime, monkeypatch
):
    runtime = feedback_runtime
    experiment, _, run = await start(runtime)
    context = await runtime.capture_decision_context(
        experiment.experiment_id, run.run_id, {"prompt": "original"}
    )
    reloaded = ExperimentRuntime(runtime.storage, environment=runtime.environment)
    assert await reloaded.ledger.get(context.context_id, "DecisionContext") == context
    events = runtime.storage.store.events(experiment.experiment_id)
    for index, event in enumerate(events):
        output = deepcopy(event.output)
        for item in output.get("objects", []):
            if item["id"] == context.context_id:
                item["data"]["payload_json"] = '{"prompt":"tampered"}'
                events[index] = event.model_copy(update={"output": output})
    monkeypatch.setattr(runtime.storage.store, "events", lambda experiment_id: events)
    with pytest.raises(ValueError, match="integrity"):
        await reloaded.ledger.get(context.context_id, "DecisionContext")


def test_http_feedback_capture_binding_and_spoof_rejection(feedback_runtime):
    runtime = feedback_runtime
    with TestClient(create_app(runtime.storage, experiment_runtime=runtime)) as client:
        created = client.post(
            "/experiments",
            json={"goal": "HTTP feedback", "environment_spec": MEMORY_SPEC.model_dump(mode="json")},
        )
        assert created.status_code == 200, created.text
        eid = created.json()["experiment_id"]
        base = f"/experiments/{eid}"
        started = client.post(base + "/runs", json={})
        assert started.status_code == 200, started.text
        rid = started.json()["run_id"]
        feedback = client.get(base + "/feedback", params={"run_id": rid})
        assert feedback.status_code == 200, feedback.text
        payload = {"instructions": INSTRUCTIONS, "feedback": feedback.json()["data"]}
        assert payload["feedback"]["observations"] == []
        assert payload["feedback"]["verifier_feedback"] == []
        request = {"run_id": rid, "payload": payload, "coverage": "complete"}
        # Coverage is the caller's report. No request field can promote it to
        # verified consumption, host authority, or an attacker-selected digest.
        for field, value in [
            ("coverage_verified", True),
            ("model_consumed", True),
            ("trust_level", "host_evaluated"),
            ("payload_sha256", "0" * 64),
        ]:
            rejected = client.post(base + "/decision-contexts", json={**request, field: value})
            assert rejected.status_code == 422, rejected.text
        rejected = client.post(
            base + "/decision-contexts", json={**request, "coverage": "model-proven"}
        )
        assert rejected.status_code == 422
        captured = client.post(base + "/decision-contexts", json=request)
        assert captured.status_code == 200, captured.text
        context = DecisionContext.model_validate(captured.json())
        assert context.payload == payload
        assert context.coverage == "complete"
        bound = action(context_id=context.context_id).model_dump(mode="json")
        rejected = client.post(
            base + "/actions",
            json={
                "run_id": rid,
                "action": {**bound, "decision_context": context.model_dump(mode="json")},
            },
        )
        assert rejected.status_code == 422
        executed = client.post(base + "/actions", json={"run_id": rid, "action": bound})
        assert executed.status_code == 200, executed.text
        step = executed.json()
        assert step["action"]["decision_context_id"] == context.context_id
        assert step["decision_context"] == context.model_dump(mode="json")
        assert step["observation"]["exit_code"] == 1
        feedback = client.get(base + "/feedback", params={"run_id": rid}).json()["data"]
        assert [item["event_id"] for item in feedback["observations"]] == [
            step["observation"]["event_id"]
        ]
        assert feedback["verifier_feedback"] == []
        stale = client.post(base + "/actions", json={"run_id": rid, "action": bound})
        assert stale.status_code == 400 and "stale" in stale.json()["detail"]
        assert len(runtime.environment.commands) == 1
        finished = client.post(f"/runs/{rid}/finish")
        assert finished.status_code == 200, finished.text
