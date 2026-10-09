"""Benchmark decision capture uses exact HTTP input, without a live model or sandbox."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from experiment_infra.agents.openai_compatible import OpenAICompatibleAgentBackend
from experiment_infra.benchmarks.base import AdapterBase, BenchmarkTask
from experiment_infra.benchmarks.runner import BenchmarkRunner
from experiment_infra.core.schema import Action, DecisionContext, EnvironmentSpec
from experiment_infra.runtime.executor import ExperimentRuntime
from packages.sandbox.base import ExecutionResult


class MemoryEnvironment:
    def __init__(self):
        self.commands = []
        self.workspaces = {}
        self.spec = None
        self.files = {}

    async def create(self, spec):
        self.spec = spec
        self.workspaces["fixture"] = []
        return "fixture"

    async def execute(self, handle, command):
        self.commands.append(command)
        self.workspaces[handle].append(command.argv)
        return ExecutionResult(exit_code=0, stdout="fixture completed", stderr="")

    async def observe(self, handle):
        return {
            "spec": self.spec.model_dump(mode="json"),
            "fingerprint": "capture-test-v1",
            "snapshot_mime_type": "application/json",
        }

    async def upload(self, handle, path, data):
        self.files[path] = data

    async def download(self, handle, path):
        return self.files[path]

    async def snapshot(self, handle):
        return json.dumps(self.workspaces[handle]).encode()

    async def restore(self, handle, snapshot):
        self.workspaces[handle] = json.loads(snapshot)

    def diff(self, before, after):
        return {"changed": before != after}

    async def destroy(self, handle):
        del self.workspaces[handle]


class CaptureAdapter(AdapterBase):
    name, version = "capture-fixture", "test-v1"

    def environment_spec(self, task):
        return EnvironmentSpec(backend="memory", image=None)

    def available_tools(self, task):
        return ["exec", "finish"]

    def agent_context(self, task):
        return {"public_context": "A deterministic fixture"}

    def validate_action(self, task, action):
        assert action.operation in {"exec", "finish"}

    async def evaluate(self, runtime, task, run, termination):
        success = bool(runtime.environment.commands)
        return {
            "benchmark": self.name,
            "version": self.version,
            "metric": "fixture_success",
            "score": int(success),
            "success": success,
            "evaluator": "test-only",
        }


@pytest.fixture
def capture_runtime(store, artifacts):
    return ExperimentRuntime(
        SimpleNamespace(store=store, artifacts=artifacts), environment=MemoryEnvironment()
    )


def make_driver(runtime, tmp_path, agent_factory):
    return BenchmarkRunner(
        runtime,
        CaptureAdapter(),
        agent_factory,
        tmp_path,
        model="fixture-model",
        seed=17,
        temperature=0.25,
        replay_samples=0,
    )


def model_response(operation="exec", *, malformed=False):
    action = {
        "tool": "environment",
        "operation": operation,
        "arguments": {"command": {"argv": ["fixture-command"]}} if operation == "exec" else {},
        "expected_effect": "Complete the fixture",
    }
    return httpx.Response(
        200,
        json={
            "id": "fixture-response",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "not-json" if malformed else json.dumps(action)},
                }
            ],
        },
    )


async def test_runner_captures_exact_http_request_and_binds_actions_tokens_and_finish(
    capture_runtime, tmp_path, monkeypatch
):
    runtime = capture_runtime
    sent, captured = [], []

    async def reply(request):
        body = json.loads(request.content)
        state = json.loads(body["messages"][1]["content"])["state"]
        objects = await runtime.ledger.objects(state["experiment_id"])
        contexts = [obj for obj in objects.values() if isinstance(obj, DecisionContext)]
        # Capture is durable before the HTTP request, including finish decisions.
        assert len(contexts) == len(sent) + 1
        context = next(obj for obj in contexts if obj.payload == body)
        assert context.coverage == "complete"
        assert context.last_step_id == state["last_step_id"]
        assert len(context.source_refs) == 1
        assert objects[context.source_refs[0]].metadata["role"] == "model_input"
        captured.append(context)
        sent.append(body)
        return model_response("exec" if len(sent) == 1 else "finish")

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        agent = OpenAICompatibleAgentBackend(
            base_url="https://private-provider.invalid/v1",
            api_key="secret-test-credential",
            model="fixture-model",
            temperature=0.25,
            seed=17,
            max_tokens=321,
            client=client,
        )
        original_capture = runtime.capture_decision_context

        async def capture_then_change_settings(*args, **kwargs):
            context = await original_capture(*args, **kwargs)
            # HTTP and command normalization must use the frozen request, even
            # if mutable backend configuration changes after durable capture.
            agent.model, agent.temperature = "changed-after-capture", 0.9
            agent.seed, agent.max_tokens = 91, 99
            return context

        monkeypatch.setattr(runtime, "capture_decision_context", capture_then_change_settings)
        driver = make_driver(runtime, tmp_path, lambda task: agent)
        tasks = [BenchmarkTask("capture", "Run a fixture with Unicode: μ", "test")]
        row = (await driver.run(tasks))[0]
        assert row["failure"] is None
        assert row["success"] and row["evidence_id"] and row["claim_id"]
        assert row["steps"] == 1 and row["total_tokens"] == 16
        assert len(sent) == len(captured) == 2
        assert row["decision_context_ids"] == [context.context_id for context in captured]
        assert [call["decision_context_id"] for call in row["model_calls"]] == row[
            "decision_context_ids"
        ]
        assert row["pending_action"]["operation"] == "finish"
        assert row["pending_action"]["decision_context_id"] == captured[1].context_id

        trajectory = await runtime.get_trajectory(row["trajectory_id"])
        model_step, native_step = trajectory.steps
        assert model_step.action.decision_context_id == captured[0].context_id
        assert model_step.decision_context == captured[0]
        assert model_step.token_usage == row["model_calls"][0]["usage"]
        assert model_step.action.arguments["command"]["seed"] == 17
        assert native_step.action.arguments["tool_name"] == "benchmark.native_evaluate"
        assert native_step.decision_context is None and native_step.token_usage is None
        assert all(step.action.operation != "finish" for step in trajectory.steps)
        claim = await runtime.ledger.get(row["claim_id"], "Claim")
        assert claim.status == "SUPPORTED"

        system = sent[0]["messages"][0]
        assert system["role"] == "system"
        assert "Command schema:" in system["content"]
        assert '"rationale_summary"' in system["content"]
        assert sent[0]["response_format"] == {"type": "json_object"}
        assert sent[0]["temperature"] == 0.25 and sent[0]["max_tokens"] == 321
        assert sent[0]["seed"] == 17 and sent[0]["model"] == "fixture-model"
        serialized = "".join(context.payload_json for context in captured)
        assert "private-provider.invalid" not in serialized
        assert "secret-test-credential" not in serialized
        sft = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text().splitlines()]
        assert len(sft) == 1
        assert sft[0]["input"] == sent[0] and sft[0]["context_coverage"] == "complete"
        transitions = [
            json.loads(line) for line in (tmp_path / "rl_transition.jsonl").read_text().splitlines()
        ]
        assert len(transitions) == 1 and transitions[0]["state"] == sent[0]
        assert transitions[0]["next_state"] == sent[1]
        assert transitions[0]["next_decision_context_id"] == captured[1].context_id
        assert transitions[0]["next_context_coverage"] == "complete"
        assert transitions[0]["episode_boundary"] and transitions[0]["terminal"]
        assert transitions[0]["boundary_scope"] == "model_action_sequence"
        assert transitions[0]["terminated"] is None and transitions[0]["truncated"] is None
        assert transitions[0]["termination_reason"] == "agent_finish"
        canonical = [
            json.loads(line)
            for line in (Path(row["export_directory"]) / "rl_transition.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(canonical) == 2 and canonical[-1]["step_id"] == native_step.step_id
        assert canonical[0]["episode_boundary"] is False
        assert canonical[-1]["episode_boundary"] is True
        assert canonical[0]["next_state"] is None
        assert all("boundary_scope" not in value for value in canonical)
        # Resume does not request another decision or replay a finished action.
        again = (await driver.run(tasks, resume=True))[0]
        assert again["decision_context_ids"] == row["decision_context_ids"]
        assert len(sent) == 2 and len(runtime.environment.commands) == 1


async def test_custom_agent_captures_only_supplied_state_with_unknown_coverage(
    capture_runtime, tmp_path
):
    runtime = capture_runtime
    delivered = []

    class CustomAgent:
        def prepare_request(self, state):
            raise AssertionError("An arbitrary backend cannot attest to complete HTTP input")

        async def next_action(self, state):
            objects = await runtime.ledger.objects(state["state"]["experiment_id"])
            contexts = [obj for obj in objects.values() if isinstance(obj, DecisionContext)]
            assert len(contexts) == len(delivered) + 1
            assert any(obj.payload == state and obj.coverage == "unknown" for obj in contexts)
            delivered.append(deepcopy(state))
            # Local mutation must not rewrite captured input or the driver's history.
            state["history"].append({"hidden_agent_mutation": True})
            state["task"] = "changed only inside agent"
            return Action(
                tool="environment",
                operation="exec" if len(delivered) == 1 else "finish",
                arguments={"command": {"argv": ["fixture-command"], "seed": 17}}
                if len(delivered) == 1
                else {},
                decision_context_id="untrusted-agent-supplied-id",
            )

    driver = make_driver(runtime, tmp_path, lambda task: CustomAgent())
    row = (await driver.run([BenchmarkTask("custom", "Custom agent fixture", "test")]))[0]
    assert row["failure"] is None and row["steps"] == 1
    contexts = [
        await runtime.ledger.get(context_id, "DecisionContext")
        for context_id in row["decision_context_ids"]
    ]
    assert len(contexts) == 2
    assert [context.payload for context in contexts] == delivered
    assert all(context.coverage == "unknown" for context in contexts)
    assert delivered[0]["history"] == [] and len(delivered[1]["history"]) == 1
    assert "hidden_agent_mutation" not in json.dumps(delivered)
    trajectory = await runtime.get_trajectory(row["trajectory_id"])
    assert trajectory.steps[0].action.decision_context_id == contexts[0].context_id
    assert not (tmp_path / "sft.jsonl").read_text()
    transition = json.loads((tmp_path / "rl_transition.jsonl").read_text())
    assert transition["state"] == delivered[0]
    assert transition["context_coverage"] == "unknown"
    assert transition["next_state"] == delivered[1]
    assert transition["next_context_coverage"] == "unknown"
    assert transition["episode_boundary"] and transition["termination_reason"] == "agent_finish"


async def test_malformed_model_response_retains_input_without_retry(capture_runtime, tmp_path):
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        return model_response(malformed=True)

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        driver = make_driver(
            capture_runtime,
            tmp_path,
            lambda task: OpenAICompatibleAgentBackend(
                base_url="https://model.invalid/v1", api_key="secret", model="test", client=client
            ),
        )
        row = (await driver.run([BenchmarkTask("malformed", "Fixture", "test")]))[0]
    assert len(sent) == len(row["decision_context_ids"]) == 1
    assert row["failure"]["failure_type"] == "FORMAT_FAILURE"
    assert row["steps"] == 0 and not capture_runtime.environment.commands
    assert row["total_tokens"] == 8
    context = await capture_runtime.ledger.get(row["decision_context_ids"][0], "DecisionContext")
    assert context.payload == sent[0]
    assert row["model_calls"][0]["decision_context_id"] == context.context_id


async def test_prepared_request_is_not_rebuilt_from_changed_backend_settings():
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        return model_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        agent = OpenAICompatibleAgentBackend(
            base_url="https://model.invalid/v1",
            api_key="secret",
            model="original-model",
            seed=17,
            client=client,
        )
        state = {"task": "original input"}
        request = agent.prepare_request(state)
        expected = deepcopy(request)
        state["task"] = "mutated after preparation"
        agent.seed, agent.model, agent.temperature = 91, "changed-model", 1.0
        agent.max_tokens = 99
        action = await agent.next_action_from_request(request)
    assert sent == [expected]
    assert action.arguments["command"]["seed"] == 17
    assert agent.last_call["model"] == "original-model"


@pytest.mark.parametrize("ending", ["finish", "budget", "agent_error", "timeout"])
async def test_model_sequence_boundaries_and_next_inputs(capture_runtime, tmp_path, ending):
    runtime = capture_runtime

    class SequenceAgent:
        async def next_action(self, state):
            if len(state["history"]) == 2:
                if ending == "agent_error":
                    raise RuntimeError("Fixture agent stopped responding")
                return Action(tool="environment", operation="finish")
            return Action(
                tool="environment",
                operation="exec",
                arguments={"command": {"argv": ["fixture-command"], "seed": 17}},
            )

    if ending == "timeout":
        original_execute = runtime.environment.execute

        async def timeout_second(handle, command):
            result = await original_execute(handle, command)
            if len(runtime.environment.commands) == 2:
                return ExecutionResult(exit_code=-1, stdout="", stderr="timeout", timed_out=True)
            return result

        runtime.environment.execute = timeout_second
    driver = make_driver(runtime, tmp_path, lambda task: SequenceAgent())
    if ending == "budget":
        driver.max_steps = 2
    row = (await driver.run([BenchmarkTask(ending, "Two fixture executions", "test")]))[0]
    assert row["steps"] == 2
    transitions = [
        json.loads(line) for line in (tmp_path / "rl_transition.jsonl").read_text().splitlines()
    ]
    first, last = transitions
    assert first["next_state"] == last["state"]
    assert first["next_decision_context_id"] == last["decision_context_id"]
    assert first["episode_boundary"] is False
    assert first["terminated"] is False and first["truncated"] is False
    assert last["episode_boundary"] is True and last["terminal"] is True
    assert last["boundary_scope"] == "model_action_sequence"
    if ending in {"finish", "agent_error"}:
        context = await runtime.ledger.get(row["decision_context_ids"][-1], "DecisionContext")
        assert last["next_state"] == context.payload
        assert last["next_decision_context_id"] == context.context_id
        assert last["terminated"] is None and last["truncated"] is None
        assert last["termination_reason"] == (
            "agent_finish" if ending == "finish" else "agent_error"
        )
    else:
        assert last["next_state"] is None and last["next_decision_context_id"] is None
        assert last["terminated"] is False and last["truncated"] is True
        assert last["termination_reason"] == (
            "step_budget" if ending == "budget" else "execution_timeout"
        )

    # Canonical evaluator records and original full-trajectory boundaries survive.
    canonical = [
        json.loads(line)
        for line in (Path(row["export_directory"]) / "rl_transition.jsonl").read_text().splitlines()
    ]
    assert len(canonical) == 3
    assert canonical[-1]["action"]["arguments"]["tool_name"] == "benchmark.native_evaluate"
    assert canonical[-1]["episode_boundary"] is True
    assert all(value["episode_boundary"] is False for value in canonical[:-1])
