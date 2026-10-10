import json
from types import SimpleNamespace as NS

import pytest

from experiment_infra.core.schema import PolicyTrace
from experiment_infra.integrations.verl.loop import LoopLimits, parse_action, run_loop
from experiment_infra.integrations.verl.policy_trace import (
    identity_from_server,
    validate_on_policy_batch,
)
from experiment_infra.integrations.verl.session import RuntimeSession, ToyAdapter
from experiment_infra.runtime.executor import ExperimentRuntime
from packages.sandbox.base import ExecutionResult


class MemoryEnvironment:
    def __init__(self, fail=False):
        self.files, self.spec, self.fail = {}, None, fail
        self.destroyed = False

    async def create(self, spec):
        self.spec = spec
        return "memory"

    async def upload(self, handle, path, data):
        self.files[path] = data.decode()

    async def download(self, handle, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path].encode()

    async def execute(self, handle, command):
        if self.fail:
            raise RuntimeError("environment unavailable")
        if command.argv == ["write"]:
            self.files["answer.json"] = json.dumps({"sum": 4})
        return ExecutionResult(exit_code=0, stdout="ok", stderr="")

    async def observe(self, handle):
        return {
            "spec": self.spec.model_dump(mode="json"),
            "fingerprint": "test",
            "snapshot_mime_type": "application/json",
        }

    async def snapshot(self, handle):
        return json.dumps(self.files).encode()

    async def restore(self, handle, snapshot):
        self.files = json.loads(snapshot)

    def diff(self, a, b):
        return {"added": sorted(json.loads(b).keys() - json.loads(a).keys()), "modified": []}

    async def destroy(self, handle):
        self.destroyed = True


def action(command):
    return json.dumps(
        {"tool": "environment", "operation": "exec", "arguments": {"command": {"argv": [command]}}}
    )


FINISH = '{"tool":"environment","operation":"finish","arguments":{}}'
IDENTITY = {"policy_id": "test", "session_id": "test-session", "initial_checkpoint": "sha256:test"}


class MockHost:
    """Scripted original token IDs, intentionally unrelated to text encoding."""

    def __init__(self, replies, versions=None):
        self.replies = replies
        self.calls = 0
        self.versions = versions or [0] * len(replies)
        self.server_manager = self
        self.tokenizer = NS(decode=lambda ids, **_: self.replies[ids[0] - 100])

    async def generate(self, **kwargs):
        i = self.calls
        self.calls += 1
        return NS(
            token_ids=[100 + i, 999],
            log_probs=[-0.3, -0.2],
            extra_fields={
                "min_global_steps": self.versions[i],
                "max_global_steps": self.versions[i],
            },
        )

    async def ct_build_initial_tokens(self, messages):
        return [1, 2, 3]

    async def ct_merge_assistant_token(self, tokens, generated, mask, probs, assistant_logprobs):
        return (
            NS(token_ids=tokens + generated, removed_prefix_token_count=0, inserted_token_ids=[]),
            mask + [1] * len(generated),
            (probs or []) + assistant_logprobs,
        )

    async def ct_merge_context_msg(self, previous, updated, tokens, mask, probs):
        return (
            NS(token_ids=tokens + [44, 45], removed_prefix_token_count=0, inserted_token_ids=[]),
            mask + [0, 0],
            (probs or []) + [0.0, 0.0],
        )


@pytest.fixture
def session(store, artifacts):
    env = MemoryEnvironment()
    runtime = ExperimentRuntime(NS(store=store, artifacts=artifacts), environment=env)
    adapter = ToyAdapter()
    return RuntimeSession(runtime, adapter, adapter.list_tasks("train")[0])


async def rollout(session, replies, **limits):
    return await run_loop(
        MockHost(replies),
        session,
        [{"role": "user", "content": "task"}],
        {"temperature": 0.6},
        LoopLimits(**limits),
        IDENTITY,
    )


async def test_multistep_tokens_masks_provenance_and_export(session, tmp_path):
    trace, reward, failure, turns = await rollout(
        session, [action("inspect"), action("write"), FINISH]
    )
    assert failure is None and reward.scalar_reward == 1 and turns == 3
    assert trace.response_token_ids == [100, 999, 44, 45, 101, 999, 44, 45, 102, 999]
    assert trace.response_mask == [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
    assert trace.response_logprobs == [-0.3, -0.2, 0.0, 0.0, -0.3, -0.2, 0.0, 0.0, -0.3, -0.2]
    assert PolicyTrace.model_validate_json(trace.model_dump_json()) == trace
    assert trace.step_spans[1].action_token_start == 4
    assert trace.step_spans[1].tool_response_token_start == 6
    assert trace.step_spans[-1].trajectory_step_id is None
    rt = session.runtime
    run = await rt.ledger.get(trace.run_id, "Run")
    trajectory = await rt.get_trajectory(trace.trajectory_id)
    assert run.policy_identity == trajectory.policy_identity
    assert trajectory.trajectory_source == "ONLINE_RL"
    assert trajectory.agent_backend == "verl/test"
    graph = await rt.get_provenance(reward.reward_id)
    assert reward.evidence_ids[0] in graph["ancestors"]
    assert reward.verification_ids[0] in graph["ancestors"]
    assert any(n["kind"] == "Artifact" for n in graph["nodes"].values())
    assert any(n["kind"] == "ObservationEvent" for n in graph["nodes"].values())
    exported = await rt.export(trace.experiment_id, tmp_path)
    assert "policy_traces.jsonl" in exported["files"] and "rewards.jsonl" in exported["files"]
    assert session.runtime.environment.destroyed
    with pytest.raises(ValueError, match="Immutable"):
        from experiment_infra.integrations.verl.policy_trace import persist_trace

        await persist_trace(rt, trace)


@pytest.mark.parametrize(
    "text",
    ["no JSON", '{"tool": 3}', '{"tool":"environment","tool":"environment"}', "```json\n{}\n```"],
)
async def test_format_failure_is_terminal_and_verified(session, text):
    trace, reward, failure, _ = await rollout(session, [text])
    assert failure == "FORMAT_FAILURE" and reward.scalar_reward == 0
    assert reward.verification_ids and reward.evidence_ids
    assert trace.step_spans[0].action_token_start is None


async def test_runtime_failure_is_terminal(session):
    session.runtime.environment.fail = True
    trace, reward, failure, _ = await rollout(session, [action("inspect")])
    assert failure == "ENVIRONMENT_FAILURE" and reward.scalar_reward == 0
    assert trace.response_mask[:2] == [1, 1]


async def test_verifier_failure_is_terminal(session):
    async def broken(*args):
        raise RuntimeError("broken verifier")

    session.adapter.evaluate = broken
    _, reward, failure, _ = await rollout(session, [action("inspect"), action("write"), FINISH])
    assert failure == "VERIFIER_FAILURE" and reward.scalar_reward == 0


async def test_optional_repair_is_one_turn_only(session):
    trace, reward, failure, _ = await rollout(session, ["bad", "still bad"], allow_repair_once=True)
    assert failure == "FORMAT_FAILURE" and reward.scalar_reward == 0
    assert trace.response_mask == [1, 1, 0, 0, 1, 1]


async def test_span_and_source_integrity(session):
    trace, _, _, _ = await rollout(session, [action("inspect"), action("write"), FINISH])
    bad = trace.model_dump()
    bad["step_spans"][0]["action_token_end"] = 4
    with pytest.raises(ValueError):
        PolicyTrace.model_validate(bad)
    with pytest.raises(ValueError, match="ONLINE_RL"):
        validate_on_policy_batch([trace.model_copy(update={"trajectory_source": "OFFLINE"})])
    with pytest.raises(ValueError, match="Mixed"):
        validate_on_policy_batch([trace, trace.model_copy(update={"checkpoint_id": "another"})])
    with pytest.raises(ValueError, match="Weight version"):
        await run_loop(
            MockHost([action("inspect"), FINISH], [0, 1]),
            RuntimeSession(session.runtime, ToyAdapter(), ToyAdapter().list_tasks("train")[0]),
            [],
            {},
            LoopLimits(),
            IDENTITY,
        )


def test_missing_server_identity_fails_closed():
    with pytest.raises(ValueError):
        identity_from_server({}, **IDENTITY)


def test_no_heuristic_parsing():
    with pytest.raises(ValueError):
        parse_action("Here is the action: " + FINISH)


async def test_reward_rejects_forged_evidence(session):
    from experiment_infra.integrations.verl.reward_bridge import RewardBridge

    trace, reward, _, _ = await rollout(session, [action("inspect"), action("write"), FINISH])
    e = await session.runtime.ledger.get(reward.evidence_ids[0], "Evidence")
    with pytest.raises(ValueError, match="differs"):
        await RewardBridge().record(
            session.runtime, session.experiment, session.run, [e.model_copy(update={"value": 0})]
        )


async def test_budget_does_not_truncate_action_tokens_or_guess_tool_spans(session):
    trace, reward, failure, _ = await rollout(session, [action("inspect")], max_response_length=3)
    assert failure == "BUDGET_EXCEEDED" and reward.scalar_reward == 0
    assert trace.response_token_ids == [100, 999]
    assert trace.step_spans[0].trajectory_step_id is not None
    assert trace.step_spans[0].tool_response_token_start is None


async def test_scripted_demo_cannot_enter_grpo(session):
    session.source = "OFFLINE"
    trace, _, _, _ = await rollout(session, [action("inspect"), action("write"), FINISH])
    with pytest.raises(ValueError, match="ONLINE_RL"):
        validate_on_policy_batch([trace])


def test_dataset_contains_no_reference_solution_and_disjoint_splits(tmp_path):
    import pyarrow.parquet as pq

    from experiment_infra.integrations.verl.dataset import prepare

    result = prepare("toy", tmp_path, 5, 5)
    assert result == {"benchmark": "toy", "train": 5, "val": 5}
    train = pq.read_table(tmp_path / "train.parquet").to_pylist()
    val = pq.read_table(tmp_path / "val.parquet").to_pylist()
    assert not {r["extra_info"]["task_id"] for r in train} & {
        r["extra_info"]["task_id"] for r in val
    }
    assert all(r["agent_name"] == "experiment_runtime" for r in train)
    assert "evaluation_criteria" not in json.dumps(train)


def test_runtime_dependency_independence():
    import subprocess
    import sys

    command = "from experiment_infra.runtime.executor import ExperimentRuntime; import sys; assert not {'verl','ray','torch'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", command], check=True)


def test_token_merge_rejects_rewriting_and_wrong_mask():
    from experiment_infra.integrations.verl.policy_trace import validate_merge

    with pytest.raises(ValueError, match="append-only"):
        validate_merge([1, 2], [1], NS(token_ids=[1, 4], removed_prefix_token_count=1), [1])
    with pytest.raises(ValueError, match="mask=0"):
        validate_merge([1, 2], [1], NS(token_ids=[1, 2, 4], removed_prefix_token_count=0), [1, 1])


async def test_summary_validates_exported_provenance(session, tmp_path):
    from experiment_infra.integrations.verl.summary import summarize, write_sample

    trace, reward, failure, _ = await rollout(session, [action("inspect"), action("write"), FINISH])
    bundle = await session.runtime.export(trace.experiment_id, tmp_path / "exports")
    write_sample(
        tmp_path,
        trace,
        reward,
        failure,
        {"task_id": "sum-0", "split": "train"},
        bundle["directory"],
    )
    result = summarize(tmp_path)
    assert result[0]["valid_reward_provenance"] == 1
    assert result[0]["avg_policy_tokens"] == 6 and result[0]["avg_tool_tokens"] == 4
    path = tmp_path / "samples" / f"{trace.policy_trace_id}.json"
    row = json.loads(path.read_text())
    row["reward"]["scalar_reward"] = 0
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Sample does not match"):
        summarize(tmp_path)


async def test_replay_and_branch_are_never_online(session):
    from experiment_infra.core.schema import Action
    from experiment_infra.integrations.verl.policy_trace import identity_from_server

    await session.start(
        identity_from_server({"min_global_steps": 0, "max_global_steps": 0}, **IDENTITY)
    )
    step = await session.execute(Action.model_validate_json(action("inspect")))
    replayed = await session.runtime.replay(session.run.run_id)
    replay_run = await session.runtime.ledger.get(replayed.replay_run_id, "Run")
    assert replay_run.trajectory_source == "REPLAY"
    branched = await session.runtime.branch(session.run.run_id, step.step_id)
    assert branched.trajectory_source == "BRANCH"
    assert branched.policy_identity == session.run.policy_identity
    await session.runtime.finish_run(branched.run_id)
    await session.close()


def test_trace_requires_complete_generation_coverage_and_strict_token_ids():
    from experiment_infra.core.schema import PolicyTrace

    identity = identity_from_server({"min_global_steps": 0, "max_global_steps": 0}, **IDENTITY)
    values = {
        **identity.model_dump(),
        "experiment_id": "e",
        "run_id": "r",
        "trajectory_id": "t",
        "prompt_token_ids": [1],
        "response_token_ids": [2],
        "response_mask": [1],
        "step_spans": [],
        "sampling_params": {},
    }
    with pytest.raises(ValueError, match="Every policy token"):
        PolicyTrace.model_validate(values)
    values["step_spans"] = [{"response_token_start": 0, "response_token_end": 1}]
    values["response_token_ids"] = [2.1]
    with pytest.raises(ValueError):
        PolicyTrace.model_validate(values)
