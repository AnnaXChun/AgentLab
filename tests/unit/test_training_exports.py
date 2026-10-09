import hashlib
import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from experiment_infra.core import schema as m
from experiment_infra.exporters.canonical import export_experiment, lineage_root
from experiment_infra.exporters.policy import ExportPolicy
from experiment_infra.exporters.rl import rl_rows
from experiment_infra.exporters.sft import sft_rows
from experiment_infra.runtime.replay import branch_replay, exact_replay


def trajectory(*, coverage="complete", purpose="experiment", experiment_id="experiment", root=None):
    payload = {"messages": [{"role": "user", "content": "Choose using visible facts only"}]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    context = (
        m.DecisionContext(
            experiment_id=experiment_id,
            run_id="run",
            payload_json=encoded,
            payload_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
            coverage=coverage,
            source_refs=["visible-source"],
        )
        if coverage is not None
        else None
    )
    action = m.ActionEvent(
        experiment_id=experiment_id,
        run_id="run",
        step_id="step",
        tool="mcp",
        operation="run",
        arguments={"x": 1},
        expected_effect="result",
        agent_backend="external",
        decision_context_id=context.context_id if context else None,
    )
    observation = m.ObservationEvent(action_event_id=action.event_id, exit_code=0)
    verification = m.VerificationResult(
        experiment_id=experiment_id,
        run_id="run",
        verifier="later-verifier",
        status="PASSED",
        source_artifact_ids=[],
        metric="future-signal",
        value="not available before action",
    )
    step = m.TrajectoryStep(
        trajectory_id="trajectory",
        step_id="step",
        state_before={"private_retrospective": "never export as policy input"},
        context_refs=["not-a-model-payload"],
        decision_context=context,
        action=action,
        observation=observation,
        state_after={"future_feedback": "not a captured input", "snapshot_id": "snapshot"},
        verifier_feedback=[verification],
        latency=0,
        branch_id="run",
    )
    return m.Trajectory(
        trajectory_id="trajectory",
        experiment_id=experiment_id,
        run_id="run",
        agent_backend="external",
        steps=[step],
        final_status="COMPLETED",
        purpose=purpose,
        lineage_root_experiment_id=root,
    )


def test_sft_uses_captured_payload_not_retrospective_state():
    source = trajectory()
    row = next(sft_rows(source))
    assert row["input"] == source.steps[0].decision_context.payload
    assert row["context_coverage"] == "complete"
    assert row["context_source_refs"] == ["visible-source"]
    assert "private_retrospective" not in json.dumps(row)
    assert "future-signal" not in json.dumps(row)
    assert set(row["output"]) == {"action"}
    row["input"]["messages"].clear()
    assert source.steps[0].decision_context.payload["messages"]


@pytest.mark.parametrize("coverage", [None, "partial", "unknown"])
def test_incomplete_context_is_excluded_from_default_sft(coverage):
    source = trajectory(coverage=coverage)
    assert list(sft_rows(source)) == []
    policy = ExportPolicy(include_partial_context=True, include_unknown_context=True)
    row = next(sft_rows(source, policy=policy))
    assert row["context_coverage"] == (coverage or "unknown")
    assert row["input"] is None if coverage is None else row["input"] is not None


def test_rl_never_invents_context_or_reward_from_future_verifiers():
    row = next(rl_rows(trajectory(coverage=None)))
    assert row["state"] is None
    assert row["context_coverage"] == "unknown"
    assert row["reward_signals"] == {}
    assert row["verifier_feedback"][0]["metric"] == "future-signal"
    assert row["next_state"] is None
    assert row["terminal"] is True


def test_rl_next_state_is_the_next_captured_decision():
    source = trajectory()
    fresh_step = trajectory().steps[0]
    next_context = fresh_step.decision_context.model_copy(update={"last_step_id": "step"})
    next_step = fresh_step.model_copy(
        update={
            "step_id": "next-step",
            "parent_step_id": "step",
            "decision_context": next_context,
            "action": fresh_step.action.model_copy(update={"step_id": "next-step"}),
        }
    )
    source = source.model_copy(update={"steps": [source.steps[0], next_step]})
    first, last = list(rl_rows(source))
    assert first["state"] == source.steps[0].decision_context.payload
    assert first["next_state"] == next_step.decision_context.payload
    assert first["next_decision_context_id"] == next_step.decision_context.context_id
    assert not first["terminal"] and last["terminal"]
    assert last["next_state"] is None


@pytest.mark.parametrize("rows", [sft_rows, rl_rows])
def test_exact_replay_requires_explicit_policy_and_has_visible_weight(rows):
    replay = trajectory(purpose="exact_replay")
    assert list(rows(replay)) == []
    policy = ExportPolicy(include_exact_replays=True, exact_replay_weight=0.25)
    row = next(rows(replay, policy=policy))
    assert row["run_purpose"] == "exact_replay"
    assert row["sample_weight"] == 0.25
    assert next(rows(trajectory(purpose="branch")))["sample_weight"] == 1


def test_related_experiments_runs_and_branches_share_split():
    policy = ExportPolicy(split_seed="stable-dataset")
    sources = [
        trajectory(experiment_id="root", root="root"),
        trajectory(experiment_id="child", root="root", purpose="branch"),
        trajectory(experiment_id="sibling", root="root"),
    ]
    rows = [next(sft_rows(source, policy=policy)) for source in sources]
    assert {row["split_group"] for row in rows} == {"root"}
    assert len({row["split"] for row in rows}) == 1
    assert policy.split("root") == ExportPolicy(split_seed="stable-dataset").split("root")
    assert ExportPolicy(test_fraction=0).split("root") == "train"
    assert ExportPolicy(test_fraction=1).split("root") == "test"


@pytest.mark.parametrize(
    "policy", [{"exact_replay_weight": 0}, {"test_fraction": 1.01}, {"unknown": True}]
)
def test_export_policy_rejects_unsafe_or_misspelled_configuration(policy):
    with pytest.raises(ValidationError):
        ExportPolicy.model_validate(policy)


def export_runtime(source):
    root = m.Experiment(experiment_id="root", goal="root experiment")
    child = m.Experiment(
        experiment_id="child", parent_experiment_id="root", goal="child experiment"
    )
    by_id = {"root": root, "child": child}
    objects = {source.trajectory_id: source}
    if source.steps[0].decision_context:
        context = source.steps[0].decision_context
        objects[context.context_id] = context
    event = SimpleNamespace(
        artifact_refs=[],
        sequence=1,
        state_after_hash="head",
        model_dump=lambda **kwargs: {"sequence": 1},
    )
    return SimpleNamespace(
        ledger=SimpleNamespace(
            get=AsyncMock(side_effect=lambda key, kind: by_id[key]),
            objects=AsyncMock(return_value=objects),
            record=AsyncMock(),
        ),
        get_trajectory=AsyncMock(return_value=source),
        get_provenance=AsyncMock(return_value={"nodes": [], "edges": []}),
        storage=SimpleNamespace(store=SimpleNamespace(events=lambda _: [event])),
    )


async def test_canonical_export_resolves_parent_lineage_and_preserves_context(tmp_path):
    source = trajectory(experiment_id="child")
    runtime = export_runtime(source)
    result = await export_experiment(runtime, "child", tmp_path)
    from pathlib import Path

    directory = Path(result["directory"])
    row = json.loads((directory / "sft.jsonl").read_text())
    assert row["split_group"] == "root"
    assert row["input"] == source.steps[0].decision_context.payload
    context = json.loads((directory / "decision_contexts.jsonl").read_text())
    assert context["payload_json"] == source.steps[0].decision_context.payload_json
    assert result["manifest"]["lineage_root_experiment_id"] == "root"
    assert result["manifest"]["training_policy"]["include_exact_replays"] is False
    assert result["manifest"]["context_coverage"] == {"complete": 1, "partial": 0, "unknown": 0}


async def test_canonical_keeps_replay_and_legacy_records_but_excludes_default_sft(tmp_path):
    source = trajectory(experiment_id="child", purpose="exact_replay", coverage=None)
    result = await export_experiment(export_runtime(source), "child", tmp_path)
    assert result["manifest"]["files"]["trajectory.jsonl"]["rows"] == 1
    assert result["manifest"]["files"]["sft.jsonl"]["rows"] == 0
    assert result["manifest"]["files"]["rl_transition.jsonl"]["rows"] == 0
    assert result["manifest"]["context_coverage"]["unknown"] == 1


async def test_lineage_cycles_fail_closed():
    runtime = export_runtime(trajectory())
    runtime.ledger.get = AsyncMock(
        return_value=m.Experiment(experiment_id="cycle", goal="cycle", parent_experiment_id="cycle")
    )
    with pytest.raises(ValueError, match="cycle"):
        await lineage_root(runtime, "cycle")


async def test_replay_tags_run_purpose_without_reusing_source_decision_context():
    source = trajectory()
    original = SimpleNamespace(
        trajectory_id=source.trajectory_id,
        experiment_id=source.experiment_id,
        hypothesis_ids=[],
        config={},
        seed=0,
        environment_snapshot_id="initial-snapshot",
        environment_spec=m.EnvironmentSpec(),
        code_version="source",
    )
    experiment = SimpleNamespace(success_criteria={}, criteria=[])
    runtime = SimpleNamespace(
        ledger=SimpleNamespace(
            get=AsyncMock(side_effect=lambda key, kind: original if kind == "Run" else experiment),
            record=AsyncMock(),
        ),
        get_trajectory=AsyncMock(return_value=source),
        start_run=AsyncMock(return_value=SimpleNamespace(run_id="replay")),
        execute=AsyncMock(return_value=source.steps[0]),
        finish_run=AsyncMock(),
    )
    assert (await exact_replay(runtime, source.run_id)).matched
    assert runtime.start_run.call_args.kwargs["purpose"] == "exact_replay"
    submitted = runtime.execute.call_args.args[1]
    assert submitted.decision_context_id is None
    await branch_replay(runtime, source.run_id, 1)
    assert runtime.start_run.call_args.kwargs["purpose"] == "branch"


@pytest.mark.parametrize(
    "field,value", [("run_id", "foreign-run"), ("context_id", "foreign-context")]
)
def test_export_refuses_mismatched_decision_context(field, value):
    source = trajectory()
    context = source.steps[0].decision_context.model_copy(update={field: value})
    step = source.steps[0].model_copy(update={"decision_context": context})
    source = source.model_copy(update={"steps": [step]})
    for rows in (sft_rows, rl_rows):
        with pytest.raises(ValueError, match="does not belong"):
            list(rows(source))


def test_export_refuses_context_captured_after_action():
    source = trajectory()
    step = source.steps[0]
    context = step.decision_context.model_copy(
        update={"created_at": step.action.timestamp + timedelta(seconds=1)}
    )
    source = source.model_copy(
        update={"steps": [step.model_copy(update={"decision_context": context})]}
    )
    with pytest.raises(ValueError, match="after the action"):
        list(sft_rows(source))


def test_export_refuses_context_with_corrupt_payload_hash():
    source = trajectory()
    context = source.steps[0].decision_context.model_copy(
        update={"payload_json": '{"forged":true}'}
    )
    step = source.steps[0].model_copy(update={"decision_context": context})
    with pytest.raises(ValueError, match="hash mismatch"):
        list(sft_rows(source.model_copy(update={"steps": [step]})))


@pytest.mark.parametrize("rows", [sft_rows, rl_rows])
def test_missing_legacy_run_purpose_is_unknown_and_excluded(rows):
    serialized = trajectory().model_dump()
    serialized.pop("purpose")
    source = m.Trajectory.model_validate(serialized)
    assert source.purpose == "unknown"
    assert list(rows(source)) == []
    included = next(rows(source, policy=ExportPolicy(include_unknown_purpose=True)))
    assert included["run_purpose"] == "unknown"
    assert included["sample_weight"] == 1.0


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED"])
def test_rl_finish_status_does_not_invent_termination_cause(status):
    source = trajectory().model_copy(update={"final_status": status})
    row = next(rl_rows(source))
    assert row["terminal"] and row["episode_boundary"]
    assert row["terminated"] is None
    assert row["truncated"] is None
    assert row["termination_reason"] == "unknown"


def test_rl_actual_final_execution_timeout_is_truncation():
    source = trajectory()
    original = source.steps[0]
    step = original.model_copy(
        update={
            "action": original.action.model_copy(
                update={"tool": "environment", "operation": "exec"}
            )
        }
    )
    observation = step.observation.model_copy(
        update={"structured_output": {"execution": {"timed_out": True}}, "exit_code": 124}
    )
    source = source.model_copy(
        update={
            "final_status": "FAILED",
            "steps": [step.model_copy(update={"observation": observation})],
        }
    )
    row = next(rl_rows(source))
    assert row["terminated"] is False
    assert row["truncated"] is True
    assert row["termination_reason"] == "execution_timeout"
    running = next(rl_rows(source.model_copy(update={"final_status": "RUNNING"})))
    assert not running["terminal"]
    assert running["terminated"] is False and running["truncated"] is False


@pytest.mark.parametrize("field,value", [("step_id", "wrong-step"), ("branch_id", "wrong-run")])
def test_export_refuses_wrong_action_step_binding(field, value):
    source = trajectory()
    step = source.steps[0].model_copy(update={field: value})
    with pytest.raises(ValueError, match="trajectory step and branch"):
        list(sft_rows(source.model_copy(update={"steps": [step]})))


def test_export_refuses_stale_parent_context():
    source = trajectory()
    step = source.steps[0].model_copy(update={"parent_step_id": "newer-parent"})
    with pytest.raises(ValueError, match="stale"):
        list(rl_rows(source.model_copy(update={"steps": [step]})))


def test_export_refuses_cross_trajectory_step():
    source = trajectory()
    with pytest.raises(ValueError, match="Step does not belong"):
        list(sft_rows(source.model_copy(update={"run_id": "foreign-run"})))


@pytest.mark.parametrize("execution", [None, "text", {"timed_out": True}])
def test_non_environment_output_cannot_assert_a_timeout(execution):
    source = trajectory()
    step = source.steps[0]
    observation = step.observation.model_copy(
        update={"structured_output": {"execution": execution}}
    )
    step = step.model_copy(update={"observation": observation})
    row = next(rl_rows(source.model_copy(update={"steps": [step]})))
    assert row["truncated"] is None
    assert row["termination_reason"] == "unknown"
