"""Replay comparison contracts; no Docker, database or subprocess execution."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from experiment_infra.core.schema import Action, EnvironmentSpec, Experiment, ObservationEvent, Run
from experiment_infra.runtime.replay import exact_replay
from experiment_infra.verifiers.policy import SuccessCriterion, VerifierPolicy

ARGV = ["python", "-m", "unittest"]


def summary(duration="0.001", count=1, verdict="OK"):
    noun = "test" if count == 1 else "tests"
    return f".\n{'-' * 70}\nRan {count} {noun} in {duration}s\n\n{verdict}\n"


def test_criterion(name="unit", argv=ARGV):
    return SuccessCriterion(
        criterion_id=name,
        policy=VerifierPolicy(
            policy_id=name + "-policy", verifier="test", parameters={"argv": list(argv)}
        ),
    )


# This is a fixture builder, not a pytest test function.
test_criterion.__test__ = False


def experiment(declaration):
    if declaration == "legacy":
        return Experiment(goal="Tests", success_criteria={"test": {"argv": ARGV}})
    return Experiment(goal="Tests", criteria=[test_criterion()])


async def replay(
    exp,
    *,
    old_stderr=None,
    new_stderr=None,
    old_stdout="",
    new_stdout="",
    new_exit=0,
    new_hash="same",
    argv=ARGV,
    tool="environment",
    operation="exec",
):
    original = Run(
        experiment_id=exp.experiment_id,
        environment_spec=EnvironmentSpec(),
        code_version="fixture",
    )
    action = Action(tool=tool, operation=operation, arguments={"command": {"argv": list(argv)}})
    old = SimpleNamespace(
        step_id="original-step",
        action=action,
        observation=ObservationEvent(
            action_event_id="original-action",
            stdout=old_stdout,
            stderr=summary() if old_stderr is None else old_stderr,
            exit_code=0,
            structured_output={"collected": {"result.txt": "old-artifact"}},
        ),
    )
    new = SimpleNamespace(
        step_id="replay-step",
        observation=ObservationEvent(
            action_event_id="replay-action",
            stdout=new_stdout,
            stderr=summary("0.002") if new_stderr is None else new_stderr,
            exit_code=new_exit,
            structured_output={"collected": {"result.txt": "new-artifact"}},
        ),
    )
    objects = {
        original.run_id: original,
        exp.experiment_id: exp,
        "old-artifact": SimpleNamespace(sha256="same"),
        "new-artifact": SimpleNamespace(sha256=new_hash),
    }
    runtime = SimpleNamespace(
        ledger=SimpleNamespace(
            get=AsyncMock(side_effect=lambda key, kind: objects[key]), record=AsyncMock()
        ),
        get_trajectory=AsyncMock(return_value=SimpleNamespace(steps=[old])),
        start_run=AsyncMock(return_value=SimpleNamespace(run_id="replay-run")),
        execute=AsyncMock(return_value=new),
        finish_run=AsyncMock(),
    )
    result = await exact_replay(runtime, original.run_id)
    runtime.finish_run.assert_awaited_once_with("replay-run")
    assert runtime.ledger.record.await_args.kwargs["output"] == result.model_dump(mode="json")
    return result


@pytest.mark.parametrize("declaration", ["legacy", "structured"])
@pytest.mark.parametrize("verdict", ["OK", "OK (skipped=1)", "FAILED (failures=1, errors=1)"])
async def test_declared_tests_ignore_only_unittest_elapsed_seconds(declaration, verdict):
    result = await replay(
        experiment(declaration),
        old_stderr=summary(verdict=verdict),
        new_stderr=summary("12.345", verdict=verdict),
    )
    assert result.matched
    comparison = result.comparisons[0]
    assert comparison["stdout_match"]
    assert not comparison["stderr_match"]
    assert comparison["stderr_comparison_match"]
    assert comparison["comparison_policy"] == "declared_test_unittest_timing_normalized"


@pytest.mark.parametrize("declaration", ["legacy", "structured"])
@pytest.mark.parametrize(
    "changes",
    [
        {"new_stdout": "changed"},
        {"old_stdout": summary(), "new_stdout": summary("0.002")},
        {"new_stderr": summary("0.002", count=2)},
        {"new_stderr": summary("0.002", verdict="FAILED (failures=1)")},
        {"new_stderr": summary("0.002", verdict="OK (skipped=1)")},
        {"old_stderr": "diagnostic A\n" + summary(), "new_stderr": "diagnostic B\n" + summary()},
        {"old_stderr": "elapsed=0.001", "new_stderr": "elapsed=0.002"},
        {"old_stderr": "Ran 1 test in 0.001s\nOK", "new_stderr": "Ran 1 test in 0.002s\nOK"},
        {"old_stderr": summary() + "trailing", "new_stderr": summary("0.002") + "trailing"},
        {"new_stderr": summary("<elapsed>")},
        {"new_stderr": summary("0.002").replace("-" * 70, "-" * 69)},
        {"old_stderr": summary() + summary(), "new_stderr": summary("0.002") + summary()},
        {"new_exit": 1},
        {"new_hash": "changed"},
    ],
)
async def test_other_changes_are_never_hidden(declaration, changes):
    assert not (await replay(experiment(declaration), **changes)).matched


@pytest.mark.parametrize("declaration", ["legacy", "structured"])
@pytest.mark.parametrize(
    "changes",
    [
        {"argv": [*ARGV, "other_suite"]},
        {"operation": "unsupported"},
        {"tool": "mcp", "operation": "call"},
    ],
)
async def test_only_declared_environment_exec_commands_normalize_timing(declaration, changes):
    result = await replay(experiment(declaration), **changes)
    assert not result.matched
    assert result.comparisons[0]["comparison_policy"] == "exit_code_artifacts_stdout_stderr"


async def test_every_structured_test_policy_is_recognized():
    other_argv = ["python", "tests.py"]
    exp = Experiment(
        goal="Multiple suites", criteria=[test_criterion(), test_criterion("other", other_argv)]
    )
    assert (await replay(exp)).matched
    assert (await replay(exp, argv=other_argv)).matched


async def test_structured_policies_do_not_fall_back_to_stale_legacy_argv():
    exp = Experiment(
        goal="Structured policy wins",
        criteria=[test_criterion("other", ["python", "tests.py"])],
        success_criteria={"test": {"argv": ARGV}},
    )
    assert not (await replay(exp)).matched
    assert (await replay(exp, argv=["python", "tests.py"])).matched


async def test_non_test_policy_does_not_normalize_even_with_matching_argv():
    exp = Experiment(
        goal="Custom evaluator",
        criteria=[
            SuccessCriterion(
                criterion_id="host",
                policy=VerifierPolicy(
                    policy_id="host-policy", verifier="host", parameters={"argv": ARGV}
                ),
            )
        ],
        success_criteria={"test": {"argv": ARGV}},
    )
    assert not (await replay(exp)).matched


@pytest.mark.parametrize("success_criteria", [{}, {"test": None}, {"test": []}])
async def test_undeclared_commands_keep_exact_output_comparison(success_criteria):
    exp = Experiment(goal="No declared test", success_criteria=success_criteria)
    assert not (await replay(exp)).matched
    assert (await replay(exp, new_stderr=summary())).matched
