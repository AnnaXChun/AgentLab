import re

from experiment_infra.core.schema import Action, CounterfactualLink, ReplayResult
from packages.core.models.schemas import EventType as T

_UNITTEST_SUMMARY = re.compile(
    r"(?m)^(?P<prefix>-{70}\nRan \d+ tests? in )\d+\.\d+"
    r"(?P<suffix>s\n\n(?:OK(?: \([^\r\n]*\))?|FAILED \([^\r\n]*\))\n?)\Z"
)


def _declared_test(experiment, action):
    if action.tool != "environment" or action.operation != "exec":
        return False
    command = action.arguments.get("command")
    argv = command.get("argv") if isinstance(command, dict) else None
    if argv is None:
        return False
    if experiment.criteria:
        # Structured policies are authoritative, as in resolve_criterion. A stale
        # legacy declaration must not broaden the comparison exception.
        return any(
            criterion.policy.verifier == "test" and criterion.policy.parameters["argv"] == argv
            for criterion in experiment.criteria
        )
    legacy = experiment.success_criteria.get("test")
    return isinstance(legacy, dict) and legacy.get("argv") == argv


def _test_stderr_match(expected, actual):
    # Recognize only the terminal standard unittest summary. Test counts,
    # verdicts, tracebacks and all other output stay byte-for-byte significant.
    old = _UNITTEST_SUMMARY.search(expected)
    new = _UNITTEST_SUMMARY.search(actual)
    if old is None or new is None:
        return expected == actual
    return (
        expected[: old.start()] == actual[: new.start()]
        and old["prefix"] == new["prefix"]
        and old["suffix"] == new["suffix"]
    )


async def exact_replay(runtime, run_id):
    original = await runtime.ledger.get(run_id, "Run")
    trajectory = await runtime.get_trajectory(original.trajectory_id)
    replayed = await runtime.start_run(
        original.experiment_id,
        hypothesis_ids=original.hypothesis_ids,
        config=original.config,
        seed=original.seed,
        snapshot_id=original.environment_snapshot_id,
        parent_run_id=run_id,
        purpose="exact_replay",
        environment_spec=original.environment_spec,
        code_version=original.code_version,
    )
    experiment = await runtime.ledger.get(original.experiment_id, "Experiment")
    comparisons = []
    try:
        for step in trajectory.steps:
            # Re-execution is not a new agent decision. Do not reuse the original
            # decision_context_id: it belongs to the source run and decision.
            action = Action(
                **step.action.model_dump(
                    include={
                        "tool",
                        "operation",
                        "arguments",
                        "expected_effect",
                        "context_refs",
                        "reasoning_summary",
                    }
                )
            )
            executed = await runtime.execute(original.experiment_id, action, run_id=replayed.run_id)
            expected = {
                path: (await runtime.ledger.get(aid, "Artifact")).sha256
                for path, aid in step.observation.structured_output.get("collected", {}).items()
            }
            actual = {
                path: (await runtime.ledger.get(aid, "Artifact")).sha256
                for path, aid in executed.observation.structured_output.get("collected", {}).items()
            }
            matched = (
                expected == actual and step.observation.exit_code == executed.observation.exit_code
            )
            stdout_match = step.observation.stdout == executed.observation.stdout
            stderr_match = step.observation.stderr == executed.observation.stderr
            is_test = _declared_test(experiment, action)
            stderr_comparison_match = (
                _test_stderr_match(step.observation.stderr, executed.observation.stderr)
                if is_test
                else stderr_match
            )
            matched = matched and stdout_match and stderr_comparison_match
            if action.tool != "environment":
                old = {
                    k: v
                    for k, v in step.observation.structured_output.items()
                    if k != "log_artifact_id"
                }
                new = {
                    k: v
                    for k, v in executed.observation.structured_output.items()
                    if k != "log_artifact_id"
                }
                matched = matched and old == new
            comparisons.append(
                {
                    "original_step_id": step.step_id,
                    "replay_step_id": executed.step_id,
                    "expected_artifact_hashes": expected,
                    "actual_artifact_hashes": actual,
                    "expected_exit_code": step.observation.exit_code,
                    "actual_exit_code": executed.observation.exit_code,
                    "matched": matched,
                    "stdout_match": stdout_match,
                    "stderr_match": stderr_match,
                    "stderr_comparison_match": stderr_comparison_match,
                    "comparison_policy": "declared_test_unittest_timing_normalized"
                    if is_test
                    else "exit_code_artifacts_stdout_stderr",
                }
            )
    finally:
        await runtime.finish_run(replayed.run_id)
    result = ReplayResult(
        original_run_id=run_id,
        replay_run_id=replayed.run_id,
        matched=all(row["matched"] for row in comparisons),
        comparisons=comparisons,
    )
    await runtime.ledger.record(
        original.experiment_id, T.REPLAY_RESULT, output=result.model_dump(mode="json")
    )
    return result


async def branch_replay(runtime, run_id, from_step, *, config=None):
    original = await runtime.ledger.get(run_id, "Run")
    trajectory = await runtime.get_trajectory(original.trajectory_id)
    if isinstance(from_step, int):
        if not 1 <= from_step <= len(trajectory.steps):
            raise ValueError("from_step is one-based and must name a completed step")
        step = trajectory.steps[from_step - 1]
    else:
        step = next((s for s in trajectory.steps if s.step_id == from_step), None)
        if step is None:
            raise ValueError("Step does not belong to source run")
    if not step.state_after.get("snapshot_id"):
        raise ValueError("Selected step has no restorable snapshot")
    return await runtime.start_run(
        original.experiment_id,
        hypothesis_ids=original.hypothesis_ids,
        config=original.config if config is None else {**original.config, **config},
        seed=original.seed,
        snapshot_id=step.state_after["snapshot_id"],
        parent_run_id=run_id,
        parent_step_id=step.step_id,
        purpose="branch",
        environment_spec=original.environment_spec,
        code_version=original.code_version,
    )


async def link_counterfactual(
    runtime,
    original_step_id,
    counterfactual_step_id,
    intervention_type,
    *,
    removed_context_refs=(),
    modified_context_refs=(),
    outcome_delta=None,
):
    original = await runtime.ledger.get(original_step_id, "TrajectoryStep")
    alternative = await runtime.ledger.get(counterfactual_step_id, "TrajectoryStep")
    experiment_id = original.action.experiment_id
    if alternative.action.experiment_id != experiment_id:
        raise ValueError("Counterfactual steps must belong to one experiment")
    run = await runtime.ledger.get(alternative.action.run_id, "Run")
    if run.parent_run_id != original.action.run_id:
        raise ValueError("Counterfactual step must belong to a branch of the original run")
    link = CounterfactualLink(
        original_step_id=original_step_id,
        counterfactual_step_id=counterfactual_step_id,
        intervention_type=intervention_type,
        removed_context_refs=list(removed_context_refs),
        modified_context_refs=list(modified_context_refs),
        original_outcome=original.observation.model_dump(mode="json"),
        counterfactual_outcome=alternative.observation.model_dump(mode="json"),
        outcome_delta=outcome_delta or {},
    )
    await runtime.ledger.record(
        experiment_id,
        T.COUNTERFACTUAL_LINKED,
        [link],
        [
            (original_step_id, link.link_id, "original"),
            (counterfactual_step_id, link.link_id, "counterfactual"),
        ],
    )
    return link
