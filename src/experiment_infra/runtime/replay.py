from experiment_infra.core.schema import Action, CounterfactualLink, ReplayResult
from packages.core.models.schemas import EventType as T


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
        environment_spec=original.environment_spec,
        code_version=original.code_version,
    )
    experiment = await runtime.ledger.get(original.experiment_id, "Experiment")
    comparisons = []
    try:
        for step in trajectory.steps:
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
            test_command = experiment.success_criteria.get("test", {}).get("argv")
            is_test = (
                action.tool == "environment"
                and test_command is not None
                and action.arguments.get("command", {}).get("argv") == test_command
            )
            if not is_test:
                matched = matched and stdout_match and stderr_match
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
                    "comparison_policy": "declared_test_exit_code_and_artifacts"
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
