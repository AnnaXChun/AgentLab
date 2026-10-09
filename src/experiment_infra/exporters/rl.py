from experiment_infra.exporters.context import captured_context
from experiment_infra.exporters.policy import ExportPolicy, training_metadata


def boundary_metadata(trajectory, step, index):
    boundary = (
        trajectory.final_status in {"COMPLETED", "FAILED"} and index == len(trajectory.steps) - 1
    )
    execution = step.observation.structured_output.get("execution")
    timed_out = (
        step.action.tool == "environment"
        and step.action.operation == "exec"
        and isinstance(execution, dict)
        and execution.get("timed_out") is True
    )
    if not boundary:
        terminated, truncated, reason = False, False, None
    elif timed_out:
        terminated, truncated, reason = False, True, "execution_timeout"
    else:
        # A caller finishing a run does not establish natural task termination.
        terminated, truncated, reason = None, None, "unknown"
    return {
        "episode_boundary": boundary,
        # Compatibility field: this is only a recorded episode boundary.
        "terminal": boundary,
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": reason,
    }


def rl_rows(trajectory, *, policy=None):
    policy = ExportPolicy() if policy is None else ExportPolicy.model_validate(policy)
    metadata = training_metadata(trajectory, policy)
    if metadata["sample_weight"] is None:
        return
    for index, step in enumerate(trajectory.steps):
        payload, context_metadata = captured_context(step, trajectory)
        # The next decision's captured input is the only known next policy state.
        # Execution state and subsequently attached verifier results are diagnostics.
        next_context = None
        next_metadata = {"context_coverage": "unknown", "decision_context_id": None}
        if index + 1 < len(trajectory.steps):
            next_context, next_metadata = captured_context(trajectory.steps[index + 1], trajectory)
        yield {
            "schema_version": "2.0",
            "experiment_id": trajectory.experiment_id,
            "run_id": trajectory.run_id,
            "step_id": step.step_id,
            **metadata,
            **context_metadata,
            "state": payload,
            "action": step.action.model_dump(mode="json"),
            "observation": step.observation.model_dump(mode="json"),
            "reward_signals": step.reward_signals,
            "next_state": next_context,
            "next_context_coverage": next_metadata["context_coverage"],
            "next_decision_context_id": next_metadata["decision_context_id"],
            **boundary_metadata(trajectory, step, index),
            "verifier_feedback": [v.model_dump(mode="json") for v in step.verifier_feedback],
        }
