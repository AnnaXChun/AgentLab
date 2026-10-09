from experiment_infra.exporters.context import captured_context
from experiment_infra.exporters.policy import ExportPolicy, training_metadata


def sft_rows(trajectory, *, policy=None):
    policy = ExportPolicy() if policy is None else ExportPolicy.model_validate(policy)
    metadata = training_metadata(trajectory, policy)
    if metadata["sample_weight"] is None:
        return
    for step in trajectory.steps:
        payload, context_metadata = captured_context(step, trajectory)
        coverage = context_metadata["context_coverage"]
        if coverage == "partial" and not policy.include_partial_context:
            continue
        if coverage == "unknown" and not policy.include_unknown_context:
            continue
        action = step.action.model_dump(
            mode="json",
            include={
                "schema_version",
                "tool",
                "operation",
                "arguments",
                "expected_effect",
                "reasoning_summary",
            },
        )
        yield {
            "schema_version": "2.0",
            "experiment_id": trajectory.experiment_id,
            "run_id": trajectory.run_id,
            "step_id": step.step_id,
            **metadata,
            **context_metadata,
            "input": payload,
            "output": {"action": action},
        }
