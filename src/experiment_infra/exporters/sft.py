def sft_rows(trajectory):
    for step in trajectory.steps:
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
            "input": {"state": step.state_before, "context_refs": step.context_refs},
            "output": {"action": action},
        }
