def rl_rows(trajectory):
    for index, step in enumerate(trajectory.steps):
        yield {
            "schema_version": "2.0",
            "experiment_id": trajectory.experiment_id,
            "run_id": trajectory.run_id,
            "step_id": step.step_id,
            "state": step.state_before,
            "action": step.action.model_dump(mode="json"),
            "observation": step.observation.model_dump(mode="json"),
            "reward_signals": step.reward_signals,
            "next_state": step.state_after,
            "terminal": trajectory.final_status != "RUNNING" and index == len(trajectory.steps) - 1,
            "verifier_feedback": [v.model_dump(mode="json") for v in step.verifier_feedback],
        }
