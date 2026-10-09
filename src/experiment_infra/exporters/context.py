"""Use only context actually captured before the decision, never ledger state."""

import json

from experiment_infra.core.schema import DecisionContext


def captured_context(step, trajectory=None):
    if step.step_id != step.action.step_id or step.branch_id != step.action.run_id:
        raise ValueError("Action does not belong to this trajectory step and branch")
    if trajectory is not None and (
        step.trajectory_id != trajectory.trajectory_id
        or step.action.run_id != trajectory.run_id
        or step.action.experiment_id != trajectory.experiment_id
    ):
        raise ValueError("Step does not belong to this trajectory")
    context = step.decision_context
    if context is None:
        return None, {
            "decision_context_id": None,
            "context_coverage": "unknown",
            "decision_context_sha256": None,
            "context_source_refs": [],
        }
    # Validate again for callers passing an unchecked Pydantic model_copy.
    context = DecisionContext.model_validate(context.model_dump())
    if (
        context.context_id != step.action.decision_context_id
        or context.run_id != step.action.run_id
        or context.experiment_id != step.action.experiment_id
    ):
        raise ValueError("Decision context does not belong to this action and run")
    if context.last_step_id != step.parent_step_id:
        raise ValueError("Decision context is stale for this step's parent")
    if context.created_at > step.action.timestamp:
        raise ValueError("Decision context was captured after the action")
    # Returning a new decoded value also avoids aliasing caller-owned dictionaries.
    return json.loads(context.payload_json), {
        "decision_context_id": context.context_id,
        "context_coverage": context.coverage,
        "decision_context_sha256": context.payload_sha256,
        "context_source_refs": list(context.source_refs),
    }
