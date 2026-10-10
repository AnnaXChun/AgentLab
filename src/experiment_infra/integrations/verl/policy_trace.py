"""Validate official token merges and persist policy/environment links."""

from experiment_infra.core.schema import PolicyIdentity, PolicyTrace
from packages.core.models.schemas import EventType as T


def identity_from_server(fields, policy_id, session_id, initial_checkpoint):
    low, high = fields.get("min_global_steps"), fields.get("max_global_steps")
    if type(low) is not int or low < 0 or low != high:
        raise ValueError("Missing or mixed inference-server weight versions")
    return PolicyIdentity(
        policy_id=policy_id,
        policy_version=f"{session_id}:global_step_{low:06d}",
        checkpoint_id=initial_checkpoint if low == 0 else f"{session_id}:actor/global_step_{low}",
        training_step=low,
        rollout_batch_id=f"{session_id}:sync_batch_{low:06d}",
    )


def validate_merge(previous_ids, previous_mask, result, mask, generated=None):
    """Consume CT metadata, never reconstruct generated token IDs or their masks.

    V0.4 supports append-only CT builders (including Qwen). A builder rewriting
    an earlier token prefix is rejected rather than guessing revised span bounds.
    """
    if result.removed_prefix_token_count or result.token_ids[: len(previous_ids)] != previous_ids:
        raise ValueError("This bridge requires append-only continuous-token merges")
    if mask[: len(previous_mask)] != previous_mask:
        raise ValueError("Official merge changed an existing response mask")
    appended = result.token_ids[len(previous_ids) :]
    extension = mask[len(previous_mask) :]
    if len(appended) != len(extension):
        raise ValueError("Token/mask alignment lost")
    if generated is not None:
        boundary = len(result.inserted_token_ids)
        if appended[boundary:] != generated or extension != [0] * boundary + [1] * len(generated):
            raise ValueError("Generated IDs changed or policy mask is invalid")
        return len(previous_mask) + boundary, len(mask)
    if any(extension):
        raise ValueError("Tool/context tokens must have mask=0")
    return len(previous_mask), len(mask)


def validate_on_policy_batch(traces):
    if not traces:
        raise ValueError("Empty on-policy batch")
    identities = set()
    for trace in traces:
        trace = PolicyTrace.model_validate(trace.model_dump())
        if trace.policy_id.startswith("SCRIPTED") or trace.checkpoint_id.startswith("NONE-"):
            raise ValueError("Scripted test tokens cannot enter GRPO")
        if trace.trajectory_source != "ONLINE_RL":
            raise ValueError("Only fresh ONLINE_RL traces may enter GRPO")
        identities.add(
            tuple(getattr(trace, k) for k in PolicyIdentity.model_fields if k != "schema_version")
        )
    if len(identities) != 1:
        raise ValueError("Mixed policy generations or rollout batches")


async def persist_trace(runtime, trace):
    trace = PolicyTrace.model_validate(trace.model_dump())
    run = await runtime._scoped(trace.run_id, trace.experiment_id, "Run")
    identity = PolicyIdentity.model_validate(
        {k: getattr(trace, k) for k in PolicyIdentity.model_fields}
    )
    if (
        run.trajectory_id != trace.trajectory_id
        or run.policy_identity != identity
        or run.trajectory_source != trace.trajectory_source
    ):
        raise ValueError("Trace identity does not match producing run")
    links = [(trace.trajectory_id, trace.policy_trace_id, "policy_tokens")]
    for span in trace.step_spans:
        if span.trajectory_step_id is None:
            continue
        step = await runtime._scoped(span.trajectory_step_id, trace.experiment_id, "TrajectoryStep")
        if step.trajectory_id != trace.trajectory_id:
            raise ValueError("Token span refers to another trajectory")
        links.append((step.step_id, trace.policy_trace_id, "action_token_span"))
    await runtime.ledger.record(trace.experiment_id, T.OBJECT_RECORDED, [trace], links)
