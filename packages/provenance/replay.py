from copy import deepcopy

from packages.core.events.store import GENESIS, event_hash
from packages.core.models.schemas import EventType as T


def empty_state():
    return {
        "evidence": {},
        "artifacts": {},
        "claims": {},
        "verifications": {},
        "executions": {},
        "sandbox_handle": None,
        "snapshot_id": None,
    }


async def replay(events, resolve):
    branches, checkpoints, heads, seen = {}, {}, {}, {}
    for sequence, event in enumerate(events, 1):
        if event.episode_id != events[0].episode_id:
            raise ValueError("Mixed episode history")
        if (sequence == 1) != (event.event_type == T.EPISODE_STARTED):
            raise ValueError("History requires exactly one initial episode.started")
        if event.sequence != sequence or event.event_id in seen:
            raise ValueError("Invalid event ordering or duplicate identity")
        if event.parent_event_id and event.parent_event_id not in seen:
            raise ValueError("Missing or forward parent reference")
        before = heads.get(
            event.branch_id,
            seen[event.parent_event_id].state_after_hash if event.parent_event_id else GENESIS,
        )
        if event.state_before_hash != before or event_hash(event) != event.state_after_hash:
            raise ValueError("Event hash chain failed")
        output = await resolve(event.output)
        inputs = await resolve(event.input)
        if event.event_type == T.BRANCH_CREATED:
            if event.branch_id in branches:
                raise ValueError("Branch already exists")
            cp = checkpoints[inputs["checkpoint_id"]]
            branches[event.branch_id] = deepcopy(cp["state"])
            branches[event.branch_id]["sandbox_handle"] = None
            branches[event.branch_id]["snapshot_id"] = cp["snapshot_id"]
        elif event.branch_id not in branches:
            if event.event_type != T.EPISODE_STARTED:
                raise ValueError("Branch lacks creation event")
            branches[event.branch_id] = empty_state()
        state = branches[event.branch_id]
        if event.status == "SUCCESS":
            if event.event_type == T.RETRIEVAL_RESULT:
                for evidence in output.get("evidence", []):
                    state["evidence"][evidence["evidence_id"]] = evidence
            elif event.event_type == T.ARTIFACT_CREATED and "artifact" in output:
                artifact = output["artifact"]
                state["artifacts"][artifact["artifact_id"]] = artifact
            elif event.event_type in (T.CLAIM_CREATED, T.CLAIM_UPDATED):
                claim = output["claim"]
                state["claims"][claim["claim_id"]] = claim
            elif event.event_type == T.VERIFICATION_RESULT and "verification" in output:
                verification = output["verification"]
                state["verifications"][verification["verification_id"]] = verification
                state["claims"][output["claim"]["claim_id"]] = output["claim"]
            elif event.event_type == T.SANDBOX_RESULT:
                if "handle" in output:
                    state["sandbox_handle"] = output["handle"]
                if output.get("destroyed"):
                    state["sandbox_handle"] = None
                if "exit_code" in output:
                    state["executions"][event.event_id] = output
            elif event.event_type == T.CHECKPOINT_CREATED:
                checkpoints[event.event_id] = {
                    "branch_id": event.branch_id,
                    "state": deepcopy(state),
                    "snapshot_id": output.get("snapshot_id"),
                }
            elif event.event_type == T.ROLLBACK_CREATED:
                cp = checkpoints[inputs["checkpoint_id"]]
                handle = state["sandbox_handle"]
                branches[event.branch_id] = deepcopy(cp["state"])
                branches[event.branch_id]["sandbox_handle"] = handle
                branches[event.branch_id]["snapshot_id"] = cp["snapshot_id"]
        if event.event_type == T.SANDBOX_RESULT and "exit_code" in output:
            branches[event.branch_id]["executions"][event.event_id] = output
        heads[event.branch_id] = event.state_after_hash
        seen[event.event_id] = event
    return {
        "schema_version": "1.0",
        "branches": branches,
        "checkpoints": checkpoints,
        "heads": heads,
        "event_count": len(events),
    }
