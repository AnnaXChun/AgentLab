import json
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from packages.core.models.schemas import EventType as T


class TrajectoryExporter(Protocol):
    async def export(self, events, edges, destination: Path, resolve) -> dict[str, str]: ...


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


class CanonicalExporter:
    async def export(self, events, edges, destination: Path, resolve) -> dict[str, str]:
        destination.mkdir(parents=True, exist_ok=False)
        canonical = [event.model_dump(mode="json") for event in events]
        write_jsonl(destination / "raw.jsonl", canonical)
        normalized = []
        for row in canonical:
            row = dict(row)
            usage = row.pop("usage")
            row.update({f"usage_{key}": value for key, value in usage.items()})
            for key in ("input", "output", "metadata"):
                row[key + "_json"] = json.dumps(row.pop(key), ensure_ascii=False, sort_keys=True)
            normalized.append(row)
        pq.write_table(pa.Table.from_pylist(normalized), destination / "events.parquet")
        write_jsonl(destination / "edges.jsonl", [edge.model_dump(mode="json") for edge in edges])
        inputs = {event.event_id: await resolve(event.input) for event in events}
        outputs = {event.event_id: await resolve(event.output) for event in events}
        by_id = {event.event_id: event for event in events}
        results = {
            inputs[e.event_id]["request_event_id"]: e
            for e in events
            if "request_event_id" in inputs[e.event_id]
        }
        messages, rl = {}, []
        action_types = {T.RETRIEVAL_QUERY, T.TOOL_CALL, T.SANDBOX_COMMAND, T.VERIFICATION_STARTED}
        for event in events:
            branch = messages.setdefault(event.branch_id, [])
            data = inputs[event.event_id]
            if event.event_type == T.AGENT_MESSAGE and outputs[event.event_id].get("visible"):
                branch.append({"role": "assistant", "content": outputs[event.event_id]["text"]})
            if event.event_type not in action_types:
                continue
            if event.event_type == T.SANDBOX_COMMAND and data.get("operation") != "exec":
                continue
            result = results.get(event.event_id)
            if result is None:
                continue  # Unfinished requests are not successful training examples.
            name = data.get("tool_name", event.event_type.value)
            arguments = data.get("arguments", data)
            branch.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": event.event_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            )
            branch.append(
                {
                    "role": "tool",
                    "tool_call_id": event.event_id,
                    "content": json.dumps(outputs[result.event_id], ensure_ascii=False),
                }
            )
            rl.append(
                {
                    "schema_version": "1.0",
                    "episode_id": event.episode_id,
                    "branch_id": event.branch_id,
                    "step": len(rl),
                    "state_ref": event.state_before_hash,
                    "action_event_id": event.event_id,
                    "action": data,
                    "observation": outputs[result.event_id],
                    "terminal": False,
                    "reward_optional": None,
                }
            )
        if any(e.event_type in (T.EPISODE_COMPLETED, T.EPISODE_FAILED) for e in events):
            for branch_id in {row["branch_id"] for row in rl}:
                next(row for row in reversed(rl) if row["branch_id"] == branch_id)["terminal"] = (
                    True
                )
        write_jsonl(
            destination / "sft.jsonl",
            [
                {
                    "schema_version": "1.0",
                    "episode_id": events[0].episode_id,
                    "branch_id": branch_id,
                    "messages": rows,
                }
                for branch_id, rows in messages.items()
                if rows
            ],
        )
        write_jsonl(destination / "rl.jsonl", rl)
        recovery = []
        for edge in edges:
            if edge.relation != "retry":
                continue
            failure = by_id[edge.source_event_id]
            corrected = by_id[edge.target_event_id]
            success = results.get(corrected.event_id)
            if failure.status != "FAILED" or success is None or success.status != "SUCCESS":
                continue
            original = by_id.get(inputs[failure.event_id].get("request_event_id"))
            recovery.append(
                {
                    "schema_version": "1.0",
                    "episode_id": failure.episode_id,
                    "failed_event_id": failure.event_id,
                    "corrected_event_id": corrected.event_id,
                    "success_event_id": success.event_id,
                    "failed_action": inputs[original.event_id] if original else {},
                    "failure": outputs[failure.event_id],
                    "corrected_action": inputs[corrected.event_id],
                    "success": outputs[success.event_id],
                }
            )
        write_jsonl(destination / "failure_recovery.jsonl", recovery)
        return {path.name: str(path.resolve()) for path in destination.iterdir()}
