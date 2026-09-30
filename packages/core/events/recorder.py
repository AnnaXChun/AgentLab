from time import perf_counter

from packages.core.events.store import canonical
from packages.core.models.schemas import Event, EventEdge, EventType, Usage, uid
from packages.provenance.tracing import span

PAYLOAD_LIMIT = 16 * 1024


class Recorder:
    def __init__(self, store, artifacts, episode_id: str, branch_id: str):
        self.store = store
        self.artifacts = artifacts
        self.episode_id = episode_id
        self.branch_id = branch_id

    async def emit(self, event_type: EventType, **fields) -> Event:
        edges = fields.pop("edges", None)
        if edges:
            fields["metadata"] = {
                **fields.get("metadata", {}),
                "edges": [e.model_dump(mode="json") for e in edges],
            }
        refs = list(fields.pop("artifact_refs", []))
        for key in ("input", "output", "metadata"):
            payload = fields.get(key, {})
            if len(canonical(payload)) > PAYLOAD_LIMIT:
                artifact = await self.artifact(
                    canonical(payload), "application/json", {"purpose": f"event.{key}"}
                )
                fields[key] = {"$artifact": artifact.artifact_id}
                refs.append(artifact.artifact_id)
        return self.store.append(
            Event(
                episode_id=self.episode_id,
                branch_id=self.branch_id,
                event_type=event_type,
                artifact_refs=refs,
                **fields,
            ),
            edges=edges,
        )

    async def artifact(self, data: bytes, mime_type: str, metadata: dict | None = None):
        event_id = uid()
        # The intent identifies the writer before the CAS is changed.
        self.store.append(
            Event(
                episode_id=self.episode_id,
                branch_id=self.branch_id,
                event_id=event_id,
                event_type=EventType.ARTIFACT_CREATED,
                status="PENDING",
                input={"mime_type": mime_type, "size_bytes": len(data)},
            )
        )
        try:
            artifact = await self.artifacts.put(data, event_id, mime_type, metadata)
        except Exception as exc:
            await self.emit(
                EventType.ARTIFACT_CREATED,
                status="FAILED",
                parent_event_id=event_id,
                output={"error": str(exc)},
            )
            raise
        await self.emit(
            EventType.ARTIFACT_CREATED,
            parent_event_id=event_id,
            output={"artifact": artifact.model_dump(mode="json")},
            artifact_refs=[artifact.artifact_id],
        )
        return artifact

    async def operation(
        self, start_type, result_type, inputs, function, result_edge_source=None, **fields
    ):
        component = {
            "retrieval": "RETRIEVER",
            "tool": "TOOL",
            "sandbox": "SANDBOX",
            "verification": "VERIFIER",
        }.get(start_type.value.split(".")[0], "HARNESS")
        with span(component, start_type.value, self.episode_id):
            intent = await self.emit(start_type, input=inputs, status="PENDING", **fields)
            started = perf_counter()
            try:
                output = await function()
            except Exception as exc:
                await self.emit(
                    result_type,
                    parent_event_id=intent.event_id,
                    input={"request_event_id": intent.event_id},
                    output={"error_type": type(exc).__name__, "error": str(exc)},
                    status="FAILED",
                    usage=Usage(wall_time_ms=(perf_counter() - started) * 1000),
                )
                raise
            status = "FAILED" if output.get("exit_code", 0) != 0 else "SUCCESS"
            result_id = uid()
            result_edges = (
                [
                    EventEdge(
                        source_event_id=result_edge_source,
                        target_event_id=result_id,
                        relation="verified_by",
                    )
                ]
                if result_edge_source
                else []
            )
            evidence_refs = [e["evidence_id"] for e in output.get("evidence", [])]
            artifact_refs = (
                list(output.get("files", {}).values())
                + list(output.get("input_artifacts", {}).values())
                + output.get("verification", {}).get("artifact_refs", [])
            )
            result = await self.emit(
                result_type,
                event_id=result_id,
                edges=result_edges,
                evidence_refs=evidence_refs,
                artifact_refs=artifact_refs,
                parent_event_id=intent.event_id,
                input={"request_event_id": intent.event_id},
                output=output,
                status=status,
                usage=Usage(wall_time_ms=(perf_counter() - started) * 1000),
            )
            return output, result

    async def resolve(self, payload):
        if isinstance(payload, dict) and set(payload) == {"$artifact"}:
            import json

            return json.loads(await self.artifacts.get(payload["$artifact"]))
        return payload
