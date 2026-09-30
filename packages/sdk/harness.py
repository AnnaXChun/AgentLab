import json
import mimetypes
from pathlib import Path

from packages.core.events.recorder import Recorder
from packages.core.models.schemas import Claim, Episode, Event, EventEdge, uid
from packages.core.models.schemas import EventType as T
from packages.core.verification.service import VerifierService
from packages.provenance.replay import replay
from packages.sandbox.base import Command


class Harness:
    def __init__(
        self, store, artifacts, sandbox, retriever, gateway, pack, episode_id, branch_id=None
    ):
        self.store, self.artifacts, self.sandbox = store, artifacts, sandbox
        self.retriever, self.gateway, self.pack = retriever, gateway, pack
        self.episode_id = episode_id
        self.branch_id = branch_id or store.episode(episode_id).root_branch_id
        self.recorder = Recorder(store, artifacts, episode_id, self.branch_id)
        self.verifiers = VerifierService()

    @classmethod
    async def create(
        cls,
        store,
        artifacts,
        sandbox,
        retriever,
        gateway,
        pack,
        goal,
        agent_id="mock",
        metadata=None,
    ):
        ep = Episode(
            goal=goal,
            domain_pack=pack.metadata["name"],
            agent_id=agent_id,
            metadata={"pack": pack.model_dump(mode="json"), **(metadata or {})},
        )
        store.create_episode(
            ep,
            Event(
                episode_id=ep.episode_id,
                branch_id=ep.root_branch_id,
                event_type=T.EPISODE_STARTED,
                output={"episode": ep.model_dump(mode="json")},
            ),
        )
        return cls(store, artifacts, sandbox, retriever, gateway, pack, ep.episode_id)

    async def replay(self):
        return await replay(self.store.events(self.episode_id), self.recorder.resolve)

    async def state(self):
        result = await self.replay()
        if self.branch_id not in result["branches"]:
            raise ValueError("Unknown branch")
        return result["branches"][self.branch_id]

    async def writable(self):
        if self.store.episode(self.episode_id).status != "RUNNING":
            raise ValueError("Episode is terminal")
        return await self.state()

    async def ingest(self, data: bytes, source_uri: str, title: str):
        await self.writable()

        async def perform():
            source = await self.recorder.artifact(data, "text/markdown", {"source_uri": source_uri})
            ids = await self.retriever.ingest(data, source_uri, title)
            return {"evidence_ids": ids, "source_artifact": source.artifact_id}

        return (
            await self.recorder.operation(
                T.RETRIEVAL_INGEST,
                T.RETRIEVAL_INGESTED,
                {"source_uri": source_uri, "title": title},
                perform,
            )
        )[0]

    async def retrieve(self, query: str, top_k: int = 5):
        await self.writable()

        async def perform():
            evidence = await self.retriever.retrieve(query, top_k)
            return {"evidence": [e.model_dump(mode="json") for e in evidence]}

        return (
            await self.recorder.operation(
                T.RETRIEVAL_QUERY, T.RETRIEVAL_RESULT, {"query": query, "top_k": top_k}, perform
            )
        )[0]

    async def tool(self, tool_name: str, arguments: dict):
        await self.writable()
        validation_error = None
        try:
            self.gateway.validate(tool_name, arguments)
        except Exception as exc:
            validation_error = exc

        async def perform():
            if validation_error is not None:
                raise validation_error
            return await self.gateway.call(tool_name, arguments)

        return (
            await self.recorder.operation(
                T.TOOL_CALL,
                T.TOOL_RESULT,
                {"tool_name": tool_name, "arguments": arguments},
                perform,
            )
        )[0]

    async def ensure_sandbox(self):
        state = await self.writable()
        if state["sandbox_handle"]:
            return state["sandbox_handle"]

        async def create():
            handle = await self.sandbox.create({**self.pack.environment, **self.pack.resources})
            return {"handle": handle}

        result, _ = await self.recorder.operation(
            T.SANDBOX_COMMAND,
            T.SANDBOX_RESULT,
            {
                "operation": "create",
                "environment": self.pack.environment,
                "resources": self.pack.resources,
            },
            create,
        )
        handle = result["handle"]
        if state["snapshot_id"]:

            async def restore():
                await self.sandbox.restore(handle, await self.artifacts.get(state["snapshot_id"]))
                return {"restored": state["snapshot_id"]}

            await self.recorder.operation(
                T.SANDBOX_COMMAND,
                T.SANDBOX_RESULT,
                {"operation": "restore", "snapshot_id": state["snapshot_id"]},
                restore,
            )
        return handle

    async def execute(self, command: Command, retry_of: str | None = None):
        await self.writable()
        if retry_of:
            old = next(
                (e for e in self.store.events(self.episode_id) if e.event_id == retry_of), None
            )
            if old is None or old.status != "FAILED" or old.event_type != T.SANDBOX_RESULT:
                raise ValueError("retry_of must reference a failed sandbox result in this episode")
        handle = await self.ensure_sandbox()
        event_id = uid()

        async def perform():
            input_artifacts = {}
            for path, content in command.files.items():
                artifact = await self.recorder.artifact(
                    content.encode(),
                    mimetypes.guess_type(path)[0] or "text/plain",
                    {"path": path, "purpose": "sandbox.input", "command_event_id": event_id},
                )
                input_artifacts[path] = artifact.artifact_id
                await self.sandbox.upload(handle, path, content.encode())
            result = await self.sandbox.exec(handle, command)
            files = {}
            if result.exit_code == 0:
                for path in command.collect:
                    data = await self.sandbox.download(handle, path)
                    artifact = await self.recorder.artifact(
                        data,
                        mimetypes.guess_type(path)[0] or "application/octet-stream",
                        {"path": path, "command_event_id": event_id},
                    )
                    files[path] = artifact.artifact_id
            payload = result.model_dump(mode="json")
            payload["files"] = files
            payload["input_artifacts"] = input_artifacts
            return payload

        fields = {"event_id": event_id}
        if retry_of:
            fields["edges"] = [
                EventEdge(source_event_id=retry_of, target_event_id=event_id, relation="retry")
            ]
        output, event = await self.recorder.operation(
            T.SANDBOX_COMMAND,
            T.SANDBOX_RESULT,
            {"operation": "exec", "command": command.model_dump(mode="json"), "handle": handle},
            perform,
            **fields,
        )
        return {**output, "event_id": event.event_id}

    async def claim(self, text: str, evidence_refs: list[str], artifact_refs: list[str]):
        state = await self.writable()
        if (
            not set(evidence_refs) <= state["evidence"].keys()
            or not set(artifact_refs) <= state["artifacts"].keys()
        ):
            raise ValueError("Unresolved or cross-branch claim references")
        for artifact_id in artifact_refs:
            if not await self.artifacts.exists(artifact_id):
                raise ValueError("Missing claim artifact")
        event_id = uid()
        claim = Claim(
            episode_id=self.episode_id,
            text=text,
            evidence_refs=evidence_refs,
            artifact_refs=artifact_refs,
            created_by_event_id=event_id,
        )
        await self.recorder.emit(
            T.CLAIM_CREATED,
            event_id=event_id,
            output={"claim": claim.model_dump(mode="json")},
            evidence_refs=evidence_refs,
            artifact_refs=artifact_refs,
            edges=[
                EventEdge(
                    source_event_id=source_id, target_event_id=event_id, relation="derived_from"
                )
                for source_id in sorted(
                    {state["artifacts"][ref]["created_by_event_id"] for ref in artifact_refs}
                )
            ],
        )
        supported = self.verifiers.support(claim, set(state["evidence"]), set(state["artifacts"]))
        if supported != claim:
            await self.recorder.emit(
                T.CLAIM_UPDATED,
                output={"claim": supported.model_dump(mode="json")},
                evidence_refs=evidence_refs,
                artifact_refs=artifact_refs,
            )
        return supported.model_dump(mode="json")

    async def verify(self, claim_id: str, verifier_type: str, artifact_id: str, criteria: dict):
        state = await self.writable()
        claim = Claim.model_validate(state["claims"][claim_id])

        async def perform():
            if verifier_type not in self.pack.validators or artifact_id not in claim.artifact_refs:
                raise ValueError("Verifier or artifact is outside the claim contract")
            context = {
                "artifact_bytes": await self.artifacts.get(artifact_id),
                "artifact_id": artifact_id,
                "criteria": criteria,
            }
            if verifier_type == "execution":
                executions = [
                    x
                    for x in state["executions"].values()
                    if set(claim.artifact_refs) <= set(x.get("files", {}).values())
                ]
                if not executions:
                    raise ValueError("No recorded execution produced the required claim artifacts")
                context.update(
                    execution=executions[-1],
                    required_exist=[await self.artifacts.exists(a) for a in claim.artifact_refs],
                )
            updated, verification = await self.verifiers.verify(claim, verifier_type, context)
            return {
                "claim": updated.model_dump(mode="json"),
                "verification": verification.model_dump(mode="json"),
            }

        output, result = await self.recorder.operation(
            T.VERIFICATION_STARTED,
            T.VERIFICATION_RESULT,
            {
                "claim_id": claim_id,
                "verifier_type": verifier_type,
                "artifact_id": artifact_id,
                "criteria": criteria,
            },
            perform,
            result_edge_source=claim.created_by_event_id,
        )
        return output

    async def checkpoint(self):
        state = await self.writable()
        snapshot_id = None
        if state["sandbox_handle"]:

            async def snapshot():
                data = await self.sandbox.snapshot(state["sandbox_handle"])
                artifact = await self.recorder.artifact(
                    data, "application/x-tar", {"purpose": "sandbox.snapshot"}
                )
                return {"snapshot_id": artifact.artifact_id}

            output, _ = await self.recorder.operation(
                T.SANDBOX_COMMAND,
                T.SANDBOX_RESULT,
                {"operation": "snapshot", "handle": state["sandbox_handle"]},
                snapshot,
            )
            snapshot_id = output["snapshot_id"]
        event = await self.recorder.emit(T.CHECKPOINT_CREATED, output={"snapshot_id": snapshot_id})
        return {
            "schema_version": "1.0",
            "checkpoint_id": event.event_id,
            "snapshot_id": snapshot_id,
        }

    async def fork(self, checkpoint_id: str):
        await self.writable()
        cp = (await self.replay())["checkpoints"].get(checkpoint_id)
        if cp is None or cp["branch_id"] != self.branch_id:
            raise ValueError("Checkpoint must belong to the source branch")
        branch_id, event_id = uid(), uid()
        rec = Recorder(self.store, self.artifacts, self.episode_id, branch_id)
        await rec.emit(
            T.BRANCH_CREATED,
            event_id=event_id,
            parent_event_id=checkpoint_id,
            input={"checkpoint_id": checkpoint_id},
            edges=[
                EventEdge(source_event_id=checkpoint_id, target_event_id=event_id, relation="fork")
            ],
        )
        return self.on_branch(branch_id)

    def on_branch(self, branch_id: str):
        return Harness(
            self.store,
            self.artifacts,
            self.sandbox,
            self.retriever,
            self.gateway,
            self.pack,
            self.episode_id,
            branch_id,
        )

    async def rollback(self, checkpoint_id: str):
        await self.writable()
        cp = (await self.replay())["checkpoints"].get(checkpoint_id)
        if cp is None or cp["branch_id"] != self.branch_id:
            raise ValueError("Cannot roll back to a different branch checkpoint")
        if cp["snapshot_id"]:
            handle = await self.ensure_sandbox()

            async def restore():
                await self.sandbox.restore(handle, await self.artifacts.get(cp["snapshot_id"]))
                return {"restored": cp["snapshot_id"]}

            await self.recorder.operation(
                T.SANDBOX_COMMAND,
                T.SANDBOX_RESULT,
                {"operation": "restore", "checkpoint_id": checkpoint_id},
                restore,
            )
        else:
            await self.destroy_sandbox()
        event_id = uid()
        await self.recorder.emit(
            T.ROLLBACK_CREATED,
            event_id=event_id,
            input={"checkpoint_id": checkpoint_id},
            edges=[
                EventEdge(
                    source_event_id=checkpoint_id, target_event_id=event_id, relation="rollback"
                )
            ],
        )
        return await self.state()

    async def destroy_sandbox(self):
        state = await self.state()
        if state["sandbox_handle"]:

            async def destroy():
                await self.sandbox.destroy(state["sandbox_handle"])
                return {"destroyed": True}

            await self.recorder.operation(
                T.SANDBOX_COMMAND,
                T.SANDBOX_RESULT,
                {"operation": "destroy", "handle": state["sandbox_handle"]},
                destroy,
            )

    async def complete(self, failed=False):
        await self.writable()
        if self.branch_id != self.store.episode(self.episode_id).root_branch_id:
            raise ValueError("Only root branch may finalize the episode")
        for branch_id in (await self.replay())["branches"]:
            await self.on_branch(branch_id).destroy_sandbox()
        await self.recorder.emit(T.EPISODE_FAILED if failed else T.EPISODE_COMPLETED)
        return self.store.episode(self.episode_id)

    async def export(self, output_root: str | Path):
        from packages.exporters.trajectory import CanonicalExporter

        destination = Path(output_root) / self.episode_id / uid()
        await self.recorder.emit(
            T.EXPORT_STARTED, input={"destination": str(destination)}, status="PENDING"
        )
        events = self.store.events(self.episode_id)
        try:
            paths = await CanonicalExporter().export(
                events, self.store.edges(self.episode_id), destination, self.recorder.resolve
            )
            # Export the CAS closure so replay does not depend on the live store.
            from adapters.storage.local import LocalArtifactStore

            exported_cas = LocalArtifactStore(destination / "artifacts")
            refs = {ref for event in events for ref in event.artifact_refs}
            for ref in refs:
                await exported_cas.put(await self.artifacts.get(ref), "export")
            manifest = {
                "schema_version": "1.0",
                "episode_id": self.episode_id,
                "event_count": len(events),
                "last_event_hash": events[-1].state_after_hash,
                "artifact_ids": sorted(refs),
            }
            (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
            paths["manifest.json"] = str((destination / "manifest.json").resolve())
            await self.recorder.emit(
                T.EXPORT_COMPLETED, output={"paths": paths, "cutoff_sequence": events[-1].sequence}
            )
        except Exception as exc:
            await self.recorder.emit(
                T.EXPORT_COMPLETED, status="FAILED", output={"error": str(exc)}
            )
            raise
        return {"schema_version": "1.0", "paths": paths, "event_count": len(events)}
