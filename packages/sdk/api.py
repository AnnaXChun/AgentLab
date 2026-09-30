import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from jsonschema import ValidationError as JSONSchemaError
from pydantic import Field
from sqlalchemy import text

from packages.core.models.schemas import Claim, Episode, Event, Schema
from packages.sandbox.base import Command
from packages.sdk.runtime import Runtime


class CreateEpisode(Schema):
    goal: str = Field(min_length=1, max_length=8000)
    domain_pack: str = "demo"
    agent_id: str = "external"


class BranchRequest(Schema):
    branch_id: str | None = None


class RetrieveRequest(BranchRequest):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=100)


class ToolRequest(BranchRequest):
    arguments: dict[str, Any] = Field(default_factory=dict)


class ExecRequest(BranchRequest):
    command: Command
    retry_of: str | None = None


class ClaimRequest(BranchRequest):
    text: str = Field(min_length=1, max_length=8000)
    evidence_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


class VerifyRequest(BranchRequest):
    verifier_type: str
    artifact_id: str
    criteria: dict[str, Any]


class CheckpointRequest(BranchRequest):
    checkpoint_id: str


class IngestRequest(BranchRequest):
    content: str = Field(max_length=1000000)
    source_uri: str
    title: str


class CompleteRequest(BranchRequest):
    failed: bool = False


class Result(Schema):
    data: dict[str, Any]


class EventList(Schema):
    events: list[Event]


def create_app(runtime=None, experiment_runtime=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime or Runtime()
        app.state.locks = {}
        from experiment_infra.runtime.executor import ExperimentRuntime

        app.state.experiments = experiment_runtime or ExperimentRuntime(app.state.runtime)
        app.state.experiment_api_lock = asyncio.Lock()
        yield
        if runtime is None:
            app.state.runtime.engine.dispose()

    app = FastAPI(title="Agent Experiment Infrastructure", version="0.3.0", lifespan=lifespan)

    async def invoke(episode_id, branch_id, operation):
        lock = app.state.locks.setdefault(episode_id, asyncio.Lock())
        async with lock:
            try:
                h = app.state.runtime.load(episode_id, branch_id)
                await h.state()
                return await operation(h)
            except KeyError as exc:
                raise HTTPException(404, detail=str(exc)) from exc
            except (ValueError, PermissionError, JSONSchemaError) as exc:
                raise HTTPException(400, detail=str(exc)) from exc

    @app.get("/health", response_model=Result)
    async def health():
        with app.state.runtime.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return Result(data={"status": "ok"})

    @app.post("/episodes", response_model=Episode)
    async def create(body: CreateEpisode):
        try:
            h = await app.state.runtime.create(body.goal, body.domain_pack, body.agent_id)
            return h.store.episode(h.episode_id)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @app.get("/episodes/{episode_id}", response_model=Episode)
    async def get(episode_id: str):
        try:
            return app.state.runtime.store.episode(episode_id)
        except KeyError as exc:
            raise HTTPException(404, detail="Episode not found") from exc

    @app.post("/episodes/{episode_id}/retrieve", response_model=Result)
    async def retrieve(episode_id: str, body: RetrieveRequest):
        return Result(
            data=await invoke(
                episode_id, body.branch_id, lambda h: h.retrieve(body.query, body.top_k)
            )
        )

    @app.post("/episodes/{episode_id}/ingest", response_model=Result)
    async def ingest(episode_id: str, body: IngestRequest):
        return Result(
            data=await invoke(
                episode_id,
                body.branch_id,
                lambda h: h.ingest(body.content.encode(), body.source_uri, body.title),
            )
        )

    @app.post("/episodes/{episode_id}/tools/{tool_name}", response_model=Result)
    async def tool(episode_id: str, tool_name: str, body: ToolRequest):
        return Result(
            data=await invoke(
                episode_id, body.branch_id, lambda h: h.tool(tool_name, body.arguments)
            )
        )

    @app.post("/episodes/{episode_id}/sandbox/exec", response_model=Result)
    async def execute(episode_id: str, body: ExecRequest):
        return Result(
            data=await invoke(
                episode_id, body.branch_id, lambda h: h.execute(body.command, body.retry_of)
            )
        )

    @app.post("/episodes/{episode_id}/claims", response_model=Claim)
    async def claim(episode_id: str, body: ClaimRequest):
        return await invoke(
            episode_id,
            body.branch_id,
            lambda h: h.claim(body.text, body.evidence_refs, body.artifact_refs),
        )

    @app.post("/episodes/{episode_id}/claims/{claim_id}/verify", response_model=Result)
    async def verify(episode_id: str, claim_id: str, body: VerifyRequest):
        return Result(
            data=await invoke(
                episode_id,
                body.branch_id,
                lambda h: h.verify(claim_id, body.verifier_type, body.artifact_id, body.criteria),
            )
        )

    @app.post("/episodes/{episode_id}/checkpoint", response_model=Result)
    async def checkpoint(episode_id: str, body: BranchRequest):
        return Result(data=await invoke(episode_id, body.branch_id, lambda h: h.checkpoint()))

    @app.post("/episodes/{episode_id}/fork", response_model=Result)
    async def fork(episode_id: str, body: CheckpointRequest):
        async def perform(h):
            branch = await h.fork(body.checkpoint_id)
            return {"branch_id": branch.branch_id}

        return Result(data=await invoke(episode_id, body.branch_id, perform))

    @app.post("/episodes/{episode_id}/rollback", response_model=Result)
    async def rollback(episode_id: str, body: CheckpointRequest):
        return Result(
            data=await invoke(episode_id, body.branch_id, lambda h: h.rollback(body.checkpoint_id))
        )

    @app.get("/episodes/{episode_id}/events", response_model=EventList)
    async def events(episode_id: str):
        try:
            app.state.runtime.store.episode(episode_id)
        except KeyError as exc:
            raise HTTPException(404, detail="Episode not found") from exc
        return EventList(events=app.state.runtime.store.events(episode_id))

    @app.get("/episodes/{episode_id}/replay", response_model=Result)
    async def replay(episode_id: str):
        return Result(data=await invoke(episode_id, None, lambda h: h.replay()))

    @app.post("/episodes/{episode_id}/complete", response_model=Episode)
    async def complete(episode_id: str, body: CompleteRequest):
        return await invoke(episode_id, body.branch_id, lambda h: h.complete(body.failed))

    @app.post("/episodes/{episode_id}/export", response_model=Result)
    async def export(episode_id: str, body: BranchRequest):
        return Result(
            data=await invoke(
                episode_id, body.branch_id, lambda h: h.export(os.getenv("EXPORT_ROOT", "outputs"))
            )
        )

    from experiment_infra.runtime.api import router

    app.include_router(router())
    return app


app = create_app()
