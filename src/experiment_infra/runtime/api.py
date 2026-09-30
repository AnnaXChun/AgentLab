import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from experiment_infra.core import schema as m


class CreateExperiment(m.Model):
    goal: str = Field(min_length=1, max_length=8000)
    environment_spec: m.EnvironmentSpec = Field(default_factory=m.EnvironmentSpec)
    agent_backend: str = "external"
    constraints: dict[str, Any] = Field(default_factory=dict)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    parent_experiment_id: str | None = None


class HypothesisRequest(m.Model):
    statement: str
    rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
    expected_observation: dict[str, Any] = Field(default_factory=dict)
    parent_hypothesis_id: str | None = None


class RunRequest(m.Model):
    hypothesis_ids: list[str] | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0


class ActionRequest(m.Model):
    run_id: str
    action: m.Action


class ClaimRequest(m.Model):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)


class VerifyRequest(m.Model):
    run_id: str
    hypothesis_id: str
    verifier: Literal["metric", "test"]
    artifact_ids: list[str] = Field(min_length=1)


class BranchRequest(m.Model):
    from_step: int | str
    config: dict[str, Any] | None = None


class ExportRequest(m.Model):
    experiment_id: str


class Result(m.Model):
    data: dict[str, Any]


def router():
    routes = APIRouter(tags=["Experiment infrastructure"])

    async def call(request, fn):
        try:
            async with request.app.state.experiment_api_lock:
                return await fn(request.app.state.experiments)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc
        except (ValueError, PermissionError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @routes.post("/experiments", response_model=m.Experiment)
    async def create(request: Request, body: CreateExperiment):
        return await call(
            request,
            lambda rt: rt.create_experiment(
                **body.model_dump(exclude={"schema_version", "environment_spec"}),
                environment_spec=body.environment_spec,
            ),
        )

    @routes.get("/experiments/{experiment_id}", response_model=m.Experiment)
    async def get(request: Request, experiment_id: str):
        return await call(request, lambda rt: rt.ledger.get(experiment_id, "Experiment"))

    @routes.get("/experiments/{experiment_id}/state", response_model=Result)
    async def state(request: Request, experiment_id: str, run_id: str | None = None):
        return Result(data=await call(request, lambda rt: rt.get_state(experiment_id, run_id)))

    @routes.post("/experiments/{experiment_id}/hypotheses", response_model=m.Hypothesis)
    async def hypothesis(request: Request, experiment_id: str, body: HypothesisRequest):
        return await call(
            request,
            lambda rt: rt.register_hypothesis(
                experiment_id, **body.model_dump(exclude={"schema_version"})
            ),
        )

    @routes.post("/experiments/{experiment_id}/runs", response_model=m.Run)
    async def run(request: Request, experiment_id: str, body: RunRequest):
        return await call(
            request,
            lambda rt: rt.start_run(experiment_id, **body.model_dump(exclude={"schema_version"})),
        )

    @routes.post("/experiments/{experiment_id}/actions", response_model=m.TrajectoryStep)
    async def action(request: Request, experiment_id: str, body: ActionRequest):
        return await call(
            request, lambda rt: rt.execute(experiment_id, body.action, run_id=body.run_id)
        )

    @routes.post("/experiments/{experiment_id}/claims", response_model=m.Claim)
    async def claim(request: Request, experiment_id: str, body: ClaimRequest):
        return await call(
            request,
            lambda rt: rt.create_claim(
                experiment_id, **body.model_dump(exclude={"schema_version"})
            ),
        )

    @routes.post("/experiments/{experiment_id}/verify", response_model=m.Evidence)
    async def verify(request: Request, experiment_id: str, body: VerifyRequest):
        return await call(
            request,
            lambda rt: rt.verify(experiment_id, **body.model_dump(exclude={"schema_version"})),
        )

    @routes.post("/runs/{run_id}/replay", response_model=m.ReplayResult)
    async def replay(request: Request, run_id: str):
        return await call(request, lambda rt: rt.replay(run_id))

    @routes.post("/runs/{run_id}/branch", response_model=m.Run)
    async def branch(request: Request, run_id: str, body: BranchRequest):
        return await call(request, lambda rt: rt.branch(run_id, body.from_step, config=body.config))

    @routes.post("/runs/{run_id}/finish", response_model=m.Run)
    async def finish_run(request: Request, run_id: str):
        return await call(request, lambda rt: rt.finish_run(run_id))

    @routes.post("/experiments/{experiment_id}/finish", response_model=m.Experiment)
    async def finish_experiment(request: Request, experiment_id: str):
        return await call(request, lambda rt: rt.finish_experiment(experiment_id))

    @routes.get("/objects/{object_id}/provenance", response_model=Result)
    async def provenance(request: Request, object_id: str):
        return Result(data=await call(request, lambda rt: rt.get_provenance(object_id)))

    @routes.get("/trajectories/{trajectory_id}", response_model=m.Trajectory)
    async def trajectory(request: Request, trajectory_id: str):
        return await call(request, lambda rt: rt.get_trajectory(trajectory_id))

    @routes.post("/exports", response_model=Result)
    async def export(request: Request, body: ExportRequest):
        destination = Path(os.getenv("EXPORT_ROOT", "outputs")) / "experiments"
        return Result(
            data=await call(request, lambda rt: rt.export(body.experiment_id, destination))
        )

    return routes
