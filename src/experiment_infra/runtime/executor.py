import asyncio
import hashlib
import json
import mimetypes
from time import perf_counter

from experiment_infra.core.ledger import Ledger
from experiment_infra.core.schema import (
    Action,
    ActionEvent,
    Artifact,
    Claim,
    DecisionContext,
    EnvironmentSpec,
    Evidence,
    Experiment,
    Hypothesis,
    ObservationEvent,
    Run,
    Snapshot,
    Trajectory,
    TrajectoryStep,
)
from experiment_infra.environments.docker import DockerEnvironmentBackend
from experiment_infra.verifiers.metric import MetricVerifier
from experiment_infra.verifiers.test import TestVerifier
from packages.core.models.schemas import EventType as T
from packages.core.models.schemas import now, uid
from packages.provenance.tracing import span
from packages.sandbox.base import Command
from packages.sdk.runtime import environment_metadata


class ExperimentRuntime:
    """Execute one caller-selected action. This class never invokes an agent."""

    def __init__(
        self,
        storage_runtime=None,
        *,
        environment=None,
        gateway=None,
        retriever=None,
        trusted_evaluators=None,
    ):
        from experiment_infra.verifiers.policy import HostEvaluator

        if storage_runtime is None:
            from packages.sdk.runtime import Runtime

            storage_runtime = Runtime(initialize_sandbox=environment is None)
        self.storage = storage_runtime
        self.ledger = Ledger(storage_runtime.store, storage_runtime.artifacts)
        self.environment = environment or DockerEnvironmentBackend(storage_runtime.sandbox)
        self.gateway, self.retriever = gateway, retriever
        self.verifiers = {"metric": MetricVerifier(), "test": TestVerifier()}
        for name, evaluator in (trusted_evaluators or {}).items():
            if not isinstance(name, str) or not name or name in self.verifiers:
                raise ValueError("Host evaluator names must be nonempty and distinct from builtins")
            if not isinstance(evaluator, HostEvaluator):
                raise ValueError("Trusted evaluators must be host-created HostEvaluator adapters")
            self.verifiers[name] = evaluator
        self.locks = {}

    async def create_experiment(
        self,
        goal,
        *,
        environment_spec=None,
        agent_backend="external",
        constraints=None,
        success_criteria=None,
        budget=None,
        parent_experiment_id=None,
        criteria=None,
    ):
        if parent_experiment_id:
            await self.ledger.get(parent_experiment_id, "Experiment")
        experiment = Experiment(
            criteria=criteria or [],
            goal=goal,
            environment_spec=environment_spec or EnvironmentSpec(),
            agent_backend=agent_backend,
            constraints=constraints or {},
            success_criteria=success_criteria or {},
            budget=budget or {},
            parent_experiment_id=parent_experiment_id,
        )
        self.ledger.create_partition(experiment)
        links = (
            [(parent_experiment_id, experiment.experiment_id, "derived_from")]
            if parent_experiment_id
            else []
        )
        await self.ledger.record(
            experiment.experiment_id, T.EXPERIMENT_CREATED, [experiment], links
        )
        return experiment

    async def register_hypothesis(
        self,
        experiment_id,
        statement,
        *,
        rationale="",
        assumptions=None,
        expected_observation=None,
        parent_hypothesis_id=None,
        criterion_ids=None,
    ):
        experiment = await self.ledger.get(experiment_id, "Experiment")
        if parent_hypothesis_id:
            await self._scoped(parent_hypothesis_id, experiment_id, "Hypothesis")
        criterion_ids = criterion_ids or []
        declared = {c.criterion_id for c in experiment.criteria}
        if declared and not criterion_ids:
            raise ValueError("Structured hypotheses require declared criterion IDs")
        if len(set(criterion_ids)) != len(criterion_ids) or not set(criterion_ids) <= declared:
            raise ValueError("Unknown or duplicate hypothesis criteria")
        hypothesis = Hypothesis(
            criterion_ids=criterion_ids,
            experiment_id=experiment_id,
            statement=statement,
            rationale=rationale,
            assumptions=assumptions or [],
            expected_observation=expected_observation or {},
            parent_hypothesis_id=parent_hypothesis_id,
        )
        updated = experiment.model_copy(
            update={"hypothesis_ids": [*experiment.hypothesis_ids, hypothesis.hypothesis_id]}
        )
        links = [(experiment_id, hypothesis.hypothesis_id, "hypothesis")]
        if parent_hypothesis_id:
            links.append((parent_hypothesis_id, hypothesis.hypothesis_id, "refines"))
        await self.ledger.record(
            experiment_id, T.HYPOTHESIS_REGISTERED, [hypothesis, updated], links
        )
        return hypothesis

    async def _scoped(self, object_id, experiment_id, kind=None):
        if self.ledger.locate(object_id, kind) != experiment_id:
            raise ValueError("Cross-experiment reference")
        return await self.ledger.get(object_id, kind)

    async def _artifact(
        self,
        run,
        data,
        artifact_type,
        *,
        step_id=None,
        parents=(),
        metadata=None,
        source_id=None,
        mime_type="application/octet-stream",
    ):
        for parent in parents:
            await self._scoped(parent, run.experiment_id, "Artifact")
        intent = await self.ledger.record(
            run.experiment_id,
            T.ARTIFACT_CREATED,
            input={
                "run_id": run.run_id,
                "step_id": step_id,
                "type": artifact_type,
                "size": len(data),
            },
            status="PENDING",
        )
        stored = await self.storage.artifacts.put(data, intent.event_id, mime_type)
        artifact = Artifact(
            type=artifact_type,
            uri=stored.uri,
            sha256=stored.sha256,
            size=stored.size_bytes,
            mime_type=mime_type,
            producer_run_id=run.run_id,
            producer_step_id=step_id,
            parent_artifact_ids=list(parents),
            metadata=metadata or {},
        )
        links = [(run.run_id, artifact.artifact_id, "produced")]
        links += [(parent, artifact.artifact_id, "artifact_version") for parent in parents]
        if source_id:
            links.append((source_id, artifact.artifact_id, "produced"))
        await self.ledger.record(
            run.experiment_id,
            T.ARTIFACT_CREATED,
            [artifact],
            links,
            output={"intent_event_id": intent.event_id},
        )
        return artifact

    async def artifact_bytes(self, artifact_id):
        artifact = await self.ledger.get(artifact_id, "Artifact")
        return await self.storage.artifacts.get("sha256:" + artifact.sha256)

    async def _snapshot(self, run, *, step_id=None, source_id=None):
        intent = await self.ledger.record(
            run.experiment_id,
            T.ENVIRONMENT_OPERATION,
            input={"operation": "snapshot", "run_id": run.run_id},
            status="PENDING",
        )
        data = await self.environment.snapshot(run.environment_handle)
        environment = await self.environment.observe(run.environment_handle)
        artifact = await self._artifact(
            run,
            data,
            "checkpoint",
            step_id=step_id,
            source_id=source_id,
            mime_type=environment.get("snapshot_mime_type", "application/octet-stream"),
        )
        snapshot = Snapshot(
            experiment_id=run.experiment_id,
            run_id=run.run_id,
            step_id=step_id,
            artifact_id=artifact.artifact_id,
            environment=environment,
        )
        workspace_versions = {}
        for obj in (await self.ledger.objects(run.experiment_id)).values():
            if (
                isinstance(obj, Artifact)
                and obj.producer_run_id == run.run_id
                and obj.metadata.get("path")
            ):
                workspace_versions[obj.metadata["path"]] = obj.artifact_id
        await self.ledger.record(
            run.experiment_id,
            T.ENVIRONMENT_OPERATION,
            [snapshot],
            [(artifact.artifact_id, snapshot.snapshot_id, "snapshot")]
            + [(aid, snapshot.snapshot_id, "captured") for aid in workspace_versions.values()],
            output={"operation": "snapshot", "intent_event_id": intent.event_id},
        )
        return snapshot

    async def start_run(
        self,
        experiment_id,
        *,
        hypothesis_ids=None,
        config=None,
        seed=0,
        snapshot_id=None,
        parent_run_id=None,
        parent_step_id=None,
        environment_spec=None,
        code_version=None,
        purpose="experiment",
    ):
        experiment = await self.ledger.get(experiment_id, "Experiment")
        hypotheses = hypothesis_ids if hypothesis_ids is not None else experiment.hypothesis_ids
        for hypothesis_id in hypotheses:
            await self._scoped(hypothesis_id, experiment_id, "Hypothesis")
        initial = (
            await self._scoped(snapshot_id, experiment_id, "Snapshot") if snapshot_id else None
        )
        if parent_run_id:
            await self._scoped(parent_run_id, experiment_id, "Run")
        if parent_step_id:
            await self._scoped(parent_step_id, experiment_id, "TrajectoryStep")
        spec = environment_spec or experiment.environment_spec
        if initial:
            spec = EnvironmentSpec.model_validate(
                initial.environment.get("spec", spec.model_dump(mode="json"))
            )
        execution_metadata = environment_metadata()
        run = Run(
            purpose=purpose,
            experiment_id=experiment_id,
            hypothesis_ids=hypotheses,
            environment_snapshot_id=snapshot_id,
            environment_spec=spec,
            code_version=code_version or execution_metadata["source_sha256"],
            config=config or {},
            seed=seed,
            resource_spec={"cpu": spec.cpu, "memory_gb": spec.memory_gb},
            parent_run_id=parent_run_id,
            parent_step_id=parent_step_id,
        )
        trajectory = Trajectory(
            purpose=purpose,
            trajectory_id=run.trajectory_id,
            experiment_id=experiment_id,
            run_id=run.run_id,
            agent_backend=experiment.agent_backend,
            steps=[],
            final_status="RUNNING",
        )
        links = [
            (experiment_id, run.run_id, "run"),
            (run.run_id, trajectory.trajectory_id, "trajectory"),
        ] + [(h, run.run_id, "tests") for h in hypotheses]
        if parent_run_id:
            links.append((parent_run_id, run.run_id, "replay_or_branch"))
        if snapshot_id:
            links.append((snapshot_id, run.run_id, "restored_from"))
        await self.ledger.record(
            experiment_id,
            T.RUN_STARTED,
            [
                run,
                trajectory,
                experiment.model_copy(update={"status": "RUNNING", "finished_at": None}),
            ],
            links,
            output={
                "execution_metadata": execution_metadata,
                "requested_code_version": code_version,
            },
        )
        intent = await self.ledger.record(
            experiment_id,
            T.ENVIRONMENT_OPERATION,
            input={
                "operation": "create",
                "run_id": run.run_id,
                "spec": spec.model_dump(mode="json"),
            },
            status="PENDING",
        )
        try:
            handle = await self.environment.create(spec)
            run = run.model_copy(update={"environment_handle": handle})
            await self.ledger.record(
                experiment_id,
                T.ENVIRONMENT_OPERATION,
                [run],
                output={"operation": "create", "intent_event_id": intent.event_id},
            )
            if initial:
                await self.ledger.record(
                    experiment_id,
                    T.ENVIRONMENT_OPERATION,
                    input={
                        "operation": "restore",
                        "snapshot_id": snapshot_id,
                        "run_id": run.run_id,
                    },
                    status="PENDING",
                )
                await self.environment.restore(
                    handle, await self.artifact_bytes(initial.artifact_id)
                )
                await self.ledger.record(
                    experiment_id,
                    T.ENVIRONMENT_OPERATION,
                    output={
                        "operation": "restore",
                        "snapshot_id": snapshot_id,
                        "run_id": run.run_id,
                    },
                )
            observed = await self.environment.observe(handle)
            pinned = EnvironmentSpec.model_validate(observed["spec"])
            if pinned.backend != spec.backend:
                raise ValueError("Environment backend does not match the requested spec")
            if (
                initial
                and initial.environment.get("fingerprint")
                and observed["fingerprint"] != initial.environment["fingerprint"]
            ):
                raise ValueError("Restored environment fingerprint differs from the original")
            if spec.dependencies and any(
                pinned.dependencies.get(k) != v for k, v in spec.dependencies.items()
            ):
                raise ValueError("Environment dependency mismatch")
            run = run.model_copy(update={"environment_spec": pinned})
            previous_config = await self._previous_file(run, "config.json")
            config_artifact = await self._artifact(
                run,
                json.dumps(run.config, sort_keys=True).encode(),
                "config",
                parents=[previous_config.artifact_id] if previous_config else [],
                mime_type="application/json",
                metadata={"path": "config.json"},
            )
            await self.ledger.record(
                experiment_id,
                T.ENVIRONMENT_OPERATION,
                input={
                    "operation": "upload",
                    "artifact_id": config_artifact.artifact_id,
                    "run_id": run.run_id,
                },
                status="PENDING",
            )
            await self.environment.upload(
                handle, "config.json", await self.artifact_bytes(config_artifact.artifact_id)
            )
            await self.ledger.record(
                experiment_id,
                T.ENVIRONMENT_OPERATION,
                output={
                    "operation": "upload",
                    "artifact_id": config_artifact.artifact_id,
                    "run_id": run.run_id,
                },
            )
            snapshot = await self._snapshot(run)
            run = run.model_copy(update={"environment_snapshot_id": snapshot.snapshot_id})
            await self.ledger.record(experiment_id, T.OBJECT_RECORDED, [run])
            return run
        except Exception as exc:
            if run.environment_handle:
                await self.ledger.record(
                    experiment_id,
                    T.ENVIRONMENT_OPERATION,
                    input={"operation": "destroy", "run_id": run.run_id},
                    status="PENDING",
                )
                await self.environment.destroy(run.environment_handle)
            await self.ledger.record(
                experiment_id,
                T.RUN_FINISHED,
                [
                    run.model_copy(
                        update={
                            "status": "FAILED",
                            "finished_at": now(),
                            "environment_handle": None,
                        }
                    )
                ],
                output={"error": str(exc)},
                status="FAILED",
            )
            raise

    async def get_state(self, experiment_id, run_id=None):
        objects = await self.ledger.objects(experiment_id)
        experiment = objects[experiment_id]
        run = objects.get(run_id) if run_id else None
        if run_id and not isinstance(run, Run):
            raise KeyError(run_id)
        steps = [
            x
            for x in objects.values()
            if isinstance(x, TrajectoryStep) and run and x.trajectory_id == run.trajectory_id
        ]
        artifacts = [
            x
            for x in objects.values()
            if isinstance(x, Artifact) and run and x.producer_run_id == run_id
        ]
        snapshot_id = (
            (steps[-1].state_after["snapshot_id"] if steps else run.environment_snapshot_id)
            if run
            else None
        )
        return {
            "schema_version": "2.0",
            "experiment_id": experiment_id,
            "goal": experiment.goal,
            "success_criteria": experiment.success_criteria,
            "criteria": [c.model_dump(mode="json") for c in experiment.criteria],
            "run_id": run_id,
            "config": run.config if run else {},
            "seed": run.seed if run else None,
            "environment": run.environment_spec.model_dump(mode="json")
            if run
            else experiment.environment_spec.model_dump(mode="json"),
            "code_version": run.code_version if run else None,
            "snapshot_id": snapshot_id,
            "artifact_ids": [a.artifact_id for a in artifacts],
            "hypothesis_ids": run.hypothesis_ids if run else experiment.hypothesis_ids,
            "last_step_id": steps[-1].step_id if steps else (run.parent_step_id if run else None),
        }

    async def _validate_decision_source(self, ref, experiment_id, run_id):
        """Explicit ownership rules; an absent run_id never implies current-run scope."""
        from experiment_infra.core.schema import CounterfactualLink, VerificationResult

        obj = await self._scoped(ref, experiment_id)
        if isinstance(obj, (Experiment, Hypothesis)):
            return
        if isinstance(obj, Claim):
            for evidence_id in obj.evidence_ids:
                await self._validate_decision_source(evidence_id, experiment_id, run_id)
            return
        if isinstance(obj, ObservationEvent):
            return await self._validate_decision_source(obj.action_event_id, experiment_id, run_id)
        if isinstance(obj, TrajectoryStep):
            return await self._validate_decision_source(obj.action.event_id, experiment_id, run_id)
        if isinstance(obj, CounterfactualLink):
            for step_id in (obj.original_step_id, obj.counterfactual_step_id):
                await self._validate_decision_source(step_id, experiment_id, run_id)
            return
        if isinstance(obj, Evidence):
            owners = set(obj.source_run_ids)
        elif isinstance(obj, Artifact):
            owners = {obj.producer_run_id}
        elif isinstance(
            obj, (Run, ActionEvent, Snapshot, Trajectory, VerificationResult, DecisionContext)
        ):
            owners = {obj.run_id}
        else:
            raise ValueError("Unsupported decision source type")
        if owners != {run_id}:
            raise ValueError("Decision source belongs to another run")

    async def capture_decision_context(
        self, experiment_id, run_id, payload, *, coverage="unknown", source_refs=()
    ):
        """Record the input the client reports delivering before choosing an action.

        The runtime cannot attest to hidden client prompts or actual model consumption.
        Callers must include all prompts/tool text for complete coverage.
        """
        async with self.locks.setdefault(experiment_id, asyncio.Lock()):
            run = await self._scoped(run_id, experiment_id, "Run")
            if run.status != "RUNNING":
                raise ValueError("Run is not executable")
            for ref in source_refs:
                await self._validate_decision_source(ref, experiment_id, run_id)
            state = await self.get_state(experiment_id, run_id)
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            context = DecisionContext(
                experiment_id=experiment_id,
                run_id=run_id,
                payload_json=encoded,
                payload_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                coverage=coverage,
                source_refs=tuple(source_refs),
                last_step_id=state["last_step_id"],
            )
            await self.ledger.record(
                experiment_id,
                T.OBJECT_RECORDED,
                [context],
                [(run_id, context.context_id, "decision_context")]
                + [(ref, context.context_id, "observed_input") for ref in source_refs],
            )
            return context

    async def get_feedback(self, experiment_id, run_id):
        """Deliver existing observations/verification only, never future projections."""
        run = await self._scoped(run_id, experiment_id, "Run")
        objects = await self.ledger.objects(experiment_id)
        steps = [
            x
            for x in objects.values()
            if isinstance(x, TrajectoryStep) and x.trajectory_id == run.trajectory_id
        ]
        from experiment_infra.core.schema import VerificationResult

        return {
            "state": await self.get_state(experiment_id, run_id),
            "observations": [s.observation.model_dump(mode="json") for s in steps],
            "verifier_feedback": [
                x.model_dump(mode="json")
                for x in objects.values()
                if isinstance(x, VerificationResult) and x.run_id == run_id
            ],
        }

    async def execute(self, experiment_id, action: Action, *, run_id, token_usage=None):
        lock = self.locks.setdefault(experiment_id, asyncio.Lock())
        async with lock:
            return await self._execute(
                experiment_id, Action.model_validate(action), run_id, token_usage
            )

    async def _execute(self, experiment_id, action, run_id, token_usage=None):
        run = await self._scoped(run_id, experiment_id, "Run")
        if run.status != "RUNNING" or not run.environment_handle:
            raise ValueError("Run is not executable")
        experiment = await self.ledger.get(experiment_id, "Experiment")
        for ref in action.context_refs:
            await self._scoped(ref, experiment_id)
        before = await self.get_state(experiment_id, run_id)
        context = None
        if action.decision_context_id:
            context = await self._scoped(
                action.decision_context_id, experiment_id, "DecisionContext"
            )
            if context.run_id != run_id:
                raise ValueError("Decision context belongs to another run")
            if context.last_step_id != before["last_step_id"]:
                raise ValueError("Decision context is stale")
        step_id = uid()
        request = ActionEvent(
            experiment_id=experiment_id,
            run_id=run_id,
            step_id=step_id,
            agent_backend=experiment.agent_backend,
            **action.model_dump(exclude={"schema_version"}),
        )
        links = [(run_id, request.event_id, "action")] + [
            (ref, request.event_id, "context") for ref in action.context_refs
        ]
        if context:
            links.append((context.context_id, request.event_id, "decision_input"))
        if before["snapshot_id"]:
            links.append((before["snapshot_id"], request.event_id, "state_before"))
        with span(
            {"environment": "SANDBOX", "mcp": "TOOL", "rag": "RETRIEVER"}[action.tool],
            action.operation,
            experiment_id,
        ):
            await self.ledger.record(
                experiment_id, T.ACTION_REQUESTED, [request], links, status="PENDING"
            )
            started = perf_counter()
            observation_id = uid()
            created = []
            input_ids = []
            output = {}
            exit_code = None
            stdout = stderr = ""
            try:
                if action.tool == "environment" and action.operation == "exec":
                    command = Command.model_validate(action.arguments["command"])
                    if command.seed != run.seed:
                        raise ValueError("Action seed must equal the run seed")
                    for path, content in command.files.items():
                        previous = await self._previous_file(run, path)
                        artifact = await self._artifact(
                            run,
                            content.encode(),
                            "source_code" if path.endswith(".py") else "config",
                            step_id=step_id,
                            parents=[previous.artifact_id] if previous else [],
                            source_id=request.event_id,
                            mime_type=mimetypes.guess_type(path)[0] or "text/plain",
                            metadata={"path": path, "role": "input"},
                        )
                        created.append(artifact.artifact_id)
                        input_ids.append(artifact.artifact_id)
                        await self.environment.upload(
                            run.environment_handle, path, content.encode()
                        )
                    result = await self.environment.execute(run.environment_handle, command)
                    stdout, stderr, exit_code = result.stdout, result.stderr, result.exit_code
                    output = {
                        "command": command.model_dump(mode="json"),
                        "execution": result.model_dump(mode="json"),
                        "collected": {},
                    }
                    if exit_code == 0:
                        for path in command.collect:
                            previous = await self._previous_file(run, path)
                            artifact = await self._artifact(
                                run,
                                await self.environment.download(run.environment_handle, path),
                                action.arguments.get("artifact_types", {}).get(
                                    path, "generated_file"
                                ),
                                step_id=step_id,
                                parents=[previous.artifact_id] if previous else [],
                                source_id=request.event_id,
                                mime_type=mimetypes.guess_type(path)[0]
                                or "application/octet-stream",
                                metadata={
                                    "path": path,
                                    "observation_id": observation_id,
                                    "role": "output",
                                },
                            )
                            created.append(artifact.artifact_id)
                            output["collected"][path] = artifact.artifact_id
                elif action.tool == "mcp" and action.operation == "call":
                    if self.gateway is None:
                        raise ValueError("No MCP integration configured")
                    output = await self.gateway.call(
                        action.arguments["tool_name"], action.arguments.get("arguments", {})
                    )
                    exit_code = 0
                elif action.tool == "rag" and action.operation == "retrieve":
                    if self.retriever is None:
                        raise ValueError("No retrieval integration configured")
                    retrieved = await self.retriever.retrieve(
                        action.arguments["query"], action.arguments.get("top_k", 5)
                    )
                    output = {"context": [x.model_dump(mode="json") for x in retrieved]}
                    exit_code = 0
                else:
                    raise ValueError("Unsupported tool operation")
            except Exception as exc:
                exit_code = -1
                stderr = str(exc)
                output = {**output, "error_type": type(exc).__name__, "error": str(exc)}
            log = await self._artifact(
                run,
                json.dumps(
                    {
                        "argv": action.arguments.get("command", {}).get("argv"),
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": exit_code,
                        "structured_output": output,
                    },
                    allow_nan=False,
                ).encode(),
                "log",
                step_id=step_id,
                source_id=request.event_id,
                mime_type="application/json",
                metadata={"origin": "runtime.observation", "observation_id": observation_id},
            )
            created.append(log.artifact_id)
            observation = ObservationEvent(
                event_id=observation_id,
                action_event_id=request.event_id,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                structured_output={**output, "log_artifact_id": log.artifact_id},
                created_artifacts=created,
            )
            await self.ledger.record(
                experiment_id,
                T.OBSERVATION_RECORDED,
                [observation],
                [(request.event_id, observation_id, "observation")]
                + [(observation_id, a, "output") for a in created if a not in input_ids]
                + [(a, observation_id, "input") for a in input_ids],
                status="SUCCESS" if exit_code == 0 else "FAILED",
            )
            snapshot_id = None
            snapshot_error = None
            try:
                after_snapshot = await self._snapshot(
                    run, step_id=step_id, source_id=observation_id
                )
                snapshot_id = after_snapshot.snapshot_id
                if before["snapshot_id"]:
                    previous_snapshot = await self.ledger.get(before["snapshot_id"], "Snapshot")
                    diff = self.environment.diff(
                        await self.artifact_bytes(previous_snapshot.artifact_id),
                        await self.artifact_bytes(after_snapshot.artifact_id),
                    )
                else:
                    diff = {"baseline_unavailable": True}
            except Exception as exc:
                snapshot_error = str(exc)
                diff = {"snapshot_error": snapshot_error}
                await self.ledger.record(
                    experiment_id,
                    T.ENVIRONMENT_OPERATION,
                    output={
                        "operation": "snapshot",
                        "run_id": run_id,
                        "step_id": step_id,
                        "error": snapshot_error,
                    },
                    status="FAILED",
                )
            observation = observation.model_copy(update={"environment_diff": diff})
            after = await self.get_state(experiment_id, run_id)
            after.update(snapshot_id=snapshot_id, last_step_id=step_id)
            if snapshot_error is not None:
                after["snapshot_error"] = snapshot_error
            step = TrajectoryStep(
                decision_context=context,
                trajectory_id=run.trajectory_id,
                step_id=step_id,
                state_before=before,
                context_refs=action.context_refs,
                action=request,
                observation=observation,
                artifact_ids=created,
                state_after=after,
                latency=(perf_counter() - started) * 1000,
                token_usage=token_usage,
                parent_step_id=before["last_step_id"],
                branch_id=run_id,
            )
            await self.ledger.record(
                experiment_id,
                T.TRAJECTORY_RECORDED,
                [observation, step],
                [
                    (run.trajectory_id, step_id, "step"),
                    (observation_id, step_id, "observed_step"),
                ]
                + ([(snapshot_id, step_id, "state_after")] if snapshot_id else []),
            )
            return step

    async def _previous_file(self, run, path):
        objects = await self.ledger.objects(run.experiment_id)
        # Only this run's produced versions; a branch's inherited bytes live in its initial snapshot.
        candidates = [
            o
            for o in objects.values()
            if isinstance(o, Artifact)
            and o.producer_run_id == run.run_id
            and o.metadata.get("path") == path
        ]
        if candidates:
            return candidates[-1]
        if run.parent_step_id:
            source_step = objects[run.parent_step_id]
            inherited = [
                objects[aid]
                for aid in source_step.state_after["artifact_ids"]
                if aid in objects
                and isinstance(objects[aid], Artifact)
                and objects[aid].metadata.get("path") == path
            ]
            return inherited[-1] if inherited else None
        return None

    async def verify(
        self, experiment_id, run_id, hypothesis_id, verifier, artifact_ids, criterion_id=None
    ):
        from types import MappingProxyType

        from experiment_infra.core.schema import VerificationResult
        from experiment_infra.verifiers.policy import (
            EvaluationContext,
            HostEvaluator,
            meets_trust,
            policy_digest,
            resolve_criterion,
            verifier_trust,
        )

        run = await self._scoped(run_id, experiment_id, "Run")
        experiment = await self.ledger.get(experiment_id, "Experiment")
        hypothesis = await self._scoped(hypothesis_id, experiment_id, "Hypothesis")
        if hypothesis_id not in run.hypothesis_ids:
            raise ValueError("Run does not test this hypothesis")
        criterion = resolve_criterion(experiment, hypothesis, verifier, criterion_id)
        if not artifact_ids or len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("Verification requires unique artifact inputs")
        intent = await self.ledger.record(
            experiment_id,
            T.VERIFICATION_STARTED,
            input={
                "run_id": run_id,
                "hypothesis_id": hypothesis_id,
                "verifier": verifier,
                "artifact_ids": artifact_ids,
                "criterion_id": criterion.criterion_id if criterion else None,
                "policy_sha256": policy_digest(criterion) if criterion else None,
            },
            status="PENDING",
        )
        with span("VERIFIER", verifier, experiment_id):
            try:
                plugin = self.verifiers[verifier]
                inputs = {}
                for aid in artifact_ids:
                    artifact = await self._scoped(aid, experiment_id, "Artifact")
                    if artifact.producer_run_id != run_id:
                        raise ValueError("Verifier inputs must be produced by the selected run")
                    if artifact.metadata.get("observation_id"):
                        observation = await self._scoped(
                            artifact.metadata["observation_id"], experiment_id, "ObservationEvent"
                        )
                        action = await self._scoped(
                            observation.action_event_id, experiment_id, "ActionEvent"
                        )
                        if (
                            aid not in observation.created_artifacts
                            or action.run_id != run_id
                            or action.step_id != artifact.producer_step_id
                        ):
                            raise ValueError("Artifact has no matching producing observation")
                    inputs[aid] = (artifact, await self.artifact_bytes(aid))
                if isinstance(plugin, HostEvaluator):
                    if criterion is None:
                        raise ValueError("Host evaluators require a structured criterion")
                    evaluation = await plugin.evaluate(
                        EvaluationContext(
                            experiment=experiment,
                            hypothesis=hypothesis,
                            run=run,
                            criterion=criterion,
                            artifacts=MappingProxyType(inputs),
                        )
                    )
                    result = VerificationResult(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        verifier=verifier,
                        source_artifact_ids=list(inputs),
                        source_observation_ids=sorted(
                            {
                                artifact.metadata["observation_id"]
                                for artifact, _ in inputs.values()
                                if artifact.metadata.get("observation_id")
                            }
                        ),
                        **evaluation.model_dump(),
                    )
                else:
                    selected_experiment = experiment
                    if criterion:
                        selected_experiment = experiment.model_copy(
                            update={
                                "success_criteria": {
                                    **experiment.success_criteria,
                                    verifier: criterion.policy.parameters,
                                }
                            }
                        )
                    claims = [
                        x
                        for x in (await self.ledger.objects(experiment_id)).values()
                        if isinstance(x, Claim)
                        and (not criterion or x.hypothesis_id == hypothesis_id)
                    ]
                    result = await plugin.verify(selected_experiment, run, inputs, claims)
                    result = VerificationResult.model_validate(result.model_dump())
                if (
                    result.experiment_id != experiment_id
                    or result.run_id != run_id
                    or result.verifier != verifier
                    or not result.source_artifact_ids
                    or len(set(result.source_artifact_ids)) != len(result.source_artifact_ids)
                    or not set(result.source_artifact_ids).issubset(inputs)
                ):
                    raise ValueError("Verifier result has mismatched scope or artifact inputs")
                observed = {
                    inputs[aid][0].metadata["observation_id"]
                    for aid in result.source_artifact_ids
                    if inputs[aid][0].metadata.get("observation_id")
                }
                if set(result.source_observation_ids) != observed:
                    raise ValueError("Verifier result has mismatched source observations")
                trust = verifier_trust(plugin)
                details = {
                    **result.details,
                    "assessment_scope": "criterion_bound" if criterion else "unbound_legacy",
                    "trust_boundary": {
                        "agent_reported": "Checks agent-supplied values; no independent measurement",
                        "runtime_observed": "Observed command execution; tests may be agent-authored",
                        "host_evaluated": "Evaluation performed by host-registered trusted code",
                    }[trust],
                }
                status = result.status
                if criterion:
                    if type(plugin) is TestVerifier and status == "FAILED":
                        # A process failure is not a negative scientific measurement.
                        # Only a host evaluator can interpret failure semantically.
                        details["execution_failure"] = True
                        status = "ERROR"
                    details["criteria"] = criterion.policy.parameters
                    details["trust_requirement"] = criterion.policy.trust_requirement
                    if not meets_trust(trust, criterion.policy.trust_requirement):
                        details["untrusted_verifier_status"] = result.status
                        details["trust_requirement_met"] = False
                        status = "INCONCLUSIVE"
                    else:
                        details["trust_requirement_met"] = True
                result = result.model_copy(
                    update={
                        "hypothesis_id": hypothesis_id,
                        "criterion_id": criterion.criterion_id if criterion else None,
                        "policy_id": criterion.policy.policy_id if criterion else None,
                        "policy_sha256": policy_digest(criterion) if criterion else None,
                        "trust_level": trust,
                        "status": status,
                        "details": details,
                    }
                )
            except Exception as exc:
                await self.ledger.record(
                    experiment_id,
                    T.VERIFICATION_RESULT,
                    output={"intent_event_id": intent.event_id, "error": str(exc)},
                    status="FAILED",
                )
                raise
            evidence = Evidence(
                experiment_id=experiment_id,
                hypothesis_id=hypothesis_id,
                criterion_id=result.criterion_id,
                policy_id=result.policy_id,
                policy_sha256=result.policy_sha256,
                trust_level=result.trust_level,
                source_artifact_ids=result.source_artifact_ids,
                source_run_ids=[run_id],
                source_observation_ids=result.source_observation_ids,
                metric=result.metric,
                value=result.value,
                uncertainty=result.uncertainty,
                verifier=result.verifier,
                verification_id=result.verification_id,
                verification_status=result.status,
                metadata={
                    "criteria": result.details.get("criteria", {}),
                    "assessment_scope": result.details["assessment_scope"],
                    "trust_boundary": result.details["trust_boundary"],
                },
            )
            links = [
                (a, result.verification_id, "verified_input") for a in result.source_artifact_ids
            ] + [
                (result.verification_id, evidence.evidence_id, "evidence"),
                (run_id, evidence.evidence_id, "run_evidence"),
                (hypothesis_id, evidence.evidence_id, "hypothesis_evidence"),
            ]
            links += [
                (oid, evidence.evidence_id, "observed_evidence")
                for oid in result.source_observation_ids
            ]
            await self.ledger.record(
                experiment_id,
                T.EVIDENCE_GENERATED,
                [result, evidence, hypothesis.model_copy(update={"status": "TESTED"})],
                links,
                output={"intent_event_id": intent.event_id},
            )
            return evidence

    async def create_claim(
        self,
        experiment_id,
        statement,
        evidence_ids=(),
        confidence=None,
        hypothesis_id=None,
        criterion_ids=None,
    ):
        from experiment_infra.verifiers.policy import criteria_by_id, meets_trust, policy_digest

        experiment = await self.ledger.get(experiment_id, "Experiment")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Claim evidence IDs must be unique")
        evidence = [await self._scoped(e, experiment_id, "Evidence") for e in evidence_ids]
        available = criteria_by_id(experiment)
        bound = bool(
            available or hypothesis_id or criterion_ids or any(e.criterion_id for e in evidence)
        )
        selected = []
        hypothesis = None
        if bound:
            if hypothesis_id is None:
                targets = {item.hypothesis_id for item in evidence}
                if len(targets) != 1:
                    raise ValueError("A structured claim requires one hypothesis_id")
                hypothesis_id = next(iter(targets))
            hypothesis = await self._scoped(hypothesis_id, experiment_id, "Hypothesis")
            if statement.strip() != hypothesis.statement.strip():
                raise ValueError("A bound claim statement must match the registered hypothesis")
            selected = list(criterion_ids or hypothesis.criterion_ids)
            if (
                not selected
                or len(set(selected)) != len(selected)
                or set(selected) != set(hypothesis.criterion_ids)
                or not set(selected).issubset(available)
            ):
                raise ValueError("A bound claim must assess all declared hypothesis criteria")
        for item in evidence:
            verification = await self._scoped(
                item.verification_id, experiment_id, "VerificationResult"
            )
            if (
                verification.status != item.verification_status
                or verification.source_artifact_ids != item.source_artifact_ids
                or verification.source_observation_ids != item.source_observation_ids
                or [verification.run_id] != item.source_run_ids
                or verification.metric != item.metric
                or verification.value != item.value
                or verification.uncertainty != item.uncertainty
                or verification.verifier != item.verifier
                or verification.criterion_id != item.criterion_id
                or verification.policy_id != item.policy_id
                or verification.policy_sha256 != item.policy_sha256
                or verification.trust_level != item.trust_level
                or (
                    verification.hypothesis_id is not None
                    and verification.hypothesis_id != item.hypothesis_id
                )
            ):
                raise ValueError("Evidence verification mismatch")
            if bound:
                if item.hypothesis_id != hypothesis_id or item.criterion_id not in selected:
                    raise ValueError("Evidence is unrelated to the claim hypothesis or criteria")
                criterion = available[item.criterion_id]
                if (
                    verification.hypothesis_id != hypothesis_id
                    or item.policy_id != criterion.policy.policy_id
                    or item.policy_sha256 != policy_digest(criterion)
                    or item.verifier != criterion.policy.verifier
                ):
                    raise ValueError("Evidence does not match the declared verifier policy")
                if not meets_trust(
                    item.trust_level, criterion.policy.trust_requirement
                ) and item.verification_status in {"PASSED", "FAILED"}:
                    raise ValueError("Evidence does not meet the criterion trust requirement")
            for artifact_id in item.source_artifact_ids:
                await self.artifact_bytes(artifact_id)
        statuses = {e.verification_status for e in evidence}
        status = (
            "UNVERIFIED"
            if not evidence
            else "SUPPORTED"
            if statuses == {"PASSED"}
            else "REFUTED"
            if statuses == {"FAILED"}
            else "INCONCLUSIVE"
        )
        if bound and evidence:
            coverage = {item.criterion_id for item in evidence}
            if coverage != set(selected):
                status = "INCONCLUSIVE"
        claim = Claim(
            experiment_id=experiment_id,
            statement=statement,
            evidence_ids=list(evidence_ids),
            confidence=confidence,
            status=status,
            hypothesis_id=hypothesis_id,
            criterion_ids=selected,
            assessment_scope="criterion_bound" if bound else "unbound_legacy",
        )
        links = [(e, claim.claim_id, "supports_or_refutes") for e in evidence_ids]
        links.append((experiment_id, claim.claim_id, "claim"))
        if hypothesis:
            links.append((hypothesis_id, claim.claim_id, "hypothesis_claim"))
        await self.ledger.record(experiment_id, T.CLAIM_ASSESSED, [claim], links)
        return claim

    async def get_trajectory(self, trajectory_id):
        base = await self.ledger.get(trajectory_id, "Trajectory")
        ancestor = await self.ledger.get(base.experiment_id, "Experiment")
        seen = set()
        while ancestor.parent_experiment_id:
            if ancestor.experiment_id in seen:
                raise ValueError("Experiment lineage cycle")
            seen.add(ancestor.experiment_id)
            ancestor = await self.ledger.get(ancestor.parent_experiment_id, "Experiment")
        base = base.model_copy(update={"lineage_root_experiment_id": ancestor.experiment_id})
        objects = await self.ledger.objects(base.experiment_id)
        run = objects[base.run_id]
        evidence = [
            x
            for x in objects.values()
            if isinstance(x, Evidence) and run.run_id in x.source_run_ids
        ]
        steps = []
        for obj in objects.values():
            if isinstance(obj, TrajectoryStep) and obj.trajectory_id == trajectory_id:
                relevant = [
                    e for e in evidence if obj.observation.event_id in e.source_observation_ids
                ]
                steps.append(
                    obj.model_copy(
                        update={
                            "evidence_ids": [e.evidence_id for e in relevant],
                            "verifier_feedback": [objects[e.verification_id] for e in relevant],
                        }
                    )
                )
        claims = [
            x
            for x in objects.values()
            if isinstance(x, Claim) and set(x.evidence_ids) & {e.evidence_id for e in evidence}
        ]
        return base.model_copy(
            update={
                "steps": steps,
                "final_status": run.status,
                "final_claims": claims,
                "total_cost": sum(s.monetary_cost or 0 for s in steps)
                if any(s.monetary_cost is not None for s in steps)
                else None,
            }
        )

    async def finish_run(self, run_id):
        run = await self.ledger.get(run_id, "Run")
        if run.status != "RUNNING":
            return run
        trajectory = await self.get_trajectory(run.trajectory_id)
        await self.ledger.record(
            run.experiment_id,
            T.ENVIRONMENT_OPERATION,
            input={"operation": "destroy", "run_id": run_id},
            status="PENDING",
        )
        await self.environment.destroy(run.environment_handle)
        status = (
            "FAILED"
            if any(
                s.observation.exit_code != 0 or "snapshot_error" in s.state_after
                for s in trajectory.steps
            )
            else "COMPLETED"
        )
        run = run.model_copy(
            update={"environment_handle": None, "status": status, "finished_at": now()}
        )
        await self.ledger.record(run.experiment_id, T.RUN_FINISHED, [run])
        return run

    async def finish_experiment(self, experiment_id):
        objects = await self.ledger.objects(experiment_id)
        runs = [x for x in objects.values() if isinstance(x, Run)]
        if any(r.status == "RUNNING" for r in runs):
            raise ValueError("Finish all runs first")
        experiment = objects[experiment_id].model_copy(
            update={
                "status": "COMPLETED"
                if runs and any(r.status == "COMPLETED" for r in runs)
                else "FAILED",
                "finished_at": now(),
            }
        )
        await self.ledger.record(experiment_id, T.EXPERIMENT_UPDATED, [experiment])
        return experiment

    async def replay(self, run_id):
        from experiment_infra.runtime.replay import exact_replay

        return await exact_replay(self, run_id)

    async def branch(self, run_id, from_step, *, config=None):
        from experiment_infra.runtime.replay import branch_replay

        return await branch_replay(self, run_id, from_step, config=config)

    async def get_ancestors(self, object_id):
        return await self.ledger.get_ancestors(object_id)

    async def get_descendants(self, object_id):
        return await self.ledger.get_descendants(object_id)

    async def get_provenance(self, object_id):
        return await self.ledger.get_provenance(object_id)

    async def export(self, experiment_id, directory, *, policy=None):
        from experiment_infra.exporters.canonical import export_experiment

        return await export_experiment(self, experiment_id, directory, policy=policy)
