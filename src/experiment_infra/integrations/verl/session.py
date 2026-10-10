"""Own the runtime lifecycle; veRL never operates an environment directly."""

import json

from experiment_infra.benchmarks.base import AdapterBase, BenchmarkTask
from experiment_infra.benchmarks.evaluation import evaluate_and_record
from experiment_infra.core.schema import EnvironmentSpec, VerificationResult
from experiment_infra.integrations.verl.reward_bridge import RewardBridge
from experiment_infra.runtime.executor import ExperimentRuntime


class ToyAdapter(AdapterBase):
    name, version = "toy", "sum-files-v1"

    def list_tasks(self, split):
        # Deterministic and disjoint task IDs/inputs for train and validation.
        offset = 0 if split == "train" else 100
        return [
            BenchmarkTask(
                f"sum-{i}",
                "Read inputs.json, then write answer.json containing "
                '{"sum": <sum of the numbers>}. Use at least two actions, then finish.',
                split,
                {"numbers": [i + 1, i + 3]},
            )
            for i in range(offset, offset + 8)
        ]

    def environment_spec(self, task):
        return EnvironmentSpec()

    def available_tools(self, task):
        return {"contract": "environment.exec with arguments.command.argv; environment.finish"}

    def agent_context(self, task):
        return {}

    async def prepare_environment(self, runtime, task, run):
        await runtime.environment.upload(
            run.environment_handle, "inputs.json", json.dumps(task.payload).encode()
        )
        snapshot = await runtime._snapshot(run)
        # get_state finds the latest snapshot; the setup input is an immutable artifact.
        return snapshot

    def validate_action(self, task, action):
        if action.tool != "environment" or action.operation not in {"exec", "finish"}:
            raise ValueError("Only environment.exec and finish are supported")

    async def evaluate(self, runtime, task, run, termination):
        # Read output independently under the driver-owned evaluation gateway.
        # Do not recursively execute while the runtime holds its per-run lock.
        try:
            output = await runtime.environment.download(run.environment_handle, "answer.json")
            await runtime._artifact(run, output, "log", metadata={"origin": "toy.verifier.input"})
            actual = json.loads(output)
        except (ValueError, TypeError, FileNotFoundError):
            actual = None
        success = termination == "agent_stop" and actual == {"sum": sum(task.payload["numbers"])}
        return {
            "benchmark": self.name,
            "version": self.version,
            "metric": "success",
            "score": float(success),
            "success": success,
            "evaluator": "toy/sum-files-v1",
        }


class TerminalFailureVerifier:
    async def verify(self, experiment, run, artifacts, claims):
        if len(artifacts) != 1:
            raise ValueError("Expected one failure receipt")
        artifact, data = next(iter(artifacts.values()))
        receipt = json.loads(data)
        if artifact.metadata.get("origin") != "verl.bridge.failure" or not receipt.get("failure"):
            raise ValueError("Missing bridge-owned failure receipt")
        return VerificationResult(
            experiment_id=experiment.experiment_id,
            run_id=run.run_id,
            verifier="bridge/terminal_failure",
            status="FAILED",
            source_artifact_ids=[artifact.artifact_id],
            metric="task_success",
            value=0,
            details={"failure": receipt["failure"]},
        )


class RuntimeSession:
    def __init__(self, runtime, adapter, task, *, timeout=30, source="ONLINE_RL"):
        self.runtime, self.adapter, self.task = runtime, adapter, task
        self.timeout = timeout
        self.source = source
        self.experiment = self.run = self.hypothesis = None
        self.policy_steps = []

    async def start(self, identity):
        rt = self.runtime
        self.experiment = await rt.create_experiment(
            self.task.goal,
            environment_spec=self.adapter.environment_spec(self.task),
            agent_backend=("verl/" if self.source == "ONLINE_RL" else "scripted/")
            + identity.policy_id,
            success_criteria={
                "native_evaluator": self.adapter.name,
                "version": self.adapter.version,
            },
        )
        self.hypothesis = await rt.register_hypothesis(
            self.experiment.experiment_id, "Task completed"
        )
        try:
            self.run = await rt.start_run(
                self.experiment.experiment_id,
                hypothesis_ids=[self.hypothesis.hypothesis_id],
                trajectory_source=self.source,
                policy_identity=identity,
                config={"policy": identity.model_dump(mode="json")},
            )
            prepared = await self.adapter.prepare_environment(rt, self.task, self.run)
            if prepared is not None:
                from packages.core.models.schemas import EventType as T

                self.run = self.run.model_copy(
                    update={"environment_snapshot_id": prepared.snapshot_id}
                )
                await rt.ledger.record(
                    self.run.experiment_id,
                    T.OBJECT_RECORDED,
                    [self.run],
                    [(prepared.snapshot_id, self.run.trajectory_id, "prepared_initial_state")],
                )
        except Exception:
            # start_run records its run before environment creation; preserve failure provenance.
            objects = await rt.ledger.objects(self.experiment.experiment_id)
            self.run = next((v for v in objects.values() if type(v).__name__ == "Run"), None)
            raise

    async def execute(self, action):
        self.adapter.validate_action(self.task, action)
        if action.operation != "exec" or action.tool != "environment":
            raise ValueError("Only environment.exec may be executed by policy")
        command = dict(action.arguments["command"])
        command["timeout_seconds"] = min(command.get("timeout_seconds", self.timeout), self.timeout)
        action = action.model_copy(update={"arguments": {**action.arguments, "command": command}})
        step = await self.runtime.execute(self.run.experiment_id, action, run_id=self.run.run_id)
        self.policy_steps.append(step.step_id)
        return step

    async def verify(self, failure):
        rt, run = self.runtime, self.run
        if self.adapter.name == "toy" and len(self.policy_steps) < 2 and failure is None:
            failure = "TASK_INCOMPLETE"
        if failure is None:
            try:
                _, evidence, _, _ = await evaluate_and_record(
                    rt, self.adapter, self.task, run, self.hypothesis.hypothesis_id, "agent_stop"
                )
            except Exception:
                failure = "VERIFIER_FAILURE"
        if failure is not None:
            receipt = await rt._artifact(
                run,
                json.dumps({"failure": failure}).encode(),
                "log",
                metadata={"origin": "verl.bridge.failure"},
                mime_type="application/json",
            )
            rt.verifiers["bridge.terminal"] = TerminalFailureVerifier()
            evidence = await rt.verify(
                run.experiment_id,
                run.run_id,
                self.hypothesis.hypothesis_id,
                "bridge.terminal",
                [receipt.artifact_id],
            )
        reward = await RewardBridge().record(rt, self.experiment, run, [evidence])
        return reward, failure

    async def close(self):
        if self.run:
            await self.runtime.finish_run(self.run.run_id)
            await self.runtime.finish_experiment(self.run.experiment_id)
        await self.adapter.cleanup(self.task)


def make_session(extra_info, timeout):
    name = extra_info["benchmark"]
    if name == "toy":
        adapter = ToyAdapter()
        runtime = ExperimentRuntime()
    elif name == "tau2":
        from experiment_infra.benchmarks.run import make_adapter

        adapter = make_adapter(name)
        runtime = ExperimentRuntime(environment=adapter.environment)
    else:
        raise ValueError("V0.4 online bridge supports toy and tau2 only")
    task = adapter.load_task(extra_info["task_id"], extra_info["split"])
    return RuntimeSession(runtime, adapter, task, timeout=timeout)
