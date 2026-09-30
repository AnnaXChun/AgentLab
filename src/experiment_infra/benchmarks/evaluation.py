"""Native evaluators are invoked by the driver, never selected by the model."""

import hashlib
import json
import math

from experiment_infra.core.schema import Action, VerificationResult


class NativeEvaluationError(RuntimeError):
    def __init__(self, step):
        super().__init__("Official evaluator failed; see the recorded observation")
        self.step = step
        self.timed_out = step.observation.structured_output.get("error_type") == "TimeoutError"


class EvaluationGateway:
    def __init__(self, callback):
        self.callback = callback
        self.result = None

    async def call(self, name, arguments):
        if name != "benchmark.native_evaluate" or arguments:
            raise ValueError("Invalid evaluator invocation")
        self.result = await self.callback()
        return self.result


class NativeResultVerifier:
    def __init__(self, benchmark, expected_hash):
        self.benchmark, self.expected_hash = benchmark, expected_hash

    async def verify(self, experiment, run, artifacts, claims):
        if len(artifacts) != 1:
            raise ValueError("Expected one runtime-produced evaluation log")
        artifact, data = next(iter(artifacts.values()))
        if hashlib.sha256(data).hexdigest() != self.expected_hash:
            raise ValueError("Native evaluation receipt mismatch")
        log = json.loads(data)
        if log["exit_code"] != 0:
            raise ValueError("Official evaluator did not complete")
        result = log["structured_output"]
        score = result["score"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid native score")
        if result["benchmark"] != self.benchmark or not isinstance(result["success"], bool):
            raise ValueError("Invalid native result identity")
        if result["success"] != (score == 1):
            raise ValueError("Native success must equal full task success")
        return VerificationResult(
            experiment_id=experiment.experiment_id,
            run_id=run.run_id,
            verifier="native/" + self.benchmark,
            status="PASSED" if result["success"] else "FAILED",
            source_artifact_ids=[artifact.artifact_id],
            source_observation_ids=[artifact.metadata["observation_id"]],
            metric=result["metric"],
            value=score,
            details={
                "criteria": {
                    "official_evaluator": result["evaluator"],
                    "benchmark_version": result["version"],
                }
            },
        )


async def evaluate_and_record(runtime, adapter, task, run, hypothesis_id, termination):
    old_gateway = runtime.gateway
    gateway = EvaluationGateway(lambda: adapter.evaluate(runtime, task, run, termination))
    runtime.gateway = gateway
    try:
        step = await runtime.execute(
            run.experiment_id,
            Action(
                tool="mcp",
                operation="call",
                arguments={"tool_name": "benchmark.native_evaluate", "arguments": {}},
                expected_effect="Record the official benchmark evaluator result",
            ),
            run_id=run.run_id,
        )
    finally:
        runtime.gateway = old_gateway
    if step.observation.exit_code != 0:
        raise NativeEvaluationError(step)
    objects = await runtime.ledger.objects(run.experiment_id)
    logs = [objects[aid] for aid in step.artifact_ids if objects[aid].type == "log"]
    receipt = logs[-1]
    from packages.core.models.schemas import EventType as T

    related = getattr(adapter, "evaluation_artifacts", {}).get(run.run_id, [])
    if related:
        await runtime.ledger.record(
            run.experiment_id,
            T.OBJECT_RECORDED,
            links=[(aid, receipt.artifact_id, "native_evaluation_input") for aid in related],
        )
    runtime.verifiers["benchmark.native"] = NativeResultVerifier(adapter.name, receipt.sha256)
    evidence = await runtime.verify(
        run.experiment_id, run.run_id, hypothesis_id, "benchmark.native", [receipt.artifact_id]
    )
    claim = await runtime.create_claim(
        run.experiment_id, "Task completed successfully", [evidence.evidence_id]
    )
    return gateway.result, evidence, claim, step
