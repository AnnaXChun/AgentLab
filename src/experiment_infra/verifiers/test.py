import json

from experiment_infra.core.schema import VerificationResult


class TestVerifier:
    """Evaluate an independently executed test command's runtime-produced log."""

    __test__ = False

    async def verify(self, experiment, run, artifacts, claims):
        if len(artifacts) != 1:
            raise ValueError("TestVerifier requires one runtime observation log")
        artifact, data = next(iter(artifacts.values()))
        if artifact.type != "log" or artifact.metadata.get("origin") != "runtime.observation":
            raise ValueError("Agent-produced log text cannot certify test execution")
        execution = json.loads(data)
        expected = experiment.success_criteria["test"]["argv"]
        if execution["argv"] != expected:
            raise ValueError("Test command must match the experiment's declared test criterion")
        passed = execution["exit_code"] == 0
        return VerificationResult(
            experiment_id=experiment.experiment_id,
            run_id=run.run_id,
            verifier="test",
            status="PASSED" if passed else "FAILED",
            source_artifact_ids=[artifact.artifact_id],
            source_observation_ids=[artifact.metadata["observation_id"]],
            metric="test_exit_code",
            value=execution["exit_code"],
            details={
                "argv": expected,
                "stdout": execution["stdout"],
                "stderr": execution["stderr"],
            },
        )
