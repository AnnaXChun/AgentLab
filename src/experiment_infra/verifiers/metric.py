import json
import math

from experiment_infra.core.schema import VerificationResult


class MetricVerifier:
    async def verify(self, experiment, run, artifacts, claims):
        if len(artifacts) != 1:
            raise ValueError("MetricVerifier requires one metrics artifact")
        artifact, data = next(iter(artifacts.values()))
        if artifact.type != "metric":
            raise ValueError("MetricVerifier requires a typed metric artifact")
        criteria = experiment.success_criteria["metric"]
        values = json.loads(data)
        metric = criteria["field"]
        observed = values[metric]
        baseline = (
            values[criteria["baseline_field"]]
            if "baseline_field" in criteria
            else criteria["baseline"]
        )
        for value in (observed, baseline):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Metrics must be finite numbers")
        operator = criteria["operator"]
        if operator == "lt":
            passed = observed < baseline
        elif operator == "gt":
            passed = observed > baseline
        else:
            raise ValueError("Unsupported comparison")
        return VerificationResult(
            experiment_id=experiment.experiment_id,
            run_id=run.run_id,
            verifier="metric",
            status="PASSED" if passed else "FAILED",
            source_artifact_ids=[artifact.artifact_id],
            source_observation_ids=[artifact.metadata["observation_id"]],
            metric=metric,
            value=observed,
            details={
                "criteria": criteria,
                "baseline": baseline,
                "artifact_sha256": artifact.sha256,
            },
        )
