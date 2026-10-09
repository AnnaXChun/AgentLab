# Criterion-bound evidence and the evaluator trust boundary

Existing `success_criteria` and free-text claims continue to work. They are explicitly
marked `assessment_scope: unbound_legacy`; `SUPPORTED` in that legacy path remains
an aggregation of submitted evidence statuses, not proof that arbitrary prose is true.
No historical records are rewritten or promoted to trusted evaluation.

New experiments can declare typed `criteria`. Every criterion has a stable
`criterion_id`, a verifier policy ID, verifier name, parameters, and trust requirement.
Hypotheses explicitly name their required criterion IDs. Verification records the
hypothesis, criterion, policy ID and SHA-256 of the canonical criterion/policy document.
Claims bind to one hypothesis, use its exact statement, and assess all its declared
criteria. This is structural binding to a previously declared assertion, not automated
understanding or proof of natural-language scientific validity.

```python
from experiment_infra.verifiers.policy import SuccessCriterion, VerifierPolicy

criterion = SuccessCriterion(
    criterion_id="reported-score",
    description="The submitted score exceeds the declared threshold",
    policy=VerifierPolicy(
        policy_id="reported-score-v1",
        verifier="metric",
        parameters={"field": "score", "operator": "gt", "baseline": 0.5},
        trust_requirement="agent_reported",  # Explicitly accepts submitted numbers.
    ),
)
experiment = await runtime.create_experiment("Compare candidates", criteria=[criterion])
hypothesis = await runtime.register_hypothesis(
    experiment.experiment_id,
    "The submitted score exceeds 0.5",
    criterion_ids=["reported-score"],
)
# Execute a run and collect an immutable typed metrics artifact first.
evidence = await runtime.verify(
    experiment.experiment_id,
    run.run_id,
    hypothesis.hypothesis_id,
    "metric",
    [metrics_artifact_id],
    criterion_id="reported-score",
)
claim = await runtime.create_claim(
    experiment.experiment_id,
    hypothesis.statement,
    [evidence.evidence_id],
    hypothesis_id=hypothesis.hypothesis_id,
)
```

If several criteria use the same verifier, `criterion_id` must disambiguate the
verification. A claim may not drop one of its hypothesis's required criteria. Missing
criterion evidence yields `INCONCLUSIVE`; unrelated hypotheses, criteria or changed
policy digests are rejected. An empty evidence list yields `UNVERIFIED`. Contradictory,
error, or inconclusive evidence does not become support or refutation.

## Three different trust levels

- `agent_reported`: `MetricVerifier` validates and compares JSON numbers. It does not
  independently measure anything. Artifact hashes and runtime provenance do not make
  the content truthful.
- `runtime_observed`: `TestVerifier` verifies that the declared command ran and checks
  its exit code. Agent-authored tests can be trivial or misleading. For criterion-bound
  evaluation, nonzero exit is `ERROR`, not scientific refutation.
- `host_evaluated`: only a host-created `HostEvaluator` can assign this level. Neither
  HTTP request fields, artifact metadata, nor a declared trust requirement grants it.

The default policy requirement is `host_evaluated`. A lower-trust verifier still runs,
but its result becomes `INCONCLUSIVE` when it cannot meet the requirement. The original
comparison status and unmet requirement remain visible in verification details.

## Host-injected evaluator

A host must register trusted evaluators through the Python constructor or registry.
There is deliberately no public HTTP endpoint for installing evaluator code. Do not run
agent-supplied Python in the host process. Keep evaluator logic and fixtures outside the
agent-controlled workspace, or supply your own isolated evaluation service.

```python
import json
from experiment_infra.runtime.executor import ExperimentRuntime
from experiment_infra.verifiers.policy import HostEvaluation, HostEvaluator


async def sum_oracle(context):
    # Illustrative host-owned oracle: ignore any candidate-reported score.
    _, data = next(iter(context.artifacts.values()))
    candidate = json.loads(data)
    correct = candidate["answer"] == sum((2, 3, 5))
    return HostEvaluation(
        status="PASSED" if correct else "FAILED",
        metric="answer_correct",
        value=correct,
    )


runtime = ExperimentRuntime(
    storage_runtime,
    environment=environment,
    trusted_evaluators={"sum-oracle": HostEvaluator(sum_oracle)},
)
criterion = SuccessCriterion(
    criterion_id="correct-answer",
    policy=VerifierPolicy(policy_id="sum-oracle-v1", verifier="sum-oracle"),
)
```

The callback receives the declared experiment, hypothesis, run, criterion, and verified
CAS bytes for the selected artifacts. The runtime assigns scope, artifact/observation
provenance, policy binding and trust level itself. A legitimate negative evaluation may
return `FAILED` and support a `REFUTED` claim. A callback exception records a failed
verification attempt, not a scientific negative measurement.

This adapter makes the host's trust decision explicit. It is not remote attestation, a
security sandbox, a statistical inference engine, or a guarantee that host code is
correct. A badly written trusted evaluator can still certify false results. Use a new
policy ID when evaluator logic/fixtures change; include fixture or implementation
version hashes in policy parameters when reproducibility requires them.
