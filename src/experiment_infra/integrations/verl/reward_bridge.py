"""Deterministic verified rewards; no optimizer or model judge."""

from experiment_infra.core.schema import RewardResult
from packages.core.models.schemas import EventType as T


class RewardBridge:
    def compute(self, experiment, run, verification_results, *, evidence):
        if run.experiment_id != experiment.experiment_id:
            raise ValueError("Run belongs to another experiment")
        if not verification_results or not evidence:
            raise ValueError("Reward requires verification and evidence")
        by_id = {v.verification_id: v for v in verification_results}
        for e in evidence:
            v = by_id.get(e.verification_id)
            if (
                v is None
                or v.experiment_id != experiment.experiment_id
                or v.run_id != run.run_id
                or e.experiment_id != experiment.experiment_id
                or e.source_run_ids != [run.run_id]
                or e.verification_status != v.status
                or e.value != v.value
                or e.source_artifact_ids != v.source_artifact_ids
                or e.source_observation_ids != v.source_observation_ids
            ):
                raise ValueError("Reward provenance mismatch")
        if {e.verification_id for e in evidence} != set(by_id):
            raise ValueError("Unlinked verification")
        success = float(all(v.status == "PASSED" for v in verification_results))
        signals = {"task_success": success}
        native = [v for v in verification_results if v.verifier.startswith("native/")]
        if len(native) == 1:
            signals["benchmark_score"] = float(native[0].value)
        return RewardResult(
            experiment_id=experiment.experiment_id,
            run_id=run.run_id,
            scalar_reward=success,
            reward_signals=signals,
            verification_ids=list(by_id),
            evidence_ids=[e.evidence_id for e in evidence],
            reward_source="deterministic_verifier/task_success",
            metadata={"verifiers": [v.verifier for v in verification_results]},
        )

    async def record(self, runtime, experiment, run, evidence):
        verified_evidence = [
            await runtime._scoped(e.evidence_id, experiment.experiment_id, "Evidence")
            for e in evidence
        ]
        if verified_evidence != evidence:
            raise ValueError("Evidence differs from immutable ledger")
        results = [
            await runtime._scoped(e.verification_id, experiment.experiment_id, "VerificationResult")
            for e in evidence
        ]
        reward = self.compute(experiment, run, results, evidence=evidence)
        links = [(v.verification_id, reward.reward_id, "verified_reward") for v in results]
        links += [(e.evidence_id, reward.reward_id, "reward_evidence") for e in evidence]
        links += [(run.run_id, reward.reward_id, "rollout_reward")]
        await runtime.ledger.record(experiment.experiment_id, T.OBJECT_RECORDED, [reward], links)
        return reward
