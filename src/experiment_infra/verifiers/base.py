from typing import Protocol

from experiment_infra.core.schema import Artifact, Claim, Experiment, Run, VerificationResult

ArtifactInputs = dict[str, tuple[Artifact, bytes]]


class Verifier(Protocol):
    async def verify(
        self, experiment: Experiment, run: Run, artifacts: ArtifactInputs, claims: list[Claim]
    ) -> VerificationResult: ...


# Extension contracts only. No LLM calls, human workflow or statistics engine in V0.
class DeterministicVerifier(Verifier, Protocol): ...


class StatisticalVerifier(Verifier, Protocol): ...


class LLMVerifier(Verifier, Protocol): ...


class HumanVerifier(Verifier, Protocol): ...
