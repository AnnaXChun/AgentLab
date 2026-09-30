import json
import math
from typing import Protocol

from jsonschema import Draft202012Validator, ValidationError

from packages.core.models.schemas import Claim, ClaimStatus, Verification


class Verifier(Protocol):
    async def verify(self, claim: Claim, context: dict) -> Verification: ...


class NumericVerifier:
    async def verify(self, claim: Claim, context: dict) -> Verification:
        data = json.loads(context["artifact_bytes"])
        config = context["criteria"]
        observed = data[config["field"]]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (float, int))
            or not math.isfinite(observed)
        ):
            raise ValueError("Observed value must be finite numeric data")
        if config.get("operator", "lt") == "lt":
            expected = float(config["threshold"])
            passed = observed < expected
        elif config["operator"] == "approx":
            expected = float(config["expected"])
            epsilon = float(config["epsilon"])
            if not math.isfinite(epsilon) or epsilon <= 0:
                raise ValueError("epsilon must be positive and finite")
            passed = abs(observed - expected) < epsilon
        else:
            raise ValueError("Unsupported numeric operator")
        if not math.isfinite(expected):
            raise ValueError("Expected value must be finite")
        return Verification(
            claim_id=claim.claim_id,
            verifier_type="numeric",
            status="PASSED" if passed else "FAILED",
            score=float(passed),
            details={"observed": observed, "criteria": config},
            artifact_refs=[context["artifact_id"]],
        )


class SchemaVerifier:
    async def verify(self, claim: Claim, context: dict) -> Verification:
        schema = context["criteria"]["json_schema"]
        Draft202012Validator.check_schema(schema)
        try:
            Draft202012Validator(schema).validate(json.loads(context["artifact_bytes"]))
            status, details = "PASSED", {}
        except ValidationError as exc:
            status, details = "FAILED", {"error": exc.message}
        return Verification(
            claim_id=claim.claim_id,
            verifier_type="schema",
            status=status,
            details=details,
            artifact_refs=[context["artifact_id"]],
        )


class ExecutionVerifier:
    async def verify(self, claim: Claim, context: dict) -> Verification:
        passed = (
            context["execution"]["exit_code"] == 0
            and all(context["required_exist"])
            and bool(context["required_exist"])
        )
        return Verification(
            claim_id=claim.claim_id,
            verifier_type="execution",
            status="PASSED" if passed else "FAILED",
            details={
                "exit_code": context["execution"]["exit_code"],
                "required_exist": context["required_exist"],
            },
            artifact_refs=claim.artifact_refs,
        )


class VerifierService:
    def __init__(self, verifiers: dict[str, Verifier] | None = None):
        self.verifiers = verifiers or {
            "numeric": NumericVerifier(),
            "schema": SchemaVerifier(),
            "execution": ExecutionVerifier(),
        }

    def support(self, claim: Claim, evidence_ids: set[str], artifact_ids: set[str]) -> Claim:
        if not claim.evidence_refs or not claim.artifact_refs:
            return claim
        if (
            not set(claim.evidence_refs) <= evidence_ids
            or not set(claim.artifact_refs) <= artifact_ids
        ):
            raise ValueError("Claim references must resolve within the branch")
        return claim.model_copy(update={"status": ClaimStatus.SUPPORTED})

    async def verify(
        self, claim: Claim, verifier_type: str, context: dict
    ) -> tuple[Claim, Verification]:
        if (
            claim.status
            not in (ClaimStatus.SUPPORTED, ClaimStatus.VERIFIED, ClaimStatus.CONTRADICTED)
            or not claim.evidence_refs
            or not claim.artifact_refs
        ):
            raise ValueError("Claim must have resolved evidence and artifacts before verification")
        verifier = self.verifiers[verifier_type]
        result = await verifier.verify(claim, context)
        status = ClaimStatus.VERIFIED if result.status == "PASSED" else ClaimStatus.CONTRADICTED
        return claim.model_copy(
            update={
                "status": status,
                "verification_refs": [*claim.verification_refs, result.verification_id],
            }
        ), result
