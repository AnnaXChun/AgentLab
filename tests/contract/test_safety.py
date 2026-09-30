from pathlib import Path

import pytest
from pydantic import ValidationError

from domain_packs.loader import load_pack
from examples.replay import replay_export
from packages.core.models.schemas import Claim, ClaimStatus, Evidence
from packages.core.models.schemas import EventType as T
from packages.core.verification.service import ExecutionVerifier, NumericVerifier, SchemaVerifier
from packages.sdk.harness import Harness


def supported_claim():
    return Claim(
        episode_id="e",
        text="criterion",
        created_by_event_id="event",
        status=ClaimStatus.SUPPORTED,
        evidence_refs=["evidence"],
        artifact_refs=["artifact"],
    )


def test_public_claim_cannot_encode_unsupported_verified_state():
    with pytest.raises(ValidationError):
        Claim(episode_id="e", text="fake", created_by_event_id="x", status="SUPPORTED")
    with pytest.raises(ValidationError):
        Claim(
            episode_id="e",
            text="fake",
            created_by_event_id="x",
            status="VERIFIED",
            evidence_refs=["e"],
            artifact_refs=["a"],
        )


async def test_verifiers_check_real_values_and_schema():
    claim = supported_claim()
    context = {
        "artifact_bytes": b'{"mean": 0.2}',
        "artifact_id": "artifact",
        "criteria": {
            "json_schema": {
                "type": "object",
                "required": ["mean"],
                "properties": {"mean": {"type": "number"}},
            }
        },
    }
    assert (await SchemaVerifier().verify(claim, context)).status == "PASSED"
    context["artifact_bytes"] = b'{"mean": "wrong"}'
    assert (await SchemaVerifier().verify(claim, context)).status == "FAILED"
    assert (
        await ExecutionVerifier().verify(
            claim, {"execution": {"exit_code": 0}, "required_exist": [True]}
        )
    ).status == "PASSED"
    assert (
        await ExecutionVerifier().verify(
            claim, {"execution": {"exit_code": 0}, "required_exist": [False]}
        )
    ).status == "FAILED"
    assert (
        await ExecutionVerifier().verify(
            claim, {"execution": {"exit_code": 1}, "required_exist": [True]}
        )
    ).status == "FAILED"
    context = {
        "artifact_bytes": b'{"mean": 0.2}',
        "artifact_id": "artifact",
        "criteria": {"field": "mean", "operator": "approx", "expected": 0.2, "epsilon": 0.001},
    }
    assert (await NumericVerifier().verify(claim, context)).status == "PASSED"
    context["artifact_bytes"] = b'{"mean": true}'
    with pytest.raises(ValueError):
        await NumericVerifier().verify(claim, context)


async def test_branch_artifact_scope_and_verification_failure(store, artifacts):
    h = await Harness.create(store, artifacts, None, None, None, load_pack("demo"), "scope")
    evidence = Evidence(
        evidence_id="e",
        document_id="d",
        document_hash="hash",
        source_uri="test://e",
        title="Evidence",
        chunk_id="0",
        content="threshold 0.5",
        retrieval_score=1,
    )
    await h.recorder.emit(
        T.RETRIEVAL_RESULT, output={"evidence": [evidence.model_dump(mode="json")]}
    )
    cp = await h.checkpoint()
    a = await h.fork(cp["checkpoint_id"])
    b = await h.fork(cp["checkpoint_id"])
    artifact = await a.recorder.artifact(b'{"mean": 0.9}', "application/json")
    with pytest.raises(ValueError, match="cross-branch"):
        await b.claim("cannot borrow sibling output", ["e"], [artifact.artifact_id])
    claim = await a.claim("mean < 0.5", ["e"], [artifact.artifact_id])
    result = await a.verify(
        claim["claim_id"], "numeric", artifact.artifact_id, {"field": "mean", "threshold": 0.5}
    )
    assert result["claim"]["status"] == "CONTRADICTED"
    state = await a.state()
    assert state["claims"][claim["claim_id"]]["status"] == "CONTRADICTED"
    assert not (await b.state())["claims"]
    assert "verified_by" in {edge.relation for edge in store.edges(h.episode_id)}


async def test_export_manifest_detects_missing_tail(store, artifacts, tmp_path):
    h = await Harness.create(store, artifacts, None, None, None, load_pack("demo"), "export")
    await h.complete()
    exported = await h.export(tmp_path)
    path = Path(exported["paths"]["raw.jsonl"])
    rows = path.read_text().splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="manifest"):
        await replay_export(path.parent)
