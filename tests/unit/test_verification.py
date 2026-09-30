import pytest

from packages.core.events.recorder import Recorder
from packages.core.models.schemas import Claim, ClaimStatus, Episode, Event, EventType
from packages.core.verification.service import VerifierService


async def test_claim_contract():
    service = VerifierService()
    claim = Claim(episode_id="e", text="mean < 0.5", created_by_event_id="event")
    assert service.support(claim, set(), set()).status == ClaimStatus.UNVERIFIED
    with pytest.raises(ValueError):
        await service.verify(claim, "numeric", {})
    claim = claim.model_copy(update={"evidence_refs": ["e"], "artifact_refs": ["a"]})
    with pytest.raises(ValueError):
        service.support(claim, set(), {"a"})
    claim = service.support(claim, {"e"}, {"a"})
    context = {
        "artifact_bytes": b'{"mean": 0.2}',
        "artifact_id": "a",
        "criteria": {"field": "mean", "threshold": 0.5},
    }
    verified, result = await service.verify(claim, "numeric", context)
    assert verified.status == ClaimStatus.VERIFIED
    assert result.verification_id in verified.verification_refs
    context["artifact_bytes"] = b'{"mean": 0.9}'
    contradicted, _ = await service.verify(claim, "numeric", context)
    assert contradicted.status == ClaimStatus.CONTRADICTED


async def test_spill_and_failure_recording(store, artifacts):
    ep = Episode(goal="x", domain_pack="demo", agent_id="mock")
    store.create_episode(
        ep,
        Event(
            episode_id=ep.episode_id,
            branch_id=ep.root_branch_id,
            event_type=EventType.EPISODE_STARTED,
            output={"episode": ep.model_dump(mode="json")},
        ),
    )
    recorder = Recorder(store, artifacts, ep.episode_id, ep.root_branch_id)
    e = await recorder.emit(EventType.AGENT_MESSAGE, output={"text": "x" * 20000})
    assert "$artifact" in e.output
    assert len((await recorder.resolve(e.output))["text"]) == 20000

    async def fail():
        raise RuntimeError("broken")

    with pytest.raises(RuntimeError):
        await recorder.operation(EventType.TOOL_CALL, EventType.TOOL_RESULT, {}, fail)
    assert store.events(ep.episode_id)[-1].status == "FAILED"
