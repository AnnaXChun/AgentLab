import pytest

from packages.core.events.store import event_hash
from packages.core.models.schemas import Episode, Event, EventType


async def test_cas(artifacts):
    a = await artifacts.put(b"hello", "event")
    b = await artifacts.put(b"hello", "other")
    assert a.artifact_id == b.artifact_id
    assert await artifacts.get(a.artifact_id) == b"hello"
    assert await artifacts.exists(a.artifact_id)
    with pytest.raises(ValueError):
        await artifacts.get("../escape")
    path = artifacts.path(a.artifact_id)
    path.chmod(0o644)
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="refusing overwrite"):
        await artifacts.put(b"hello", "x")


def test_append(store):
    ep = Episode(goal="x", domain_pack="demo", agent_id="mock")
    first = store.create_episode(
        ep,
        Event(
            episode_id=ep.episode_id,
            branch_id=ep.root_branch_id,
            event_type=EventType.EPISODE_STARTED,
            output={"episode": ep.model_dump(mode="json")},
        ),
    )
    second = store.append(
        Event(
            episode_id=ep.episode_id,
            branch_id=ep.root_branch_id,
            event_type=EventType.AGENT_MESSAGE,
        )
    )
    assert second.sequence == 2
    assert second.state_before_hash == first.state_after_hash
    assert event_hash(second) == second.state_after_hash
    assert store.episode(ep.episode_id) == ep
    assert len(store.edges(ep.episode_id)) == 1
