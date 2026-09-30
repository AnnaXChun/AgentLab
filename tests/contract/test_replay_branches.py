import pytest

from domain_packs.loader import load_pack
from packages.provenance.replay import replay
from packages.sdk.harness import Harness


async def test_fork_rollback_and_integrity(store, artifacts):
    h = await Harness.create(store, artifacts, None, None, None, load_pack("demo"), "test")
    cp = await h.checkpoint()
    a = await h.fork(cp["checkpoint_id"])
    b = await h.fork(cp["checkpoint_id"])
    await a.claim("A only", [], [])
    assert len((await a.state())["claims"]) == 1
    assert not (await b.state())["claims"]
    before = [e.model_dump_json() for e in store.events(h.episode_id) if e.branch_id == a.branch_id]
    await b.claim("B only", [], [])
    assert before == [
        e.model_dump_json() for e in store.events(h.episode_id) if e.branch_id == a.branch_id
    ]
    await h.claim("after checkpoint", [], [])
    await h.rollback(cp["checkpoint_id"])
    assert not (await h.state())["claims"]
    assert len((await h.replay())["branches"]) == 3
    events = store.events(h.episode_id)
    events[-1] = events[-1].model_copy(update={"output": {"tampered": True}})
    with pytest.raises(ValueError, match="hash"):
        await replay(events, h.recorder.resolve)
    with pytest.raises(ValueError):
        await b.rollback(cp["checkpoint_id"])
