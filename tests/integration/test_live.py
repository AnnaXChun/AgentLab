from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from examples.replay import replay_export
from examples.run_demo import run_demo
from packages.core.models.schemas import Event
from packages.core.models.schemas import EventType as T
from packages.sandbox.base import Command
from packages.sdk.api import create_app

pytestmark = pytest.mark.integration


async def test_real_end_to_end(runtime, tmp_path):
    summary = await run_demo(runtime, tmp_path / "export")
    assert summary["claim"]["status"] == "VERIFIED"
    assert summary["verification"]["details"]["observed"] == pytest.approx(0.2)
    assert len(summary["branches"]) == 3
    paths = summary["export"]["paths"]
    replayed = await replay_export(Path(paths["raw.jsonl"]).parent)
    assert replayed["event_count"] == summary["export"]["event_count"]
    assert len(Path(paths["failure_recovery.jsonl"]).read_text().splitlines()) == 1
    events = runtime.store.events(summary["episode_id"])
    for start, end in [
        (T.RETRIEVAL_QUERY, T.RETRIEVAL_RESULT),
        (T.TOOL_CALL, T.TOOL_RESULT),
        (T.SANDBOX_COMMAND, T.SANDBOX_RESULT),
        (T.VERIFICATION_STARTED, T.VERIFICATION_RESULT),
    ]:
        intents = {e.event_id for e in events if e.event_type == start}
        completions = {e.input.get("request_event_id") for e in events if e.event_type == end}
        assert intents <= completions
    assert all(e.trace_id and e.span_id for e in events)
    assert not runtime.sandbox.client.containers.list(
        filters={"label": "scientific-harness=sandbox"}
    )


async def test_live_sandbox_timeout_and_snapshot(runtime):
    backend = runtime.sandbox
    handle = await backend.create({"image": "python:3.12-slim"})
    try:
        await backend.upload(handle, "subdir/data.txt", b"original")
        assert await backend.download(handle, "subdir/data.txt") == b"original"
        snapshot = await backend.snapshot(handle)
        await backend.upload(handle, "subdir/data.txt", b"modified")
        await backend.restore(handle, snapshot)
        assert await backend.download(handle, "subdir/data.txt") == b"original"
        result = await backend.exec(
            handle,
            Command(argv=["python", "-c", "import time; time.sleep(30)"], timeout_seconds=0.1),
        )
        assert result.exit_code == 124 and result.timed_out
        result = await backend.exec(
            handle,
            Command(
                argv=["python", "-c", "import os; print(os.getenv('TEST_VALUE'))"],
                environment={"TEST_VALUE": "hello"},
            ),
        )
        assert result.stdout.strip() == "hello"
        # Experiment modules must not shadow the harness supervisor's stdlib imports.
        await backend.upload(handle, "json.py", b"raise RuntimeError('shadowed control import')")
        isolated = await backend.exec(handle, Command(argv=["python", "-I", "-c", "print('safe')"]))
        assert isolated.exit_code == 0 and isolated.stdout.strip() == "safe"
        assert await backend.download(handle, "json.py")
        assert await backend.snapshot(handle)
        await backend.exec(
            handle, Command(argv=["python", "-c", "import os; os.symlink('/etc/passwd','bad')"])
        )
        with pytest.raises(RuntimeError):
            await backend.download(handle, "bad")
    finally:
        await backend.destroy(handle)


async def test_postgres_append_only_and_concurrent_writers(runtime):
    h = await runtime.create("immutable log")
    for sql in (
        "UPDATE events SET event_type='bad' WHERE episode_id=:id",
        "DELETE FROM events WHERE episode_id=:id",
        "TRUNCATE event_edges",
    ):
        with pytest.raises(DBAPIError):
            with runtime.engine.begin() as connection:
                connection.execute(text(sql), {"id": h.episode_id})

    def append(i):
        return runtime.store.append(
            Event(
                episode_id=h.episode_id,
                branch_id=h.branch_id,
                event_type=T.AGENT_MESSAGE,
                output={"text": str(i)},
            )
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append, range(12)))
    assert (await h.replay())["event_count"] == 13
    await h.complete()


async def test_retrieval_scope_and_stability(runtime):
    h = await runtime.create("retrieval")
    try:
        await h.ingest(
            b"Unique test reference: candidate mean threshold 0.5",
            f"test://{h.episode_id}",
            "Stable",
        )
        one = await h.retrieve("Unique test reference")
        two = await h.retrieve("Unique test reference")
        assert one["evidence"][0]["evidence_id"] == two["evidence"][0]["evidence_id"]
        with pytest.raises(PermissionError):
            await h.tool("not_allowed", {})
        assert h.store.events(h.episode_id)[-1].status == "FAILED"
    finally:
        await h.complete()
        with runtime.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM retrieval_chunks WHERE source_uri=:uri"),
                {"uri": f"test://{h.episode_id}"},
            )


def test_api_claim_safety_and_unknown_branch(runtime, tmp_path, monkeypatch):
    monkeypatch.setenv("EXPORT_ROOT", str(tmp_path / "api_exports"))
    with TestClient(create_app(runtime)) as client:
        assert client.get("/health").status_code == 200
        ep = client.post("/episodes", json={"goal": "API test"}).json()
        base = f"/episodes/{ep['episode_id']}"
        assert (
            client.post(base + "/claims", json={"text": "forge", "status": "VERIFIED"}).status_code
            == 422
        )
        claim = client.post(base + "/claims", json={"text": "unverified"}).json()
        assert claim["status"] == "UNVERIFIED"
        assert (
            client.post(
                base + "/claims", json={"text": "invalid", "evidence_refs": ["missing"]}
            ).status_code
            == 400
        )
        assert (
            client.post(base + "/retrieve", json={"query": "x", "branch_id": "missing"}).status_code
            == 400
        )
        assert (
            client.post(
                base + "/claims/" + claim["claim_id"] + "/verify",
                json={"verifier_type": "numeric", "artifact_id": "missing", "criteria": {}},
            ).status_code
            == 400
        )
        checkpoint = client.post(base + "/checkpoint", json={}).json()["data"]["checkpoint_id"]
        branch = client.post(base + "/fork", json={"checkpoint_id": checkpoint}).json()["data"][
            "branch_id"
        ]
        assert branch != ep["root_branch_id"]
        assert client.get(base + "/events").status_code == 200
        assert client.post(base + "/complete", json={}).status_code == 200
        assert client.post(base + "/retrieve", json={"query": "after done"}).status_code == 400
        assert client.post(base + "/export", json={}).status_code == 200
