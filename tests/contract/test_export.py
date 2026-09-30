import json

import duckdb

from domain_packs.loader import load_pack
from packages.core.models.schemas import EventEdge, uid
from packages.core.models.schemas import EventType as T
from packages.sdk.harness import Harness


async def test_all_export_formats(store, artifacts, tmp_path):
    h = await Harness.create(store, artifacts, None, None, None, load_pack("demo"), "test")
    r = h.recorder
    request = await r.emit(
        T.SANDBOX_COMMAND,
        input={"operation": "exec", "command": {"argv": ["false"]}},
        status="PENDING",
    )
    failure = await r.emit(
        T.SANDBOX_RESULT,
        input={"request_event_id": request.event_id},
        output={"exit_code": 1},
        status="FAILED",
    )
    event_id = uid()
    corrected = await r.emit(
        T.SANDBOX_COMMAND,
        event_id=event_id,
        input={"operation": "exec", "command": {"argv": ["true"]}},
        edges=[
            EventEdge(source_event_id=failure.event_id, target_event_id=event_id, relation="retry")
        ],
    )
    await r.emit(
        T.SANDBOX_RESULT, input={"request_event_id": corrected.event_id}, output={"exit_code": 0}
    )
    await h.complete()
    result = await h.export(tmp_path / "exports")
    paths = result["paths"]
    assert {
        "raw.jsonl",
        "events.parquet",
        "sft.jsonl",
        "rl.jsonl",
        "failure_recovery.jsonl",
    } <= paths.keys()
    assert len(open(paths["failure_recovery.jsonl"]).readlines()) == 1
    with duckdb.connect() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [paths["events.parquet"]]
            ).fetchone()[0]
            == result["event_count"]
        )
    rows = [json.loads(line) for line in open(paths["rl.jsonl"])]
    assert rows[-1]["terminal"] and rows[-1]["reward_optional"] is None
    assert "chain_of_thought" not in open(paths["sft.jsonl"]).read()
