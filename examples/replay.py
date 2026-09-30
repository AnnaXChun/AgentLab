"""Offline replay of an exported canonical log plus its artifact closure."""

import asyncio
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.storage.local import LocalArtifactStore
from packages.core.models.migrations import migrate
from packages.core.models.schemas import Event
from packages.provenance.replay import replay


async def replay_export(directory):
    directory = Path(directory)
    store = LocalArtifactStore(directory / "artifacts")
    events = [
        Event.model_validate(migrate(json.loads(line)))
        for line in (directory / "raw.jsonl").read_text().splitlines()
    ]

    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        not events
        or len(events) != manifest["event_count"]
        or events[-1].state_after_hash != manifest["last_event_hash"]
    ):
        raise ValueError("Export manifest does not match event history")

    async def resolve(value):
        if set(value) == {"$artifact"}:
            return json.loads(await store.get(value["$artifact"]))
        return value

    for ref in json.loads((directory / "manifest.json").read_text())["artifact_ids"]:
        await store.get(ref)
    return await replay(events, resolve)


if __name__ == "__main__":
    result = asyncio.run(replay_export(sys.argv[1]))
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "event_count": result["event_count"],
                "branches": list(result["branches"]),
            }
        )
    )
