"""Run with `python -m examples.run_demo` or `python examples/run_demo.py`."""

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.agents.driver import drive_legacy_agent
from adapters.agents.mock import MockAgentAdapter
from domain_packs.loader import ROOT
from packages.sandbox.base import Command
from packages.sdk.runtime import Runtime


async def run_demo(runtime=None, output_root=None):
    runtime = runtime or Runtime()
    h = await runtime.create("Determine whether candidate A satisfies property X.")
    output_root = output_root or os.getenv("EXPORT_ROOT", "outputs")
    try:
        await h.ingest(
            (ROOT / "demo" / "reference.md").read_bytes(),
            "pack://demo/reference.md",
            "Synthetic property X reference",
        )
        verified = await drive_legacy_agent(h, MockAgentAdapter())
        assert verified["claim"]["status"] == "VERIFIED"
        cp = await h.checkpoint()
        branch_a = await h.fork(cp["checkpoint_id"])
        failed = await branch_a.execute(
            Command(argv=["python", "-c", "raise RuntimeError('intentional failed experiment')"])
        )
        assert failed["exit_code"] != 0
        branch_b = await h.fork(cp["checkpoint_id"])
        success = await branch_b.execute(
            Command(
                argv=[
                    "python",
                    "-c",
                    "import json,pathlib; old=json.loads(pathlib.Path('result.json').read_text()); old['recovered']=True; pathlib.Path('recovered.json').write_text(json.dumps(old))",
                ],
                collect=["recovered.json"],
            ),
            retry_of=failed["event_id"],
        )
        assert success["exit_code"] == 0
        # Exercise append-only rollback and physical restoration on branch B.
        await branch_b.checkpoint()
        await h.rollback(cp["checkpoint_id"])
        await h.complete()
        replayed = await h.replay()
        exported = await h.export(output_root)
        summary = {
            "schema_version": "1.0",
            "episode_id": h.episode_id,
            "status": h.store.episode(h.episode_id).status,
            "claim": verified["claim"],
            "verification": verified["verification"],
            "branches": {
                "main": h.branch_id,
                "branch_a": branch_a.branch_id,
                "branch_b": branch_b.branch_id,
            },
            "replay_event_count": replayed["event_count"],
            "event_types": dict(Counter(e.event_type.value for e in h.store.events(h.episode_id))),
            "export": exported,
        }
        target = Path(exported["paths"]["manifest.json"]).parent / "demo_summary.json"
        target.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return summary
    except BaseException:
        if h.store.episode(h.episode_id).status == "RUNNING":
            await h.complete(failed=True)
        raise


if __name__ == "__main__":
    asyncio.run(run_demo())
