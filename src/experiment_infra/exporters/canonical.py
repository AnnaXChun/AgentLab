import hashlib
import json
from pathlib import Path

from experiment_infra.core.schema import Artifact, Evidence, Trajectory
from experiment_infra.exporters.rl import rl_rows
from experiment_infra.exporters.sft import sft_rows
from packages.core.models.schemas import EventType as T
from packages.core.models.schemas import uid
from packages.exporters.trajectory import write_jsonl


async def export_experiment(runtime, experiment_id, directory):
    await runtime.ledger.get(experiment_id, "Experiment")
    destination = Path(directory) / experiment_id / uid()
    await runtime.ledger.record(
        experiment_id, T.EXPORT_STARTED, input={"directory": str(destination)}, status="PENDING"
    )
    try:
        objects = await runtime.ledger.objects(experiment_id)
        trajectories = [
            await runtime.get_trajectory(x.trajectory_id)
            for x in objects.values()
            if isinstance(x, Trajectory)
        ]
        destination.mkdir(parents=True, exist_ok=False)
        rows = {
            "trajectory.jsonl": [t.model_dump(mode="json") for t in trajectories],
            "artifacts.jsonl": [
                x.model_dump(mode="json") for x in objects.values() if isinstance(x, Artifact)
            ],
            "evidence.jsonl": [
                x.model_dump(mode="json") for x in objects.values() if isinstance(x, Evidence)
            ],
            "sft.jsonl": [row for t in trajectories for row in sft_rows(t)],
            "rl_transition.jsonl": [row for t in trajectories for row in rl_rows(t)],
        }
        events = runtime.storage.store.events(experiment_id)
        rows["events.jsonl"] = [e.model_dump(mode="json") for e in events]
        graph = await runtime.get_provenance(experiment_id)
        rows["provenance.jsonl"] = [graph]
        for name, values in rows.items():
            write_jsonl(destination / name, values)
        # Include bytes for all versioned artifacts and spilled canonical payloads.
        from adapters.storage.local import LocalArtifactStore

        exported = LocalArtifactStore(destination / "artifacts")
        digests = {x.sha256 for x in objects.values() if isinstance(x, Artifact)}
        digests |= {ref.removeprefix("sha256:") for e in events for ref in e.artifact_refs}
        for digest in digests:
            await exported.put(await runtime.storage.artifacts.get("sha256:" + digest), "export")
        manifest = {
            "schema_version": "2.0",
            "experiment_id": experiment_id,
            "event_count": len(events),
            "last_event_hash": events[-1].state_after_hash,
            "files": {
                name: {
                    "rows": len(values),
                    "sha256": hashlib.sha256((destination / name).read_bytes()).hexdigest(),
                }
                for name, values in rows.items()
            },
            "artifact_hashes": sorted(digests),
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
        await runtime.ledger.record(
            experiment_id,
            T.EXPORT_COMPLETED,
            output={"directory": str(destination), "cutoff_sequence": events[-1].sequence},
        )
        return {
            "schema_version": "2.0",
            "directory": str(destination.resolve()),
            "files": list(rows),
            "manifest": manifest,
        }
    except Exception as exc:
        await runtime.ledger.record(
            experiment_id, T.EXPORT_COMPLETED, output={"error": str(exc)}, status="FAILED"
        )
        raise
