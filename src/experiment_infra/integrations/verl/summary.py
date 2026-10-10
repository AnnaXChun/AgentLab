"""Immutable per-rollout files and batch summaries; no worker-shared append file."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def write_sample(directory, trace, reward, failure, task, export_directory):
    root = Path(directory) / "samples"
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "task": dict(task),
        "policy_trace": trace.model_dump(mode="json"),
        "reward": reward.model_dump(mode="json"),
        "failure": failure,
        "export_directory": export_directory,
    }
    target = root / f"{trace.policy_trace_id}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, ensure_ascii=False))
    temporary.replace(target)


def verify_sample(row):
    """Check the exported immutable evidence graph, not merely nonempty IDs."""
    from experiment_infra.core.schema import OBJECT_TYPES, PolicyTrace, RewardResult
    from experiment_infra.integrations.verl.reward_bridge import RewardBridge

    trace = PolicyTrace.model_validate(row["policy_trace"])
    reward = RewardResult.model_validate(row["reward"])
    bundle = Path(row["export_directory"])
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("Canonical bundle hash mismatch")
    for digest in manifest["artifact_hashes"]:
        if (
            hashlib.sha256(
                (bundle / "artifacts/sha256" / digest[:2] / digest).read_bytes()
            ).hexdigest()
            != digest
        ):
            raise ValueError("Reward input payload hash mismatch")
    graph = json.loads((bundle / "provenance.jsonl").read_text().splitlines()[0])
    objects = {
        oid: OBJECT_TYPES[n["kind"]].model_validate(n["object"])
        for oid, n in graph["nodes"].items()
    }
    if objects.get(trace.policy_trace_id) != trace or objects.get(reward.reward_id) != reward:
        raise ValueError("Sample does not match exported trace/reward")
    run = objects[trace.run_id]
    experiment = objects[trace.experiment_id]
    if reward.run_id != trace.run_id or reward.experiment_id != trace.experiment_id:
        raise ValueError("Reward and trace scopes differ")
    evidence = [objects[x] for x in reward.evidence_ids]
    verifications = [objects[x] for x in reward.verification_ids]
    recomputed = RewardBridge().compute(experiment, run, verifications, evidence=evidence)
    if (
        recomputed.scalar_reward != reward.scalar_reward
        or recomputed.reward_signals != reward.reward_signals
    ):
        raise ValueError("Stored reward does not match verification")
    for e in evidence:
        if not all(a in objects for a in e.source_artifact_ids + e.source_observation_ids):
            raise ValueError("Reward provenance has missing inputs")
    return trace


def summarize(directory):
    from experiment_infra.core.schema import PolicyTrace
    from experiment_infra.integrations.verl.policy_trace import validate_on_policy_batch

    root = Path(directory)
    batches = defaultdict(list)
    for path in (root / "samples").glob("*.json"):
        row = json.loads(path.read_text())
        trace = verify_sample(row)
        batches[(trace.rollout_batch_id, row["task"]["split"])].append(row)
    summaries = []
    for (batch, split), rows in sorted(batches.items()):
        traces = [PolicyTrace.model_validate(r["policy_trace"]) for r in rows]
        validate_on_policy_batch(traces)
        n = len(rows)
        successful = sum(r["reward"]["scalar_reward"] == 1 for r in rows)
        summaries.append(
            {
                "rollout_batch_id": batch,
                "split": split,
                "training_step": traces[0].training_step,
                "num_tasks": len({r["task"]["task_id"] for r in rows}),
                "num_rollouts": n,
                "num_success": successful,
                "num_failure": n - successful,
                "avg_reward": sum(r["reward"]["scalar_reward"] for r in rows) / n,
                "avg_steps": sum(
                    sum(s.trajectory_step_id is not None for s in t.step_spans) for t in traces
                )
                / n,
                "avg_policy_tokens": sum(sum(t.response_mask) for t in traces) / n,
                "avg_tool_tokens": sum(len(t.response_mask) - sum(t.response_mask) for t in traces)
                / n,
                "format_failures": sum(r["failure"] == "FORMAT_FAILURE" for r in rows),
                "environment_failures": sum(r["failure"] == "ENVIRONMENT_FAILURE" for r in rows),
                "verifier_failures": sum(r["failure"] == "VERIFIER_FAILURE" for r in rows),
                "valid_policy_traces": n,
                "valid_reward_provenance": sum(
                    bool(r["reward"]["verification_ids"] and r["reward"]["evidence_ids"])
                    for r in rows
                ),
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "verl_training_summary.json").write_text(json.dumps(summaries, indent=2))
    with (root / "verl_training_summary.csv").open("w") as f:
        if summaries:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))
