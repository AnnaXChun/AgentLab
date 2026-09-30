import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from examples.toy_experiment.run import run_demo
from experiment_infra.core.schema import Action, Trajectory
from experiment_infra.runtime.executor import ExperimentRuntime
from packages.core.models.schemas import EventType as T
from packages.sdk.api import create_app

pytestmark = pytest.mark.integration


async def test_experiment_slice_replay_branch_provenance_export(runtime, tmp_path):
    rt = ExperimentRuntime(runtime)
    summary = await run_demo(rt, tmp_path / "experiment_exports")
    assert summary["replay"]["matched"]
    assert summary["original_metrics"] == {
        "N": 2000,
        "result_a": 2001000,
        "result_b": 2001000,
        "operations_a": 2000,
        "operations_b": 1,
    }
    assert summary["branch_metrics"]["N"] == 4000
    assert summary["claim"]["status"] == "SUPPORTED"
    assert summary["original_trajectory_unchanged"]
    graph = await rt.get_provenance(summary["claim"]["claim_id"])
    artifact_types = {
        v["object"]["type"] for v in graph["nodes"].values() if v["kind"] == "Artifact"
    }
    assert {"source_code", "config", "metric", "log"} <= artifact_types
    path = Path(summary["export"]["directory"])
    trajectories = [
        Trajectory.model_validate_json(line)
        for line in (path / "trajectory.jsonl").read_text().splitlines()
    ]
    assert len(trajectories) == 3
    assert [len(t.steps) for t in trajectories] == [2, 2, 1]
    assert len(trajectories[0].steps[0].verifier_feedback) == 1
    assert len(trajectories[0].steps[1].verifier_feedback) == 1
    for file in (
        "trajectory.jsonl",
        "artifacts.jsonl",
        "evidence.jsonl",
        "sft.jsonl",
        "rl_transition.jsonl",
    ):
        assert (path / file).stat().st_size > 0
    rows = [json.loads(x) for x in (path / "sft.jsonl").read_text().splitlines()]
    assert set(rows[0]["input"]) == {"state", "context_refs"}
    assert set(rows[0]["output"]) == {"action"}
    rl = [json.loads(x) for x in (path / "rl_transition.jsonl").read_text().splitlines()]
    assert all(x["reward_signals"] == {} for x in rl)
    assert len([x for x in rl if x["terminal"]]) == 3
    assert "private_chain_of_thought" not in (path / "sft.jsonl").read_text()
    # Reconnect: all first-class records survive runtime object destruction.
    reloaded = ExperimentRuntime(runtime)
    assert (await reloaded.get_trajectory(summary["trajectory_id"])).model_dump() == trajectories[
        0
    ].model_dump()
    for table in ("experiment_objects", "provenance_edges"):
        with pytest.raises(DBAPIError):
            with runtime.engine.begin() as connection:
                connection.execute(text(f"DELETE FROM {table} WHERE 1=0"))


async def test_artifact_versions_failed_action_and_replay_mismatch(runtime):
    rt = ExperimentRuntime(runtime)
    exp = await rt.create_experiment("versions")
    run = await rt.start_run(exp.experiment_id)
    try:

        def write_action(value):
            return Action(
                tool="environment",
                operation="exec",
                arguments={
                    "command": {
                        "argv": [
                            "python",
                            "-c",
                            f"from pathlib import Path; Path('value.txt').write_text('{value}')",
                        ],
                        "collect": ["value.txt"],
                    }
                },
            )

        first = await rt.execute(exp.experiment_id, write_action("a"), run_id=run.run_id)
        second = await rt.execute(exp.experiment_id, write_action("b"), run_id=run.run_id)
        a = await rt.ledger.get(
            first.observation.structured_output["collected"]["value.txt"], "Artifact"
        )
        b = await rt.ledger.get(
            second.observation.structured_output["collected"]["value.txt"], "Artifact"
        )
        assert a.artifact_id != b.artifact_id and a.sha256 != b.sha256
        assert b.parent_artifact_ids == [a.artifact_id]
        assert await rt.artifact_bytes(a.artifact_id) == b"a"
        with pytest.raises(ValueError, match="Immutable"):
            await rt.ledger.record(
                exp.experiment_id, T.OBJECT_RECORDED, [a.model_copy(update={"sha256": b.sha256})]
            )
        failed = await rt.execute(
            exp.experiment_id,
            Action(tool="environment", operation="unsupported"),
            run_id=run.run_id,
        )
        assert failed.observation.exit_code == -1
        assert failed.observation.structured_output["error_type"] == "ValueError"
        assert len((await rt.get_trajectory(run.trajectory_id)).steps) == 3
        with pytest.raises(ValueError):
            await rt.branch(run.run_id, from_step=99)
    finally:
        await rt.finish_run(run.run_id)
    # Replaying true entropy must report a mismatch, never invent success.
    noisy = await rt.start_run(exp.experiment_id)
    await rt.execute(
        exp.experiment_id,
        Action(
            tool="environment",
            operation="exec",
            arguments={
                "command": {
                    "argv": [
                        "python",
                        "-c",
                        "import secrets,pathlib; pathlib.Path('random.txt').write_text(secrets.token_hex(32))",
                    ],
                    "collect": ["random.txt"],
                }
            },
        ),
        run_id=noisy.run_id,
    )
    await rt.finish_run(noisy.run_id)
    assert not (await rt.replay(noisy.run_id)).matched


async def test_metric_evidence_refutation_and_no_agent_log_certification(runtime):
    rt = ExperimentRuntime(runtime)
    exp = await rt.create_experiment(
        "criterion",
        success_criteria={
            "metric": {"field": "score", "operator": "gt", "baseline": 0.5},
            "test": {"argv": ["python", "-c", "print('real')"]},
        },
    )
    h = await rt.register_hypothesis(exp.experiment_id, "score beats baseline")
    run = await rt.start_run(exp.experiment_id)
    try:
        step = await rt.execute(
            exp.experiment_id,
            Action(
                tool="environment",
                operation="exec",
                arguments={
                    "command": {
                        "argv": [
                            "python",
                            "-c",
                            "from pathlib import Path; Path('metrics.json').write_text('{\"score\":0.1}')",
                        ],
                        "collect": ["metrics.json"],
                    },
                    "artifact_types": {"metrics.json": "metric"},
                },
            ),
            run_id=run.run_id,
        )
        artifact_id = step.observation.structured_output["collected"]["metrics.json"]
        evidence = await rt.verify(
            exp.experiment_id, run.run_id, h.hypothesis_id, "metric", [artifact_id]
        )
        assert evidence.verification_status == "FAILED"
        claim = await rt.create_claim(exp.experiment_id, "score > baseline", [evidence.evidence_id])
        assert claim.status == "REFUTED"
        with pytest.raises(ValueError):
            await rt.verify(exp.experiment_id, run.run_id, h.hypothesis_id, "test", [artifact_id])
        other = await rt.create_experiment("other")
        with pytest.raises(ValueError, match="Cross-experiment"):
            await rt.create_claim(other.experiment_id, "borrowed", [evidence.evidence_id])
    finally:
        await rt.finish_run(run.run_id)


def test_experiment_api_surface_and_status_safety(runtime, tmp_path, monkeypatch):
    monkeypatch.setenv("EXPORT_ROOT", str(tmp_path))
    with TestClient(create_app(runtime)) as client:
        exp = client.post(
            "/experiments",
            json={
                "goal": "HTTP execution",
                "success_criteria": {
                    "metric": {"field": "score", "operator": "gt", "baseline": 0.5}
                },
            },
        ).json()
        eid = exp["experiment_id"]
        base = f"/experiments/{eid}"
        assert client.get(base).json()["schema_version"] == "2.0"
        h = client.post(base + "/hypotheses", json={"statement": "score exceeds baseline"}).json()
        run = client.post(base + "/runs", json={"config": {"N": 1}}).json()
        rid = run["run_id"]
        action = {
            "tool": "environment",
            "operation": "exec",
            "arguments": {
                "command": {
                    "argv": [
                        "python",
                        "-c",
                        "from pathlib import Path; Path('metric.json').write_text('{\"score\":0.9}')",
                    ],
                    "collect": ["metric.json"],
                },
                "artifact_types": {"metric.json": "metric"},
            },
        }
        response = client.post(base + "/actions", json={"run_id": rid, "action": action})
        assert response.status_code == 200, response.text
        step = response.json()
        evidence = client.post(
            base + "/verify",
            json={
                "run_id": rid,
                "hypothesis_id": h["hypothesis_id"],
                "verifier": "metric",
                "artifact_ids": [
                    step["observation"]["structured_output"]["collected"]["metric.json"]
                ],
            },
        ).json()
        assert (
            client.post(
                base + "/claims", json={"statement": "forged", "status": "SUPPORTED"}
            ).status_code
            == 422
        )
        claim = client.post(
            base + "/claims",
            json={"statement": "supported", "evidence_ids": [evidence["evidence_id"]]},
        ).json()
        assert claim["status"] == "SUPPORTED"
        assert client.get(f"/objects/{claim['claim_id']}/provenance").status_code == 200
        assert client.get(f"/trajectories/{run['trajectory_id']}").status_code == 200
        assert client.get(base + "/state", params={"run_id": rid}).status_code == 200
        assert (
            client.post(f"/episodes/{eid}/claims", json={"text": "legacy bypass"}).status_code
            == 400
        )
        assert client.post(f"/runs/{rid}/finish").status_code == 200
        replay = client.post(f"/runs/{rid}/replay").json()
        assert replay["matched"]
        branch = client.post(
            f"/runs/{rid}/branch", json={"from_step": 1, "config": {"N": 2}}
        ).json()
        assert client.post(f"/runs/{branch['run_id']}/finish").status_code == 200
        assert client.post(base + "/finish").status_code == 200
        assert client.post("/exports", json={"experiment_id": eid}).status_code == 200


async def test_snapshot_failure_keeps_the_action_in_trajectory(runtime):
    rt = ExperimentRuntime(runtime)
    exp = await rt.create_experiment("unsupported snapshot entry")
    run = await rt.start_run(exp.experiment_id)
    try:
        broken = await rt.execute(
            exp.experiment_id,
            Action(
                tool="environment",
                operation="exec",
                arguments={
                    "command": {
                        "argv": ["python", "-c", "import os; os.symlink('/etc/passwd','link')"]
                    }
                },
            ),
            run_id=run.run_id,
        )
        assert broken.observation.exit_code == 0
        assert broken.state_after["snapshot_id"] is None
        assert "snapshot_error" in broken.state_after
        with pytest.raises(ValueError, match="snapshot"):
            await rt.branch(run.run_id, from_step=1)
        recovered = await rt.execute(
            exp.experiment_id,
            Action(
                tool="environment",
                operation="exec",
                arguments={
                    "command": {
                        "argv": ["python", "-c", "import pathlib; pathlib.Path('link').unlink()"]
                    }
                },
            ),
            run_id=run.run_id,
        )
        assert recovered.state_after["snapshot_id"]
        assert len((await rt.get_trajectory(run.trajectory_id)).steps) == 2
    finally:
        result = await rt.finish_run(run.run_id)
        assert result.status == "FAILED"


async def test_mcp_and_rag_are_standard_action_observation_artifacts(runtime):
    from domain_packs.loader import load_pack

    legacy = await runtime.create("context ingestion")
    await legacy.ingest(
        b"Infrastructure context: deterministic addition counts.",
        f"test://{legacy.episode_id}",
        "Context",
    )
    await legacy.complete()
    retrieval, gateway = runtime.components(load_pack("demo"))
    rt = ExperimentRuntime(runtime, retriever=retrieval, gateway=gateway)
    exp = await rt.create_experiment("tool integrations")
    run = await rt.start_run(exp.experiment_id)
    try:
        tool = await rt.execute(
            exp.experiment_id,
            Action(
                tool="mcp",
                operation="call",
                arguments={"tool_name": "load_dataset", "arguments": {"candidate": "A"}},
            ),
            run_id=run.run_id,
        )
        assert tool.observation.exit_code == 0
        assert tool.observation.structured_output["structuredContent"]["values"] == [
            0.18,
            0.20,
            0.22,
        ]
        context = await rt.execute(
            exp.experiment_id,
            Action(
                tool="rag",
                operation="retrieve",
                arguments={"query": "deterministic addition counts"},
            ),
            run_id=run.run_id,
        )
        assert context.observation.exit_code == 0
        assert context.observation.created_artifacts
        assert context.observation.structured_output["context"]
        with pytest.raises(KeyError):
            await rt.create_claim(
                exp.experiment_id,
                "retrieval is not experimental evidence",
                [context.observation.structured_output["context"][0]["evidence_id"]],
            )
        assert len((await rt.get_trajectory(run.trajectory_id)).steps) == 2
    finally:
        await rt.finish_run(run.run_id)
        with runtime.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM retrieval_chunks WHERE source_uri=:uri"),
                {"uri": f"test://{legacy.episode_id}"},
            )
