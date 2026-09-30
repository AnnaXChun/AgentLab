"""An external scripted client, not an autonomous scientist inside the runtime."""

# ruff: noqa: E402 -- supports the requested direct-file entry point
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiment_infra.agents.external import ExternalAgentBackend
from experiment_infra.core.schema import Action
from experiment_infra.domains.science_demo.pack import PACK
from experiment_infra.runtime.executor import ExperimentRuntime
from experiment_infra.runtime.replay import link_counterfactual

HERE = Path(__file__).parent


async def run_demo(runtime=None, output_root=None):
    runtime = runtime or ExperimentRuntime()
    experiment = await runtime.create_experiment(
        "Compare deterministic operation counts of two sum algorithms",
        agent_backend="external-scripted-client",
        environment_spec=PACK.environment,
        success_criteria=PACK.templates["success_criteria"],
    )
    eid = experiment.experiment_id
    print("Experiment created", eid)
    hypothesis = await runtime.register_hypothesis(
        eid,
        PACK.templates["hypothesis"],
        rationale="Closed-form arithmetic removes the loop.",
        assumptions=[
            "N is an integer greater than one",
            "Metric counts additions, not wall-clock latency",
        ],
        expected_observation={"operations_b_lt_a": True},
    )
    print("Hypothesis registered", hypothesis.hypothesis_id)
    run = await runtime.start_run(eid, config={"N": 2000}, seed=0)
    active = [run.run_id]
    actions = [
        Action(
            tool="environment",
            operation="exec",
            arguments={
                "command": {
                    "argv": ["python", "benchmark.py"],
                    "files": {
                        name: (HERE / name).read_text()
                        for name in ("algorithms.py", "benchmark.py", "test_algorithms.py")
                    },
                    "collect": ["metrics.json"],
                },
                "artifact_types": {"metrics.json": "metric"},
            },
            expected_effect="Write deterministic operation counts and equal computed sums",
        ),
        Action(
            tool="environment",
            operation="exec",
            arguments={"command": {"argv": PACK.templates["success_criteria"]["test"]["argv"]}},
            expected_effect="Independently execute unit tests",
        ),
    ]

    async def next_action(state):
        return actions.pop(0)

    external_agent = ExternalAgentBackend(next_action)
    try:
        benchmark = await runtime.execute(
            eid,
            await external_agent.next_action(await runtime.get_state(eid, run.run_id)),
            run_id=run.run_id,
        )
        tests = await runtime.execute(
            eid,
            await external_agent.next_action(await runtime.get_state(eid, run.run_id)),
            run_id=run.run_id,
        )
        assert benchmark.observation.exit_code == 0, benchmark.observation.stderr
        assert tests.observation.exit_code == 0, tests.observation.stderr
        print("Run #1 executed")
        print("Artifacts stored")
        metric = await runtime.verify(
            eid,
            run.run_id,
            hypothesis.hypothesis_id,
            "metric",
            [benchmark.observation.structured_output["collected"]["metrics.json"]],
        )
        test_evidence = await runtime.verify(
            eid,
            run.run_id,
            hypothesis.hypothesis_id,
            "test",
            [tests.observation.structured_output["log_artifact_id"]],
        )
        print("Evidence generated")
        claim = await runtime.create_claim(
            eid,
            "For N=2000, B uses fewer additions and computes the same sum as A.",
            [metric.evidence_id, test_evidence.evidence_id],
        )
        assert claim.status == "SUPPORTED"
        print("Claim verified (SUPPORTED)")
        await runtime.finish_run(run.run_id)
        active.remove(run.run_id)
        original = (await runtime.get_trajectory(run.trajectory_id)).model_dump_json()
        print("Trajectory recorded")
        replayed = await runtime.replay(run.run_id)
        assert replayed.matched
        print("Replay succeeded", replayed.replay_run_id)
        branch = await runtime.branch(run.run_id, from_step=1, config={"N": 4000})
        active.append(branch.run_id)
        print("Branch created from step 1", branch.run_id)
        alternative = await runtime.execute(
            eid,
            Action(
                tool="environment",
                operation="exec",
                arguments={
                    "command": {"argv": ["python", "benchmark.py"], "collect": ["metrics.json"]},
                    "artifact_types": {"metrics.json": "metric"},
                },
                expected_effect="Re-run inherited code with N=4000",
            ),
            run_id=branch.run_id,
        )
        assert alternative.observation.exit_code == 0, alternative.observation.stderr
        print("Run #2 executed")
        changed = await runtime.verify(
            eid,
            branch.run_id,
            hypothesis.hypothesis_id,
            "metric",
            [alternative.observation.structured_output["collected"]["metrics.json"]],
        )
        branch_claim = await runtime.create_claim(
            eid, "For N=4000, B still uses fewer additions.", [changed.evidence_id]
        )
        await link_counterfactual(
            runtime,
            benchmark.step_id,
            alternative.step_id,
            "state_restore",
            modified_context_refs=[branch.environment_snapshot_id],
        )
        await runtime.finish_run(branch.run_id)
        active.remove(branch.run_id)
        assert original == (await runtime.get_trajectory(run.trajectory_id)).model_dump_json()
        graph = await runtime.get_provenance(claim.claim_id)
        kinds = {node["kind"] for node in graph["nodes"].values()}
        assert {
            "Experiment",
            "Hypothesis",
            "Run",
            "ActionEvent",
            "ObservationEvent",
            "Artifact",
            "Evidence",
            "Claim",
        } <= kinds
        print("Provenance graph validated")
        await runtime.finish_experiment(eid)
        export = await runtime.export(
            eid,
            output_root or Path(os.getenv("EXPORT_ROOT", str(ROOT / "outputs"))) / "experiments",
        )
        print("Exports generated:")
        for name in export["files"]:
            print(" ", name)
        summary = {
            "schema_version": "2.0",
            "experiment_id": eid,
            "hypothesis_id": hypothesis.hypothesis_id,
            "run_id": run.run_id,
            "trajectory_id": run.trajectory_id,
            "replay": replayed.model_dump(mode="json"),
            "branch_run_id": branch.run_id,
            "original_trajectory_unchanged": True,
            "claim": claim.model_dump(mode="json"),
            "branch_claim": branch_claim.model_dump(mode="json"),
            "original_metrics": json.loads(
                await runtime.artifact_bytes(metric.source_artifact_ids[0])
            ),
            "branch_metrics": json.loads(
                await runtime.artifact_bytes(changed.source_artifact_ids[0])
            ),
            "provenance_node_count": len(graph["nodes"]),
            "export": export,
        }
        (Path(export["directory"]) / "summary.json").write_text(json.dumps(summary, indent=2))
        print("Summary:", str(Path(export["directory"]) / "summary.json"))
        return summary
    finally:
        for run_id in active:
            await runtime.finish_run(run_id)


if __name__ == "__main__":
    asyncio.run(run_demo())
