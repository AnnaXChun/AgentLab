import json
from types import SimpleNamespace

import pytest

from experiment_infra.agents.openai_compatible import AgentFormatError
from experiment_infra.benchmarks.base import AdapterBase, BenchmarkTask
from experiment_infra.benchmarks.evaluation import NativeResultVerifier
from experiment_infra.benchmarks.runner import BenchmarkRunner
from experiment_infra.core.schema import Action, EnvironmentSpec
from experiment_infra.runtime.executor import ExperimentRuntime
from packages.sandbox.base import ExecutionResult


class MemoryEnvironment:
    def __init__(self):
        self.workspaces, self.specs = {}, {}
        self.counter = 0

    async def create(self, spec):
        self.counter += 1
        handle = str(self.counter)
        self.workspaces[handle], self.specs[handle] = {}, spec
        return handle

    async def execute(self, handle, command):
        self.workspaces[handle]["output.json"] = json.dumps(
            {"success": command.argv == ["succeed"]}
        )
        return ExecutionResult(exit_code=0, stdout="executed", stderr="")

    async def upload(self, handle, path, data):
        self.workspaces[handle][path] = data.decode()

    async def download(self, handle, path):
        return self.workspaces[handle][path].encode()

    async def observe(self, handle):
        return {
            "spec": self.specs[handle].model_dump(mode="json"),
            "fingerprint": "test-v1",
            "snapshot_mime_type": "application/json",
        }

    async def snapshot(self, handle):
        return json.dumps(self.workspaces[handle], sort_keys=True).encode()

    async def restore(self, handle, snapshot):
        self.workspaces[handle] = json.loads(snapshot)

    def diff(self, a, b):
        return {"added": list(json.loads(b).keys() - json.loads(a).keys()), "modified": []}

    async def destroy(self, handle):
        self.workspaces.pop(handle, None)


class MockAdapter(AdapterBase):
    name, version = "mock", "test-only-v1"

    def __init__(self):
        self.environment = MemoryEnvironment()

    def list_tasks(self, split):
        return [BenchmarkTask(x, x, split) for x in ["pass", "crash", "fail"]]

    def environment_spec(self, task):
        return EnvironmentSpec(backend="test", image=None)

    def available_tools(self, task):
        return ["exec", "finish"]

    def agent_context(self, task):
        return {"public_task": task.task_id}

    def validate_action(self, task, action):
        assert action.operation in {"exec", "finish"}

    async def evaluate(self, runtime, task, run, termination):
        files = self.environment.workspaces[run.environment_handle]
        result = json.loads(files.get("output.json", '{"success":false}'))
        return {
            "benchmark": self.name,
            "version": self.version,
            "metric": "test_success",
            "score": int(result["success"]),
            "success": result["success"],
            "evaluator": "test-only",
        }


class TestAgent:
    __test__ = False
    last_call = None

    def __init__(self, task):
        self.task, self.n = task, 0

    async def next_action(self, state):
        if self.task.task_id == "crash":
            raise AgentFormatError("malformed")
        self.n += 1
        if self.n > 1:
            return Action(tool="environment", operation="finish")
        return Action(
            tool="environment",
            operation="exec",
            arguments={
                "command": {
                    "argv": ["succeed" if self.task.task_id == "pass" else "fail"],
                    "collect": ["output.json"],
                }
            },
        )


def runner(store, artifacts, tmp_path):
    adapter = MockAdapter()
    rt = ExperimentRuntime(
        SimpleNamespace(store=store, artifacts=artifacts), environment=adapter.environment
    )
    return BenchmarkRunner(
        rt, adapter, TestAgent, tmp_path, model="fixture", run_kind="test_fixture"
    ), adapter


async def test_batch_isolation_evidence_replay_and_exports(store, artifacts, tmp_path):
    driver, adapter = runner(store, artifacts, tmp_path)
    rows = await driver.run(adapter.list_tasks("test"))
    assert [r["success"] for r in rows] == [True, False, False]
    assert rows[1]["failure"]["failure_type"] == "FORMAT_FAILURE"
    assert rows[2]["failure"]["failure_type"] == "AGENT_REASONING_FAILURE"
    assert all(r.get("evidence_id") for r in rows)
    assert all(r.get("valid_trajectory") for r in rows)
    assert all(r["replay"]["matched"] for r in rows if r.get("replay"))
    assert not adapter.environment.workspaces
    summary = json.loads((tmp_path / "benchmark_summary.json").read_text())
    assert summary["tasks_attempted"] == 3 and summary["success_rate"] == 1 / 3
    assert summary["avg_tokens"] is None and summary["evidence_coverage"] == 1
    failures = [json.loads(x) for x in (tmp_path / "failures.jsonl").read_text().splitlines()]
    assert len(failures) == 2 and all(f["snapshot_id"] for f in failures)
    assert len((tmp_path / "sft.jsonl").read_text().splitlines()) == 2  # no verifier/replay actions
    claim = await driver.runtime.ledger.get(rows[0]["claim_id"], "Claim")
    assert claim.status == "SUPPORTED"
    graph = await driver.runtime.get_ancestors(claim.claim_id)
    assert rows[0]["run_id"] in graph
    assert rows[0]["evidence_id"] in graph


async def test_resume_skips_finished_and_interrupted_without_retry(store, artifacts, tmp_path):
    driver, adapter = runner(store, artifacts, tmp_path)
    rows = await driver.run(adapter.list_tasks("test"))
    before = adapter.environment.counter
    again = await driver.run(adapter.list_tasks("test"), resume=True)
    assert [r["run_id"] for r in rows] == [r["run_id"] for r in again]
    assert before == adapter.environment.counter
    journal = next(
        p
        for p in (tmp_path / "tasks").glob("*.json")
        if json.loads(p.read_text())["task_id"] == "fail"
    )
    interrupted = json.loads(journal.read_text())
    interrupted["status"] = "started"
    journal.write_text(json.dumps(interrupted))
    await driver.run(adapter.list_tasks("test"), resume=True)
    assert before == adapter.environment.counter
    assert json.loads(journal.read_text())["failure"]["failure_stage"] == "interrupted"
    with pytest.raises(ValueError, match="differs"):
        await driver.run(adapter.list_tasks("test")[:1], resume=True)


async def test_native_receipt_cannot_be_replaced(store, artifacts, tmp_path):
    driver, adapter = runner(store, artifacts, tmp_path)
    row = (await driver.run(adapter.list_tasks("test")[:1]))[0]
    objects = await driver.runtime.ledger.objects(row["experiment_id"])
    evidence = objects[row["evidence_id"]]
    artifact = objects[evidence.source_artifact_ids[0]]
    verifier = NativeResultVerifier("mock", artifact.sha256)
    with pytest.raises(ValueError, match="mismatch"):
        await verifier.verify(
            objects[row["experiment_id"]],
            objects[row["run_id"]],
            {artifact.artifact_id: (artifact, b'{"success":true}')},
            [],
        )


async def test_environment_and_verifier_failures_do_not_abort_batch(store, artifacts, tmp_path):
    driver, adapter = runner(store, artifacts, tmp_path)
    original_spec = adapter.environment_spec
    original_eval = adapter.evaluate

    def spec(task):
        if task.task_id == "crash":
            raise RuntimeError("environment is unavailable")
        return original_spec(task)

    async def evaluate(runtime, task, run, termination):
        if task.task_id == "fail":
            raise RuntimeError("official evaluator failed")
        return await original_eval(runtime, task, run, termination)

    adapter.environment_spec, adapter.evaluate = spec, evaluate
    rows = await driver.run(adapter.list_tasks("test"))
    assert rows[0]["success"]
    assert rows[1]["failure"]["failure_type"] == "ENVIRONMENT_FAILURE"
    assert rows[2]["failure"]["failure_type"] == "VERIFIER_FAILURE"
    assert rows[2]["native_result"] is None
    assert not rows[2].get("evidence_id")
    failure = rows[2]["failure"]
    assert failure["failed_step_id"]
    assert failure["failed_action"]["arguments"]["tool_name"] == "benchmark.native_evaluate"


def test_summary_table_rejects_mixed_protocols(tmp_path):
    from experiment_infra.benchmarks.report import write_summary_table

    directories = [tmp_path / "a", tmp_path / "b"]
    for directory, kind in zip(directories, ["environment_calibration", "model_baseline"]):
        directory.mkdir()
        (directory / "benchmark_summary.json").write_text(json.dumps({"run_kind": kind}))
    with pytest.raises(ValueError, match="Do not combine"):
        write_summary_table(directories, tmp_path)
    rows = write_summary_table(directories[:1], tmp_path)
    assert rows[0]["run_kind"] == "environment_calibration"


async def test_budget_and_usage(store, artifacts, tmp_path):
    driver, adapter = runner(store, artifacts, tmp_path)
    driver.max_steps = 1
    rows = await driver.run(adapter.list_tasks("test")[2:])
    assert rows[0]["failure"]["failure_type"] == "BUDGET_EXCEEDED"
    assert rows[0]["total_tokens"] is None
