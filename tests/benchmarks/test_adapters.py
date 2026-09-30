import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiment_infra.benchmarks import scienceagentbench as sab
from experiment_infra.benchmarks.runner import failure_type


def test_science_adapter_mapping_keeps_evaluator_and_gold_out_of_agent_context(
    tmp_path, monkeypatch
):
    rows = [
        {
            "instance_id": 7,
            "task_inst": "Analyze observations",
            "gold_program_name": "secret.py",
            "eval_script_name": "hidden_eval.py",
            "output_fname": "pred_results/result.csv",
            "domain_knowledge": "public context",
            "dataset_preview": "x,y",
            "dataset_folder_tree": "data.csv",
        }
    ]
    annotations = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist(rows), annotations)
    monkeypatch.setattr(
        sab, "SAB_ANNOTATION_SHA256", hashlib.sha256(annotations.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(sab.subprocess, "check_output", lambda *a, **k: sab.SAB_COMMIT)
    monkeypatch.setattr(sab, "ScienceDockerEnvironment", lambda data: object())
    adapter = sab.ScienceAgentBenchAdapter(tmp_path, tmp_path, annotations)
    task = adapter.load_task("7", "verified")
    assert task.goal == rows[0]["task_inst"] and task.split == "verified"
    assert "secret.py" not in json.dumps(adapter.agent_context(task))
    assert "hidden_eval" not in json.dumps(adapter.available_tools(task))
    with pytest.raises(ValueError, match="split"):
        adapter.list_tasks("not-official")
    with pytest.raises(ValueError, match="Unknown task"):
        adapter.load_task("999", "verified")
    with pytest.raises(FileNotFoundError, match="datasets"):
        adapter.environment_spec(task)
    assert str(sab.native_result_path(tmp_path, "run", "7")).endswith(
        "logs/run_evaluation/run/7/output/result.json"
    )


def test_failure_signals():
    assert failure_type(TimeoutError(), "agent") == "TIMEOUT"
    assert failure_type(RuntimeError(), "environment") == "ENVIRONMENT_FAILURE"
    assert failure_type(RuntimeError(), "verify") == "VERIFIER_FAILURE"


def test_native_inputs_are_read_only_but_results_remain_writable():
    from experiment_infra.benchmarks.sab_worker import protect_native_inputs

    volumes = {
        "data": {"bind": "/testbed/benchmark"},
        "code": {"bind": "/testbed/pred_programs", "mode": "rw"},
        "scorer": {"bind": "/testbed/compute_scores.py", "mode": "rw"},
        "output": {"bind": "/testbed/instance_path", "mode": "rw"},
    }
    call = protect_native_inputs(lambda self, **kwargs: kwargs)
    mounted = call(None, volumes=volumes)["volumes"]
    assert all(mounted[key]["mode"] == "ro" for key in ("data", "code", "scorer"))
    assert mounted["output"]["mode"] == "rw"
    assert "mode" not in volumes["data"]


async def test_submission_transport_error_is_not_a_missing_program():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    env = object.__new__(sab.ScienceDockerEnvironment)
    env.backend = SimpleNamespace(_python=AsyncMock(return_value=b"0\n"))
    assert await env.submission("test-handle") is None
    env.backend._python.side_effect = RuntimeError("Docker unavailable")
    with pytest.raises(RuntimeError, match="Docker unavailable"):
        await env.submission("test-handle")
