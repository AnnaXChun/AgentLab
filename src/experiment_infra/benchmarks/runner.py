"""External, sequential agent loop. Atomic task journals, no retries or recovery actions."""

import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter

import httpx

from experiment_infra.agents.openai_compatible import (
    AgentFormatError,
    ContextBudgetExceeded,
    OpenAICompatibleAgentBackend,
)
from experiment_infra.benchmarks.evaluation import EvaluationGateway, evaluate_and_record
from experiment_infra.core.schema import Artifact, DecisionContext, Evidence, Run, Trajectory
from packages.core.models.schemas import EventType as T
from packages.core.models.schemas import now

FAILURE_TYPES = {
    "AGENT_REASONING_FAILURE",
    "TOOL_EXECUTION_FAILURE",
    "ENVIRONMENT_FAILURE",
    "FORMAT_FAILURE",
    "TIMEOUT",
    "BUDGET_EXCEEDED",
    "VERIFIER_FAILURE",
    "UNKNOWN",
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def failure_type(exc, stage):
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or getattr(exc, "timed_out", False):
        return "TIMEOUT"
    if isinstance(exc, AgentFormatError):
        return "FORMAT_FAILURE"
    if isinstance(exc, ContextBudgetExceeded):
        return "BUDGET_EXCEEDED"
    return {
        "environment": "ENVIRONMENT_FAILURE",
        "execute": "TOOL_EXECUTION_FAILURE",
        "verify": "VERIFIER_FAILURE",
    }.get(stage, "UNKNOWN")


def summarize(rows):
    count = len(rows)

    def ratio(n):
        return n / count if count else None

    tokens = [r["total_tokens"] for r in rows if r.get("total_tokens") is not None]
    outputs = sum(r.get("outputs_expected", 0) for r in rows)
    results = sum(r.get("native_result") is not None for r in rows)
    replay = [r["replay"] for r in rows if r.get("replay")]
    return {
        "benchmark": rows[0]["benchmark"] if rows else None,
        "model": rows[0]["model_name"] if rows else None,
        "tasks_attempted": count,
        "tasks_successful": sum(r["success"] for r in rows),
        "success_rate": ratio(sum(r["success"] for r in rows)),
        "native_results_available": results,
        "valid_trajectories": sum(r.get("valid_trajectory", False) for r in rows),
        "valid_trajectory_rate": ratio(sum(r.get("valid_trajectory", False) for r in rows)),
        "replay_eligible": sum(r.get("replay_eligible", False) for r in rows),
        "replay_eligible_rate": ratio(sum(r.get("replay_eligible", False) for r in rows)),
        "artifact_coverage": sum(r.get("outputs_stored", 0) for r in rows) / outputs
        if outputs
        else None,
        "evidence_coverage": sum(bool(r.get("evidence_id")) for r in rows) / results
        if results
        else None,
        "avg_steps": sum(r.get("steps", 0) for r in rows) / count if count else None,
        "avg_tokens": sum(tokens) / len(tokens) if tokens else None,
        "token_coverage": len(tokens) / count if count else None,
        "avg_runtime": sum(r.get("runtime_seconds", 0) for r in rows) / count if count else None,
        "failure_types": dict(
            Counter(r["failure"]["failure_type"] for r in rows if r.get("failure"))
        ),
        "replays_attempted": len(replay),
        "replay_success_rate": sum(r["completed"] for r in replay) / len(replay)
        if replay
        else None,
        "replay_match_rate": sum(r.get("matched", False) for r in replay) / len(replay)
        if replay
        else None,
    }


def model_rl_rows(row, directory):
    """Project canonical transitions onto the driver's model-action sequence.

    Native evaluator actions stay in the canonical audit bundle. Their inputs
    and episode boundaries are not policy states or model-sequence boundaries.
    """
    directory = Path(directory)
    model_steps = set(row["model_step_ids"])
    transitions = [
        value
        for line in (directory / "rl_transition.jsonl").read_text().splitlines()
        if (value := json.loads(line))["step_id"] in model_steps
    ]
    contexts_path = directory / "decision_contexts.jsonl"
    contexts = {}
    if contexts_path.exists():
        for line in contexts_path.read_text().splitlines():
            context = DecisionContext.model_validate_json(line)
            if context.run_id == row["run_id"]:
                contexts[context.context_id] = context
    context_ids = row.get("decision_context_ids", [])
    following = dict(zip(context_ids, context_ids[1:]))
    for index, value in enumerate(transitions):
        next_context = contexts.get(following.get(value.get("decision_context_id")))
        # A next input must have been captured after this exact action, not an
        # evaluator or unrelated run, and is never reconstructed from execution.
        if next_context and next_context.last_step_id != value["step_id"]:
            next_context = None
        boundary = row.get("status") == "complete" and index == len(transitions) - 1
        value.update(
            next_state=next_context.payload if next_context else None,
            next_context_coverage=next_context.coverage if next_context else "unknown",
            next_decision_context_id=next_context.context_id if next_context else None,
            boundary_scope="model_action_sequence",
            episode_boundary=boundary,
            terminal=boundary,
            terminated=False if not boundary else None,
            truncated=False if not boundary else None,
            termination_reason=None,
        )
        if boundary:
            termination = row.get("termination")
            execution = value["observation"].get("structured_output", {}).get("execution", {})
            if isinstance(execution, dict) and execution.get("timed_out") is True:
                value.update(
                    terminated=False, truncated=True, termination_reason="execution_timeout"
                )
            elif termination == "max_steps":
                value.update(terminated=False, truncated=True, termination_reason="step_budget")
            elif (
                termination == "agent_stop"
                and (row.get("pending_action") or {}).get("operation") == "finish"
                and next_context
                and row["pending_action"].get("decision_context_id") == next_context.context_id
            ):
                # A deliberate policy finish does not establish a natural MDP terminal.
                value["termination_reason"] = "agent_finish"
            else:
                value["termination_reason"] = termination or "unknown"
        yield value


class BenchmarkRunner:
    def __init__(
        self,
        runtime,
        adapter,
        agent_factory,
        output_dir,
        *,
        model,
        seed=0,
        temperature=0,
        max_steps=30,
        context_limit=120000,
        replay_samples=1,
        run_kind="model_baseline",
    ):
        self.runtime, self.adapter, self.agent_factory = runtime, adapter, agent_factory
        self.output = Path(output_dir).resolve()
        self.seed, self.max_steps, self.replay_samples = seed, max_steps, replay_samples
        self.metadata = {
            "benchmark_name": adapter.name,
            "benchmark_version": adapter.version,
            "model_name": model,
            "model_backend": "openai-compatible"
            if run_kind == "model_baseline"
            else "reference-fixture",
            "temperature": temperature,
            "seed": seed,
            "max_steps": max_steps,
            "context_limit": context_limit,
            "context_limit_unit": "utf8_bytes",
            "runtime_version": importlib.metadata.version("scientific-harness"),
            "git_commit": git_commit(),
            "run_kind": run_kind,
            "protocol": getattr(adapter, "protocol", None),
        }
        self.sampled = Counter()

    async def run(self, tasks, *, resume=False):
        import fcntl

        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = {"metadata": self.metadata, "tasks": [(t.task_id, t.split) for t in tasks]}
            # JSON conversion makes tuple/list comparison consistent.
            manifest = json.loads(json.dumps(manifest))
            path = self.output / "batch.json"
            if path.exists():
                old = json.loads(path.read_text())
                if not resume:
                    raise ValueError(
                        "Output already contains a batch; use --resume or a new directory"
                    )
                if old != manifest:
                    raise ValueError("Resume configuration/task set differs from original batch")
            else:
                atomic_json(path, manifest)
            rows = []
            for task in tasks:
                path = (
                    self.output
                    / "tasks"
                    / (hashlib.sha256(task.task_id.encode()).hexdigest() + ".json")
                )
                if path.exists():
                    row = json.loads(path.read_text())
                    if row["status"] != "complete":
                        # Do not re-run an interrupted model/action. Preserve vanilla failure data.
                        row["failure"] = self.failure("UNKNOWN", "interrupted", row, None)
                        row["failure"]["message"] = "Process interrupted; no automatic retry"
                        await self.finalize(row)
                        row["status"] = "complete"
                        atomic_json(path, row)
                    rows.append(row)
                    if row.get("replay"):
                        self.sampled[bool(row["success"])] += 1
                    continue
                try:
                    row = await self.run_task(task, path)
                except Exception as exc:
                    # Includes adapter cleanup/client-close/export bugs: preserve task isolation.
                    row = json.loads(path.read_text())
                    row["failure"] = self.failure("UNKNOWN", "driver", row, None)
                    row["failure"]["error_class"] = type(exc).__name__
                    await self.finalize(row)
                    row["status"] = "complete"
                    atomic_json(path, row)
                rows.append(row)
                self.export_summary(rows)
            self.export_summary(rows)
            return rows

    def failure(self, kind, stage, row, state, action=None, step=None):
        assert kind in FAILURE_TYPES
        if step is not None:
            state, action = step.state_before, step.action
        return {
            "benchmark": self.adapter.name,
            "task_id": row["task_id"],
            "experiment_id": row.get("experiment_id"),
            "run_id": row.get("run_id"),
            "trajectory_id": row.get("trajectory_id"),
            "failure_type": kind,
            "failure_stage": stage,
            "failed_step_id": step.step_id if step else None,
            "snapshot_id": row.get("last_valid_snapshot"),
            "last_snapshot_id": row.get("last_valid_snapshot"),
            "last_valid_snapshot": row.get("last_valid_snapshot"),
            "failed_action": action.model_dump(mode="json")
            if action
            else row.get("pending_action"),
            "state_before_failure": state or row.get("state_before_action"),
            "context_refs": action.context_refs if action else [],
            "artifact_refs": (state or row.get("state_before_action") or {}).get(
                "artifact_ids", []
            ),
        }

    async def run_task(self, task, journal):
        started, stage, state, action, step, run, agent = (
            perf_counter(),
            "environment",
            None,
            None,
            None,
            None,
            None,
        )
        rt = self.runtime
        row = {
            **self.metadata,
            "benchmark": self.adapter.name,
            "task_id": task.task_id,
            "split": task.split,
            "status": "started",
            "success": False,
            "failure": None,
            "experiment_id": None,
            "run_id": None,
            "trajectory_id": None,
            "last_valid_snapshot": None,
            "native_result": None,
            "steps": 0,
            "model_step_ids": [],
            "total_tokens": None,
            "model_calls": [],
            "decision_context_ids": [],
            "pending_decision_context_id": None,
            "environment_fingerprint": None,
            "outputs_expected": 0,
            "outputs_stored": 0,
        }
        atomic_json(journal, row)
        termination = "agent_stop"
        try:
            experiment = await self.adapter.create_experiment(rt, task, self.metadata["model_name"])
            row["experiment_id"] = experiment.experiment_id
            atomic_json(journal, row)
            hypothesis = await rt.register_hypothesis(
                experiment.experiment_id, "Task can be completed successfully"
            )
            run = await rt.start_run(
                experiment.experiment_id,
                config={**self.metadata, "task_id": task.task_id, "split": task.split},
                seed=self.seed,
                code_version=self.metadata["git_commit"],
            )
            row.update(
                run_id=run.run_id,
                trajectory_id=run.trajectory_id,
                last_valid_snapshot=run.environment_snapshot_id,
            )
            row["environment_fingerprint"] = (await rt.environment.observe(run.environment_handle))[
                "fingerprint"
            ]
            atomic_json(journal, row)
            await self.adapter.prepare_environment(rt, task, run)
            agent = self.agent_factory(task)
            history = []
            for _ in range(self.max_steps):
                stage, step, action = "agent", None, None
                state = await rt.get_state(run.experiment_id, run.run_id)
                context = {
                    "task": task.goal,
                    "state": state,
                    "context": self.adapter.agent_context(task),
                    "available_tools": self.adapter.available_tools(task),
                    "history": history,
                }
                prompt = await rt._artifact(
                    run,
                    json.dumps(context).encode(),
                    "config",
                    metadata={"role": "model_input"},
                    mime_type="application/json",
                )
                row.update(
                    state_before_action=state,
                    pending_action=None,
                    pending_decision_context_id=None,
                )
                atomic_json(journal, row)
                # Only the built-in backend guarantees that this prepared body
                # is the complete request. Custom backends (including subclasses)
                # may add hidden inputs, so capture their supplied state as unknown.
                prepared_request = type(agent) is OpenAICompatibleAgentBackend
                payload = agent.prepare_request(context) if prepared_request else context
                decision_context = await rt.capture_decision_context(
                    run.experiment_id,
                    run.run_id,
                    payload,
                    coverage="complete" if prepared_request else "unknown",
                    source_refs=[prompt.artifact_id],
                )
                row["decision_context_ids"].append(decision_context.context_id)
                row.update(
                    state_before_action=state,
                    pending_action=None,
                    pending_decision_context_id=decision_context.context_id,
                )
                atomic_json(journal, row)
                try:
                    if prepared_request:
                        action = await agent.next_action_from_request(decision_context.payload)
                    else:
                        action = await agent.next_action(decision_context.payload)
                finally:
                    call = getattr(agent, "last_call", None)
                    if call:
                        call = {**call, "decision_context_id": decision_context.context_id}
                        row["model_calls"].append(call)
                        await rt._artifact(
                            run,
                            json.dumps(call).encode(),
                            "log",
                            metadata={"role": "model_call"},
                            mime_type="application/json",
                        )
                try:
                    self.adapter.validate_action(task, action)
                except Exception as exc:
                    raise AgentFormatError("Action violates available tool schema") from exc
                action = action.model_copy(
                    update={
                        "context_refs": [*action.context_refs, prompt.artifact_id],
                        "decision_context_id": decision_context.context_id,
                    }
                )
                row["pending_action"] = action.model_dump(mode="json")
                atomic_json(journal, row)
                if action.operation == "finish":
                    break
                stage = "execute"
                step = await rt.execute(
                    run.experiment_id,
                    action,
                    run_id=run.run_id,
                    token_usage=(getattr(agent, "last_call", None) or {}).get("usage"),
                )
                row["steps"] += 1
                row["model_step_ids"].append(step.step_id)
                if step.state_after.get("snapshot_id"):
                    row["last_valid_snapshot"] = step.state_after["snapshot_id"]
                atomic_json(journal, row)
                history.append(
                    {
                        "action": action.model_dump(mode="json"),
                        "observation": step.observation.model_dump(mode="json"),
                    }
                )
                if step.observation.exit_code != 0 or "snapshot_error" in step.state_after:
                    kind = (
                        "TIMEOUT"
                        if step.observation.structured_output.get("execution", {}).get("timed_out")
                        else "ENVIRONMENT_FAILURE"
                        if "snapshot_error" in step.state_after
                        else "TOOL_EXECUTION_FAILURE"
                    )
                    row["failure"] = self.failure(kind, stage, row, state, action, step)
                    termination = "too_many_errors"
                    break
            else:
                row["failure"] = self.failure("BUDGET_EXCEEDED", "agent", row, state, action)
                termination = "max_steps"
        except Exception as exc:
            row["failure"] = self.failure(failure_type(exc, stage), stage, row, state, action, step)
            # Do not persist provider response bodies, URLs, keys, or malformed reasoning text.
            row["failure"]["error_class"] = type(exc).__name__
            termination = "agent_error" if stage == "agent" else "infrastructure_error"
        if run:
            try:
                stage = "verify"
                native, evidence, claim, _ = await evaluate_and_record(
                    rt, self.adapter, task, run, hypothesis.hypothesis_id, termination
                )
                row.update(
                    native_result=native,
                    evidence_id=evidence.evidence_id,
                    claim_id=claim.claim_id,
                    success=native["success"],
                )
                if not row["success"] and row["failure"] is None:
                    row["failure"] = self.failure(
                        "AGENT_REASONING_FAILURE", "native_evaluation", row, state, action
                    )
            except Exception as exc:
                if row["failure"] is None:
                    row["failure"] = self.failure(
                        failure_type(exc, "verify"),
                        "verify",
                        row,
                        state,
                        step=getattr(exc, "step", None),
                    )
                row["evaluation_error"] = type(exc).__name__
            row["termination"] = termination
        if agent and hasattr(agent, "close"):
            try:
                await agent.close()
            except Exception as exc:
                row["client_close_error"] = type(exc).__name__
        calls = row["model_calls"]
        if calls and all((c.get("usage") or {}).get("total_tokens") is not None for c in calls):
            row["total_tokens"] = sum(c["usage"]["total_tokens"] for c in calls)
        await self.finalize(row)
        if row.get("replay_eligible") and self.sampled[row["success"]] < self.replay_samples:
            self.sampled[row["success"]] += 1
            await self.replay(task, row)
            await self.export_task(row)
        try:
            await self.adapter.cleanup(task)
        except Exception as exc:
            row["cleanup_error"] = type(exc).__name__
        row.update(status="complete", runtime_seconds=perf_counter() - started)
        atomic_json(journal, row)
        return row

    async def finalize(self, row):
        rt, eid = self.runtime, row.get("experiment_id")
        if not eid:
            return
        try:
            objects = await rt.ledger.objects(eid)
            runs = [o for o in objects.values() if isinstance(o, Run)]
            if not row.get("run_id") and runs:
                row.update(run_id=runs[0].run_id, trajectory_id=runs[0].trajectory_id)
            for run in runs:
                if run.status == "RUNNING":
                    await rt.finish_run(run.run_id)
            if row.get("run_id"):
                run = await rt.ledger.get(row["run_id"], "Run")
                if row.get("failure") and run.status != "FAILED":
                    # An API/format/budget failure happens outside execute(), so finish_run
                    # cannot infer it from exit codes. Record it with the existing event type.
                    await rt.ledger.record(
                        eid,
                        T.RUN_FINISHED,
                        [run.model_copy(update={"status": "FAILED", "finished_at": now()})],
                        output={"benchmark_failure": row["failure"]},
                    )
                trajectory = await rt.get_trajectory(row["trajectory_id"])
                Trajectory.model_validate_json(trajectory.model_dump_json())
                row["valid_trajectory"] = True
                row["replay_eligible"] = (
                    bool(trajectory.steps)
                    and bool(run.environment_snapshot_id)
                    and all(s.state_after.get("snapshot_id") for s in trajectory.steps)
                )
                if trajectory.steps:
                    row["last_valid_snapshot"] = next(
                        (
                            s.state_after["snapshot_id"]
                            for s in reversed(trajectory.steps)
                            if s.state_after.get("snapshot_id")
                        ),
                        row.get("last_valid_snapshot"),
                    )
                # Every regular file in an execution snapshot is covered by immutable CAS,
                # including files not named in command.collect and failed-command outputs.
                outputs = [
                    o
                    for o in objects.values()
                    if isinstance(o, Artifact) and o.producer_run_id == run.run_id
                ]
                if row.get("failure"):
                    row["failure"].update(
                        run_id=row["run_id"],
                        trajectory_id=row["trajectory_id"],
                        snapshot_id=row["last_valid_snapshot"],
                        last_snapshot_id=row["last_valid_snapshot"],
                        last_valid_snapshot=row["last_valid_snapshot"],
                    )
                row["artifacts_stored"] = len(outputs)
                row["evidence_stored"] = sum(isinstance(o, Evidence) for o in objects.values())
                row["outputs_expected"] = sum(
                    len(s.observation.environment_diff.get(k, []))
                    for s in trajectory.steps
                    for k in ("added", "modified")
                )
                row["outputs_stored"] = row["outputs_expected"] if row["replay_eligible"] else 0
                if self.adapter.name == "tau2":
                    row["outputs_expected"] = len(trajectory.steps)
                    row["outputs_stored"] = sum(
                        bool(s.state_after.get("snapshot_id")) for s in trajectory.steps
                    )
            await rt.finish_experiment(eid)
            await self.export_task(row)
        except Exception as exc:
            row["finalization_error"] = type(exc).__name__
            row.setdefault("valid_trajectory", False)
            row.setdefault("replay_eligible", False)
            if row.get("failure") is None:
                row["failure"] = self.failure("ENVIRONMENT_FAILURE", "finalize", row, None)

    async def replay(self, task, row):
        rt, old_gateway = self.runtime, self.runtime.gateway

        async def evaluate_replay():
            runs = [
                o
                for o in (await rt.ledger.objects(row["experiment_id"])).values()
                if isinstance(o, Run) and o.parent_run_id == row["run_id"] and o.status == "RUNNING"
            ]
            if len(runs) != 1:
                raise ValueError("Replay run is ambiguous")
            return await self.adapter.evaluate(rt, task, runs[0], row["termination"])

        rt.gateway = EvaluationGateway(evaluate_replay)
        try:
            replay = await rt.replay(row["run_id"])
            row["replay"] = {
                "completed": True,
                "kind": self.adapter.replay_kind,
                **replay.model_dump(mode="json"),
            }
        except Exception as exc:
            row["replay"] = {
                "completed": False,
                "matched": False,
                "kind": self.adapter.replay_kind,
                "error_class": type(exc).__name__,
            }
        finally:
            rt.gateway = old_gateway
            await rt.finish_experiment(row["experiment_id"])

    async def export_task(self, row):
        exported = await self.runtime.export(row["experiment_id"], self.output / "exports")
        row["export_directory"] = exported["directory"]

    def export_summary(self, rows):
        summary = summarize(rows)
        summary["run_kind"] = self.metadata["run_kind"]
        atomic_json(self.output / "benchmark_summary.json", summary)
        with (self.output / "benchmark_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, dict) else v for k, v in summary.items()}
            )
        for filename, values in (
            ("benchmark_runs.jsonl", rows),
            ("failures.jsonl", [r["failure"] for r in rows if r.get("failure")]),
        ):
            (self.output / filename).write_text(
                "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values)
            )
        # Canonical bundles retain evaluator steps; training rows here include only model actions.
        for name in (
            "trajectory.jsonl",
            "artifacts.jsonl",
            "evidence.jsonl",
            "sft.jsonl",
            "rl_transition.jsonl",
        ):
            with (self.output / name).open("w") as stream:
                for row in rows:
                    directory = row.get("export_directory")
                    if not directory:
                        continue
                    if name == "rl_transition.jsonl":
                        if self.metadata["run_kind"] != "environment_calibration":
                            for value in model_rl_rows(row, directory):
                                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
                        continue
                    for line in (Path(directory) / name).read_text().splitlines():
                        value = json.loads(line)
                        if name in {"sft.jsonl", "rl_transition.jsonl"} and (
                            self.metadata["run_kind"] == "environment_calibration"
                            or value.get("step_id") not in row["model_step_ids"]
                        ):
                            continue
                        stream.write(line + "\n")
