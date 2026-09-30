"""Official tau2 telecom SOLO protocol; native tools and ALL evaluator unchanged."""

import base64
import hashlib
import importlib.metadata
import json
import os
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from experiment_infra.benchmarks.base import AdapterBase, BenchmarkTask
from experiment_infra.core.schema import EnvironmentSpec
from packages.sandbox.base import ExecutionResult

TAU_COMMIT = "5bfa7e37b36656b37dc6d022156be6563c1007f3"


def setup_tau(root):
    root = Path(root).resolve()
    if not (root / "data/tau2/domains/telecom/tasks.json").is_file():
        raise FileNotFoundError("Install the pinned tau2 checkout; set TAU2_ROOT")
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != TAU_COMMIT:
        raise ValueError("tau2 checkout differs from supported pinned revision")
    os.environ["TAU2_DATA_DIR"] = str(root / "data")
    import tau2
    from loguru import logger

    logger.disable("tau2")
    if importlib.metadata.version("tau2") != "1.0.1":
        raise ValueError("Install tau2 1.0.1 from the pinned checkout")
    # Data stays outside the package. Pin both the checkout and installed source tree.
    installed = Path(tau2.__file__).parent
    for source in (root / "src/tau2").rglob("*.py"):
        if source.read_bytes() != (installed / source.relative_to(root / "src/tau2")).read_bytes():
            raise ValueError("Installed tau2 differs from checkout")
    return root


class Tau2Adapter(AdapterBase):
    name = "tau2"
    version = TAU_COMMIT
    protocol = "telecom-solo"

    def __init__(self, root):
        self.root = setup_tau(root)
        self.environment = Tau2Environment(self)
        self._native_tasks = {}
        self._env = None
        hasher = hashlib.sha256()
        for path in sorted((self.root / "data/tau2/domains/telecom").rglob("*")):
            if path.is_file():
                hasher.update(str(path.relative_to(self.root)).encode())
                hasher.update(path.read_bytes())
        self.fingerprint = hashlib.sha256((self.version + hasher.hexdigest()).encode()).hexdigest()

    def list_tasks(self, split):
        from tau2.domains.telecom.environment import get_tasks

        tasks = get_tasks(split)
        # Fail rather than silently filter or change the denominator.
        if any(t.ticket is None or t.evaluation_criteria is None for t in tasks):
            raise ValueError("Selected split contains tasks unsupported by official solo mode")
        self._native_tasks.update({t.id: t for t in tasks})
        return [BenchmarkTask(t.id, t.ticket, split) for t in tasks]

    def native_task(self, task_id, split):
        if task_id not in self._native_tasks:
            self.list_tasks(split)
        return self._native_tasks[task_id]

    def environment_spec(self, task):
        return EnvironmentSpec(
            backend="tau2",
            image=None,
            settings={
                "task_id": task.task_id,
                "split": task.split,
                "version": self.version,
                "protocol": self.protocol,
            },
        )

    def available_tools(self, task):
        from tau2.domains.telecom.environment import get_environment

        env = get_environment(solo_mode=True)
        native = self.native_task(task.task_id, task.split)
        tools = env.get_tools() + env.get_user_tools(include=native.user_tools)
        return {
            "exec_argv": ["tau2", "<tool_name>", "<JSON arguments object>"],
            "tools": [t.openai_schema for t in tools],
            "finish": True,
            "files": "not available; native tool interaction only",
        }

    def agent_context(self, task):
        from tau2.domains.telecom.environment import get_environment

        native = self.native_task(task.task_id, task.split)
        history = native.initial_state.message_history if native.initial_state else []
        return {
            "policy": get_environment(solo_mode=True).get_policy(),
            "initial_messages": [m.model_dump(mode="json") for m in history or []],
            "protocol": self.protocol,
        }

    def validate_action(self, task, action):
        if action.operation == "finish":
            return
        from jsonschema import validate

        command = action.arguments["command"]
        argv = command["argv"]
        if (
            len(argv) != 3
            or argv[0] != "tau2"
            or command.get("files")
            or command.get("environment")
        ):
            raise ValueError("tau2 only accepts native tool argv, without files or environment")
        schemas = {
            t["function"]["name"]: t["function"]["parameters"]
            for t in self.available_tools(task)["tools"]
        }
        if argv[1] not in schemas:
            raise ValueError("Unavailable native tool")
        validate(json.loads(argv[2]), schemas[argv[1]])

    async def evaluate(self, runtime, task, run, termination):
        from tau2.data_model.simulation import SimulationRun
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        session = self.environment.sessions[run.environment_handle]
        native = self.native_task(task.task_id, task.split)
        ended = datetime.now(timezone.utc)
        sim = SimulationRun(
            id=run.run_id,
            task_id=task.task_id,
            start_time=run.started_at.isoformat(),
            end_time=ended.isoformat(),
            duration=(ended - run.started_at).total_seconds(),
            termination_reason=termination,
            messages=session["messages"],
            seed=run.seed,
        )
        result = evaluate_simulation(sim, native, EvaluationType.ALL, True, "telecom")
        return {
            "benchmark": self.name,
            "version": self.version,
            "metric": "reward",
            "score": result.reward,
            "success": result.reward == 1,
            "evaluator": "tau2.evaluator.evaluator.evaluate_simulation:ALL",
            "protocol": self.protocol,
            "native_result": result.model_dump(mode="json"),
        }


class Tau2Environment:
    """Isolated native DB per run. JSON checkpoints restore using official strict set_state.

    The adapter only dispatches allowlisted tools. No shell or arbitrary Python execution.
    """

    def __init__(self, adapter):
        self.adapter, self.sessions = adapter, {}

    async def create(self, spec):
        from tau2.domains.telecom.environment import get_environment

        if spec.backend != "tau2" or spec.settings["version"] != self.adapter.version:
            raise ValueError("Unsupported tau2 environment version")
        task = self.adapter.native_task(spec.settings["task_id"], spec.settings["split"])
        initial = task.initial_state
        messages = deepcopy(initial.message_history or []) if initial else []
        env = get_environment(solo_mode=True)
        env.set_state(
            initial.initialization_data if initial else None,
            initial.initialization_actions if initial else None,
            messages,
        )
        handle = str(uuid4())
        self.sessions[handle] = {"env": env, "messages": messages, "files": {}, "spec": spec}
        return handle

    async def upload(self, handle, path, data):
        self.sessions[handle]["files"][path] = base64.b64encode(data).decode()

    async def download(self, handle, path):
        return base64.b64decode(self.sessions[handle]["files"][path])

    async def execute(self, handle, command):
        from tau2.data_model.message import AssistantMessage, ToolCall

        session = self.sessions[handle]
        settings = session["spec"].settings
        task = self.adapter.load_task(settings["task_id"], settings["split"])
        from experiment_infra.core.schema import Action

        self.adapter.validate_action(
            task,
            Action(
                tool="environment",
                operation="exec",
                arguments={"command": command.model_dump(mode="json")},
            ),
        )
        name, arguments = command.argv[1], json.loads(command.argv[2])
        call = ToolCall(
            id=f"call_{len(session['messages'])}",
            name=name,
            arguments=arguments,
            requestor="assistant",
        )
        message = AssistantMessage(role="assistant", tool_calls=[call], timestamp=None)
        response = session["env"].get_response(call)
        response.timestamp = None
        session["messages"].extend([message, response])
        transcript = json.dumps([m.model_dump(mode="json") for m in session["messages"]])
        await self.upload(handle, "native_transcript.json", transcript.encode())
        return ExecutionResult(
            exit_code=1 if response.error else 0, stdout=response.content or "", stderr=""
        )

    async def observe(self, handle):
        return {
            "spec": self.sessions[handle]["spec"].model_dump(mode="json"),
            "fingerprint": self.adapter.fingerprint,
            "snapshot_mime_type": "application/json",
        }

    async def snapshot(self, handle):
        session = self.sessions[handle]
        return json.dumps(
            {
                "messages": [m.model_dump(mode="json") for m in session["messages"]],
                "files": session["files"],
                "fingerprint": self.adapter.fingerprint,
            },
            sort_keys=True,
        ).encode()

    async def restore(self, handle, snapshot):
        from pydantic import TypeAdapter
        from tau2.data_model.message import Message
        from tau2.domains.telecom.environment import get_environment

        data = json.loads(snapshot)
        if data["fingerprint"] != self.adapter.fingerprint:
            raise ValueError("tau2 checkpoint fingerprint mismatch")
        session = self.sessions[handle]
        settings = session["spec"].settings
        initial = self.adapter.native_task(settings["task_id"], settings["split"]).initial_state
        messages = TypeAdapter(list[Message]).validate_python(data["messages"])
        env = get_environment(solo_mode=True)
        env.set_state(
            initial.initialization_data if initial else None,
            initial.initialization_actions if initial else None,
            messages,
            strict=True,
        )
        session.update(env=env, messages=messages, files=data["files"])

    @staticmethod
    def diff(a, b):
        return {
            "state_changed": a != b,
            "before_sha256": hashlib.sha256(a).hexdigest(),
            "after_sha256": hashlib.sha256(b).hexdigest(),
        }

    async def destroy(self, handle):
        self.sessions.pop(handle, None)
