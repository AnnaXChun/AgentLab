from dataclasses import dataclass, field
from typing import Protocol

from experiment_infra.core.schema import EnvironmentSpec


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    goal: str
    split: str
    payload: dict = field(default_factory=dict, repr=False)  # Never put hidden criteria in prompts.


class BenchmarkAdapter(Protocol):
    name: str
    version: str
    replay_kind: str

    def list_tasks(self, split: str) -> list[BenchmarkTask]: ...
    def load_task(self, task_id: str, split: str) -> BenchmarkTask: ...
    def environment_spec(self, task: BenchmarkTask) -> EnvironmentSpec: ...
    async def prepare_environment(self, runtime, task, run): ...
    def available_tools(self, task): ...
    def agent_context(self, task): ...
    async def evaluate(self, runtime, task, run, termination): ...
    async def cleanup(self, task): ...


class AdapterBase:
    replay_kind = "deterministic_execution"

    def load_task(self, task_id, split):
        for task in self.list_tasks(split):
            if task.task_id == task_id:
                return task
        raise ValueError(f"Unknown task {task_id!r} in {split!r}")

    async def create_experiment(self, runtime, task, model):
        return await runtime.create_experiment(
            task.goal,
            environment_spec=self.environment_spec(task),
            agent_backend="openai-compatible/" + model,
            success_criteria={"native_evaluator": self.name, "version": self.version},
        )

    async def prepare_environment(self, runtime, task, run):
        return None

    async def cleanup(self, task):
        return None
