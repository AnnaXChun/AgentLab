"""Environment calibration with reference actions, explicitly NOT a model baseline.

Uses official oracle actions/programs only to verify integration and native grading.
Never merge these outputs into training data or a model benchmark summary.
"""

import argparse
import asyncio
import json
from pathlib import Path

from experiment_infra.benchmarks.run import make_adapter
from experiment_infra.benchmarks.runner import BenchmarkRunner
from experiment_infra.core.schema import Action
from experiment_infra.runtime.executor import ExperimentRuntime


class ReferenceActions:
    last_call = None

    def __init__(self, adapter, task, fail=False):
        self.adapter, self.task, self.fail, self.index = adapter, task, fail, 0

    async def next_action(self, state):
        if self.adapter.name == "tau2":
            actions = (
                self.adapter.native_task(
                    self.task.task_id, self.task.split
                ).evaluation_criteria.actions
                or []
            )
            if self.fail or self.index == len(actions):
                return Action(tool="environment", operation="finish")
            call = actions[self.index]
            self.index += 1
            return Action(
                tool="environment",
                operation="exec",
                arguments={
                    "command": {
                        "argv": ["tau2", call.name, json.dumps(call.arguments)],
                        "collect": ["native_transcript.json"],
                        "seed": state["state"]["seed"],
                    }
                },
            )
        if self.index:
            return Action(tool="environment", operation="finish")
        self.index += 1
        source = (
            "print('deliberately missing output')\n"
            if self.fail
            else (
                self.adapter.data / "gold_programs" / self.task.payload["gold_program_name"]
            ).read_text()
        )
        return Action(
            tool="environment",
            operation="exec",
            arguments={
                "command": {
                    "argv": ["python", "solution.py"],
                    "files": {"solution.py": source},
                    "seed": state["state"]["seed"],
                    "timeout_seconds": 300,
                }
            },
        )


async def main(args):
    adapter = make_adapter(args.benchmark)
    tasks = adapter.list_tasks("base" if adapter.name == "tau2" else "verified")
    if args.task_id:
        tasks = [t for t in tasks if t.task_id in args.task_id]
    tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("No smoke tasks selected")
    runtime = ExperimentRuntime(environment=adapter.environment)
    try:
        runner = BenchmarkRunner(
            runtime,
            adapter,
            lambda task: ReferenceActions(adapter, task, task == tasks[-1]),
            args.output_dir,
            model="reference-actions-NOT-A-MODEL",
            max_steps=60,
            run_kind="environment_calibration",
            replay_samples=1,
        )
        rows = await runner.run(tasks, resume=args.resume)
        print(
            json.dumps(
                {
                    "run_kind": "environment_calibration",
                    "model_calls": 0,
                    "results": [
                        {
                            k: row.get(k)
                            for k in (
                                "task_id",
                                "success",
                                "failure",
                                "replay",
                                "evaluation_error",
                                "finalization_error",
                            )
                        }
                        for row in rows
                    ],
                },
                indent=2,
            )
        )
    finally:
        runtime.storage.engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["tau2", "scienceagentbench"], required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(main(parser.parse_args()))
