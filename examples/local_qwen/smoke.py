"""One real local-model call, one Docker action, and exported runtime artifacts."""

import asyncio
import json
from pathlib import Path

from experiment_infra.agents.openai_compatible import OpenAICompatibleAgentBackend
from experiment_infra.core.schema import EnvironmentSpec
from experiment_infra.runtime.executor import ExperimentRuntime


async def main():
    agent = OpenAICompatibleAgentBackend(temperature=0.7)
    runtime = ExperimentRuntime()
    experiment = run = None
    output = Path("results/local-qwen-connectivity")
    report = {"success": False}
    try:
        task = "Run Python to calculate the sum of integers 1 through 10 and print only the resulting number. Use argv with python, -c, and the Python code as three separate elements."
        experiment = await runtime.create_experiment(
            task,
            environment_spec=EnvironmentSpec(image="python:3.12-slim"),
            agent_backend="openai-compatible/" + agent.model,
        )
        run = await runtime.start_run(
            experiment.experiment_id,
            config={
                "model": agent.model,
                "run_kind": "local_model_connectivity",
                "temperature": 0.7,
            },
            seed=0,
        )
        report["run_id"] = run.run_id
        action = await agent.next_action(
            {"task": task, "available_tools": {"exec": True, "finish": True}}
        )
        step = await runtime.execute(
            experiment.experiment_id,
            action,
            run_id=run.run_id,
            token_usage=agent.last_call["usage"],
        )
        report.update(
            step_id=step.step_id, model_call=agent.last_call, output=step.observation.stdout
        )
        assert step.observation.exit_code == 0
        assert step.observation.stdout.strip() == "55"
        report["success"] = True
    finally:
        if run:
            await runtime.finish_run(run.run_id)
        if experiment:
            await runtime.finish_experiment(experiment.experiment_id)
            report["export"] = await runtime.export(experiment.experiment_id, output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "check.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        await agent.close()
        runtime.storage.engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
