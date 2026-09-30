"""python -m experiment_infra.benchmarks.run --benchmark tau2 --limit 5"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from experiment_infra.agents.openai_compatible import OpenAICompatibleAgentBackend
from experiment_infra.benchmarks.runner import BenchmarkRunner, summarize
from experiment_infra.runtime.executor import ExperimentRuntime


def make_adapter(name):
    if name == "tau2":
        from experiment_infra.benchmarks.tau2 import Tau2Adapter

        return Tau2Adapter(os.getenv("TAU2_ROOT", ".benchmarks/tau2-bench"))
    from experiment_infra.benchmarks.scienceagentbench import ScienceAgentBenchAdapter

    return ScienceAgentBenchAdapter(
        os.getenv("SAB_ROOT", ".benchmarks/ScienceAgentBench"),
        os.getenv("SAB_DATA", ".benchmarks/scienceagentbench-data/benchmark"),
        os.getenv("SAB_ANNOTATIONS", ".benchmarks/scienceagentbench-data/verified.parquet"),
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["scienceagentbench", "tau2"], required=True)
    p.add_argument("--model", default=os.getenv("MODEL_NAME"))
    p.add_argument("--split")
    p.add_argument("--task-id", action="append", help="Repeat to select explicit IDs")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--context-limit", type=int, default=120000, help="UTF-8 input byte budget")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--replay-samples", type=int, default=1, help="First N eligible successes and failures"
    )
    p.add_argument(
        "--check-environment",
        action="store_true",
        help="Create/snapshot/restore/destroy without model calls",
    )
    return p


async def main(args):
    if min(args.limit, args.max_steps, args.context_limit) < 1 or args.replay_samples < 0:
        raise ValueError("Limits must be positive; replay samples must be nonnegative")
    adapter = make_adapter(args.benchmark)
    split = args.split or ("verified" if args.benchmark == "scienceagentbench" else "base")
    tasks = adapter.list_tasks(split)
    if args.task_id:
        unknown = set(args.task_id) - {t.task_id for t in tasks}
        if unknown:
            raise ValueError(f"Unknown task IDs: {sorted(unknown)}")
        tasks = [t for t in tasks if t.task_id in args.task_id]
    tasks = tasks[: args.limit]
    print(
        f"Benchmark loaded: {adapter.name}, {adapter.version}, split={split}, tasks={len(tasks)}",
        flush=True,
    )
    if args.check_environment:
        results = []
        for task in tasks:
            handle = None
            try:
                handle = await adapter.environment.create(adapter.environment_spec(task))
                snapshot = await adapter.environment.snapshot(handle)
                await adapter.environment.restore(handle, snapshot)
                info = await adapter.environment.observe(handle)
                results.append(
                    {
                        "task_id": task.task_id,
                        "ready": True,
                        "snapshot_bytes": len(snapshot),
                        "fingerprint": info["fingerprint"],
                    }
                )
            finally:
                if handle:
                    await adapter.environment.destroy(handle)
        print(json.dumps({"environment_checks": results, "model_calls": 0}, indent=2))
        return
    # Fail preflight before starting any tasks if model configuration is absent.
    config_check = OpenAICompatibleAgentBackend(model=args.model)
    await config_check.close()
    default_output = args.output_dir is None
    args.output_dir = args.output_dir or Path("results") / args.benchmark
    runtime = ExperimentRuntime(environment=adapter.environment)
    try:
        runner = BenchmarkRunner(
            runtime,
            adapter,
            lambda task: OpenAICompatibleAgentBackend(
                model=args.model,
                seed=args.seed,
                temperature=args.temperature,
                context_limit=args.context_limit,
            ),
            args.output_dir,
            model=args.model,
            seed=args.seed,
            temperature=args.temperature,
            max_steps=args.max_steps,
            context_limit=args.context_limit,
            replay_samples=args.replay_samples,
        )
        rows = await runner.run(tasks, resume=args.resume)
        if default_output:
            from experiment_infra.benchmarks.report import write_summary_table

            write_summary_table(
                [Path("results") / name for name in ("scienceagentbench", "tau2")],
                Path("results"),
            )
        s = summarize(rows)
        print(
            f"Tasks attempted: {s['tasks_attempted']}\nTasks succeeded: {s['tasks_successful']}\nSuccess rate: {s['success_rate']}"
        )
        print(
            f"Trajectories stored: {s['valid_trajectories']}\nArtifacts stored: {sum(r.get('artifacts_stored', 0) for r in rows)}\nEvidence stored: {sum(r.get('evidence_stored', 0) for r in rows)}"
        )
        print(
            f"Replay eligible tasks: {s['replay_eligible']}\nFailure distribution: {s['failure_types']}\nResults exported: {args.output_dir.resolve()}"
        )
    finally:
        runtime.storage.engine.dispose()


if __name__ == "__main__":
    asyncio.run(main(parser().parse_args()))
