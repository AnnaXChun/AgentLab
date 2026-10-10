"""Build fresh task inputs, never import old trajectories as GRPO samples."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiment_infra.integrations.verl.session import ToyAdapter

SYSTEM = """You execute a task through an experiment runtime. Return exactly one JSON object per turn.
No markdown, reasoning text, or commentary. Use this contract:
{"tool":"environment","operation":"exec","arguments":{"command":{"argv":["python","-c","print(1)"]}}}
To finish: {"tool":"environment","operation":"finish","arguments":{}}
For telecom, argv must be ["tau2", native_tool_name, JSON_arguments_string].
Use only the listed native tools and their exact argument schema. Tool observations arrive in user messages.
"""


def build_rows(adapter, split, limit):
    rows = []
    for i, task in enumerate(adapter.list_tasks(split)[:limit]):
        # Only public goal, policy, initial messages and tool schemas. No hidden criteria.
        user = {
            "goal": task.goal,
            "context": adapter.agent_context(task),
            "tools": adapter.available_tools(task),
        }
        rows.append(
            {
                "data_source": adapter.name,
                "agent_name": "experiment_runtime",
                "prompt": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(user)},
                ],
                "reward_model": {"style": "rule", "ground_truth": "runtime_verified"},
                "extra_info": {
                    "index": i,
                    "task_id": task.task_id,
                    "benchmark": adapter.name,
                    "split": split,
                    "benchmark_version": adapter.version,
                },
                "experiment_metadata": {"trajectory_source": "ONLINE_RL", "task_input_only": True},
            }
        )
    if not rows:
        raise ValueError("Empty task split")
    return rows


def prepare(benchmark, directory, train_limit=8, val_limit=5):
    if benchmark == "toy":
        adapter = ToyAdapter()
    else:
        from experiment_infra.benchmarks.run import make_adapter

        adapter = make_adapter(benchmark)
    train = build_rows(adapter, "train", train_limit)
    val = build_rows(adapter, "test", val_limit)
    if {r["extra_info"]["task_id"] for r in train} & {r["extra_info"]["task_id"] for r in val}:
        raise ValueError("Training and evaluation task leakage")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        pq.write_table(pa.Table.from_pylist(rows), directory / f"{name}.parquet")
    return {"benchmark": benchmark, "train": len(train), "val": len(val)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["toy", "tau2"], default="toy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=8)
    parser.add_argument("--val-limit", type=int, default=5)
    args = parser.parse_args()
    print(prepare(args.benchmark, args.output_dir, args.train_limit, args.val_limit))
