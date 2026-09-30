"""Combine explicitly selected batches; never mix calibration and model baselines."""

import argparse
import csv
import json
from pathlib import Path

from experiment_infra.benchmarks.runner import atomic_json


def write_summary_table(directories, output):
    rows = []
    for directory in directories:
        path = Path(directory) / "benchmark_summary.json"
        if path.is_file():
            rows.append(
                {**json.loads(path.read_text()), "source_directory": str(path.parent.resolve())}
            )
    if not rows:
        raise ValueError("No completed batch summaries found")
    if len({row["run_kind"] for row in rows}) != 1:
        raise ValueError("Do not combine environment calibration with model baselines")
    output = Path(output)
    atomic_json(output / "benchmark_summary.json", rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "benchmark_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, dict) else value
                    for key, value in row.items()
                }
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    write_summary_table(args.directories, args.output_dir)
