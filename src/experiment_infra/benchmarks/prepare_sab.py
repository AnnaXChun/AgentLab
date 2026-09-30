"""Build the unmodified official SAB base image after installing benchmark extras."""

import json
import os
import sys
from pathlib import Path


def main():
    root = Path(os.getenv("SAB_ROOT", ".benchmarks/ScienceAgentBench")).resolve()
    data = Path(os.getenv("SAB_DATA", ".benchmarks/scienceagentbench-data/benchmark")).resolve()
    annotations = Path(
        os.getenv("SAB_ANNOTATIONS", ".benchmarks/scienceagentbench-data/verified.parquet")
    ).resolve()
    sys.path.insert(0, str(root))
    import pyarrow.parquet as pq
    from evaluation.harness.docker_build import build_base_images

    from adapters.sandbox.docker import DockerSandboxBackend

    rows = pq.read_table(annotations).to_pylist()[:1]
    rows[0]["instance_id"] = str(rows[0]["instance_id"])
    os.chdir(root)
    build_base_images(
        DockerSandboxBackend().client, rows, str(data), str(root / "pred_programs"), False
    )
    print(json.dumps({"official_base_image": "built", "task_count": len(rows)}))


if __name__ == "__main__":
    main()
