"""Pinned official SAB inputs and Docker evaluator. No hidden gold data in agent context."""

import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from experiment_infra.benchmarks.base import AdapterBase, BenchmarkTask
from experiment_infra.core.schema import EnvironmentSpec
from experiment_infra.environments.docker import DockerEnvironmentBackend

SAB_COMMIT = "c26e151ed601ba109dc4d35e057ff8e73fec469d"
SAB_DATA_REVISION = "9c6e96c9e74572e979b0930ee735041cef528cb7"
SAB_ANNOTATION_SHA256 = "c6f937863a220bd1762a00c20a0f79cc8dfca900b819bdb552150310731ae147"


class ScienceAgentBenchAdapter(AdapterBase):
    name = "scienceagentbench"
    version = SAB_COMMIT + "/" + SAB_DATA_REVISION
    replay_kind = "deterministic_execution_with_external_native_evaluation"

    def __init__(self, root, data, annotations, image=None, evaluator_python=None):
        self.root, self.data, self.annotations = (
            Path(p).resolve() for p in (root, data, annotations)
        )
        arch = "arm64" if platform.machine() in {"arm64", "aarch64"} else "x86_64"
        self.image = image or os.getenv("SAB_AGENT_IMAGE", f"sab.base.{arch}:latest")
        self.python = evaluator_python or os.getenv("SAB_EVALUATOR_PYTHON", sys.executable)
        commit = subprocess.check_output(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != SAB_COMMIT:
            raise ValueError("ScienceAgentBench checkout must match pinned revision")
        self.environment = ScienceDockerEnvironment(self.data)
        self.evaluation_artifacts = {}

    def list_tasks(self, split):
        if split != "verified":
            raise ValueError("This adapter pins the official verified split")
        if hashlib.sha256(self.annotations.read_bytes()).hexdigest() != SAB_ANNOTATION_SHA256:
            raise ValueError("Annotation file differs from the pinned verified dataset")
        import pyarrow.parquet as pq

        rows = pq.read_table(self.annotations).to_pylist()
        return [BenchmarkTask(str(r["instance_id"]), r["task_inst"], split, r) for r in rows]

    def environment_spec(self, task):
        # Input datasets are a read-only mount; never expose eval/gold/scoring_rubrics.
        for folder in ("datasets", "eval_programs", "gold_programs"):
            if not (self.data / folder).is_dir():
                raise FileNotFoundError(f"Missing official SAB bundle directory: {folder}")
        return EnvironmentSpec(image=self.image, memory_gb=4, cpu=2)

    def available_tools(self, task):
        return {
            "exec": "Run argv in /work; read/search using Python or shell tools",
            "write": "arguments.command.files writes UTF-8 files before argv execution",
            "finish": True,
            "required_submission": "solution.py: self-contained Python program",
            "datasets": "benchmark/datasets/ (read-only)",
            "expected_output": task.payload["output_fname"],
        }

    def agent_context(self, task):
        return {
            k: task.payload.get(k)
            for k in ("domain_knowledge", "dataset_folder_tree", "dataset_preview")
        }

    def validate_action(self, task, action):
        if action.tool != "environment" or action.operation not in {"exec", "finish"}:
            raise ValueError("Unsupported SAB action")

    async def evaluate(self, runtime, task, run, termination):
        # Fresh prediction folder and unique official run_id prevent stale evaluator results.
        directory = self.root / "agentlab_evaluations" / run.run_id
        directory.mkdir(parents=True, exist_ok=False)
        predictions = directory / "predictions"
        predictions.mkdir()
        source = await runtime.environment.submission(run.environment_handle)
        if source is not None:
            (predictions / ("pred_" + task.payload["gold_program_name"])).write_bytes(source)
        task_file = directory / "task.json"
        task_file.write_text(json.dumps({**task.payload, "instance_id": task.task_id}))
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(x in k.upper() for x in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        }
        env["PYTHONPATH"] = str(self.root)
        env["DOCKER_HOST"] = runtime.environment.backend.client.api.base_url.replace(
            "http+docker://localhost",
            subprocess.check_output(
                ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
                text=True,
            ).strip(),
        )
        # No CLI key gate, live dataset fetch, or default-zero fallback. These are the same
        # official build/run functions the CLI calls, using our pinned task row.
        worker = Path(__file__).with_name("sab_worker.py")
        command = [
            self.python,
            str(worker),
            str(task_file),
            str(self.data),
            str(predictions),
            run.run_id,
        ]
        with (directory / "evaluator.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                *command, cwd=directory, env=env, stdout=log, stderr=log, start_new_session=True
            )
            try:
                await asyncio.wait_for(process.wait(), 3600)
            except TimeoutError:
                import signal

                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
                raise TimeoutError("Official SAB evaluator exceeded 3600 seconds")
        saved = []
        for output_file in sorted(directory.rglob("*")):
            if output_file.is_file() and output_file.suffix in {
                ".log",
                ".json",
                ".csv",
                ".txt",
                ".png",
                ".py",
            }:
                artifact = await runtime._artifact(
                    run,
                    output_file.read_bytes(),
                    "log" if output_file.suffix == ".log" else "generated_file",
                    metadata={
                        "role": "native_evaluator_output",
                        "relative_path": str(output_file.relative_to(directory)),
                    },
                )
                saved.append(artifact.artifact_id)
        self.evaluation_artifacts[run.run_id] = saved
        if process.returncode != 0:
            raise RuntimeError(
                f"Official SAB harness failed; inspect {directory / 'evaluator.log'}"
            )
        # Use the per-instance native result, not the harness's default-zero fallback rows.
        # The pinned harness output layout is checked explicitly below by native_result_path.
        result_path = native_result_path(directory, run.run_id, task.task_id)
        if not result_path.is_file():
            raise RuntimeError(
                "Official evaluator produced no per-instance result (no fallback score)"
            )
        values = json.loads(result_path.read_text())
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError("Unknown native SAB result shape")
        valid, codebert, score, info = values
        return {
            "benchmark": self.name,
            "version": self.version,
            "metric": "success_rate",
            "score": score,
            "success": score == 1,
            "evaluator": "ScienceAgentBench.compute_scores.compute_scores",
            "native_result": {
                "valid_program": valid,
                "codebert_score": codebert,
                "success_rate": score,
                "log_info": info,
            },
            "annotation_sha256": hashlib.sha256(self.annotations.read_bytes()).hexdigest(),
        }


def native_result_path(root, run_id, task_id):
    return Path(root) / "logs/run_evaluation" / run_id / task_id / "output/result.json"


class ScienceDockerEnvironment(DockerEnvironmentBackend):
    """Dataset mount excluded from workspace snapshots; fingerprint binds its bytes."""

    def __init__(self, data):
        super().__init__()
        self.data = Path(data)
        self._dataset_hash = None

    async def submission(self, handle):
        exists = await self.backend._python(
            handle, "import pathlib; print(int(pathlib.Path('/work/solution.py').exists()))"
        )
        # Only absence is a missing submission. Docker/download errors must not become scores.
        if exists.strip() == b"0":
            return None
        return await self.download(handle, "solution.py")

    def dataset_hash(self):
        if self._dataset_hash is None:
            h = hashlib.sha256()
            for p in sorted((self.data / "datasets").rglob("*")):
                if p.is_file():
                    h.update(str(p.relative_to(self.data)).encode())
                    with p.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(chunk)
            self._dataset_hash = h.hexdigest()
        return self._dataset_hash

    async def create(self, spec):
        # Same Docker hardening as V0.2; only the immutable input mount is benchmark-specific.
        def launch():
            self.backend.client.images.get(
                spec.image
            )  # Explicit preparation, no implicit huge pull.
            return self.backend.client.containers.run(
                spec.image,
                ["python", "-I", "-c", "import time; time.sleep(86400)"],
                detach=True,
                user="1000:1000",
                working_dir="/work",
                network_disabled=True,
                read_only=True,
                tmpfs={
                    "/work": "rw,exec,size=128m,uid=1000,gid=1000",
                    "/tmp": "rw,noexec,size=64m,uid=1000,gid=1000",
                },
                volumes={
                    str(self.data / "datasets"): {"bind": "/work/benchmark/datasets", "mode": "ro"}
                },
                mem_limit=f"{spec.memory_gb}g",
                nano_cpus=int(spec.cpu * 1e9),
                pids_limit=64,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                labels={"scientific-harness": "sandbox"},
            ).id

        handle = await asyncio.to_thread(launch)
        await self.backend._python(
            handle, "import pathlib; pathlib.Path('/work/pred_results').mkdir(exist_ok=True)"
        )
        return handle

    async def observe(self, handle):
        result = await super().observe(handle)
        result["dataset_sha256"] = self.dataset_hash()
        result["fingerprint"] = hashlib.sha256(
            (result["fingerprint"] + self.dataset_hash()).encode()
        ).hexdigest()
        return result

    async def snapshot(self, handle):
        return await self.backend._python(
            handle,
            """
import io,pathlib,sys,tarfile
root=pathlib.Path('/work')
files=[p for p in root.rglob('*') if not p.is_relative_to(root/'benchmark')]
assert all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in files)
assert sum(p.stat().st_size for p in files if p.is_file()) < 15000000
out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w') as archive:
 for p in files: archive.add(p,arcname='work/'+str(p.relative_to(root)),recursive=False)
assert out.tell() <= 16777216
sys.stdout.buffer.write(out.getvalue())
""",
        )

    async def restore(self, handle, snapshot):
        import io
        import tarfile
        from pathlib import PurePosixPath

        with tarfile.open(fileobj=io.BytesIO(snapshot)) as archive:
            for item in archive:
                path = PurePosixPath(item.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.parts[0] != "work"
                    or "benchmark" in path.parts
                    or not (item.isfile() or item.isdir())
                ):
                    raise ValueError("Unsafe benchmark snapshot")
        await self.upload(handle, ".restore.tar", snapshot)
        await self.backend._python(
            handle,
            """
import io,pathlib,shutil,tarfile
root=pathlib.Path('/work'); data=(root/'.restore.tar').read_bytes()
for p in root.iterdir():
 if p.name=='benchmark': continue
 if p.is_dir() and not p.is_symlink(): shutil.rmtree(p)
 else: p.unlink()
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
 for item in archive:
  item.name=item.name.removeprefix('work/')
  archive.extract(item,root,filter='data')
""",
        )
