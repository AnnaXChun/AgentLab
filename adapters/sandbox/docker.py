import asyncio
import base64
import io
import json
import os
import subprocess
import tarfile
from pathlib import PurePosixPath

import docker

from packages.sandbox.base import Command, ExecutionResult

MAX_FILE = 16 * 1024 * 1024
SUPERVISOR = r"""
import base64,json,os,signal,subprocess,sys,tempfile
c=json.loads(base64.b64decode(sys.argv[1]))
env={**os.environ,**c['environment'],'PYTHONHASHSEED':str(c['seed']),'HARNESS_SEED':str(c['seed'])}
with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
 p=subprocess.Popen(c['argv'],cwd='/work',env=env,stdout=out,stderr=err,start_new_session=True)
 timed=False
 try: p.wait(timeout=c['timeout_seconds'])
 except subprocess.TimeoutExpired:
  timed=True
  os.killpg(p.pid,signal.SIGKILL)
  p.wait()
 out.seek(0); err.seek(0)
 stdout=out.read(16777217); stderr=err.read(16777217)
 print(json.dumps({'exit_code':124 if timed else p.returncode,'stdout':stdout[:16777216].decode(errors='replace'),'stderr':stderr[:16777216].decode(errors='replace'),'stdout_truncated':len(stdout)>16777216,'stderr_truncated':len(stderr)>16777216,'timed_out':timed}))
"""


def safe_path(path: str) -> str:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or str(p) in ("", "."):
        raise ValueError("Sandbox path must be a relative file path")
    return str(p)


class DockerSandboxBackend:
    def __init__(self, client=None):
        if client is not None:
            self.client = client
        elif os.getenv("DOCKER_HOST"):
            self.client = docker.from_env()
        else:
            try:
                host = subprocess.check_output(
                    ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                self.client = docker.DockerClient(base_url=host)
            except (OSError, subprocess.SubprocessError):
                self.client = docker.from_env()

    async def create(self, config: dict) -> str:
        def launch():
            image = config.get("image", "python:3.12-slim")
            try:
                self.client.images.get(image)
            except docker.errors.ImageNotFound:
                self.client.images.pull(image)
            c = self.client.containers.run(
                image,
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
                mem_limit=f"{config.get('memory_gb', 1)}g",
                nano_cpus=int(config.get("cpu", 1) * 1e9),
                pids_limit=64,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                labels={"scientific-harness": "sandbox"},
            )
            return c.id

        return await asyncio.to_thread(launch)

    async def exec(self, handle: str, command: Command) -> ExecutionResult:
        def execute():
            container = self.client.containers.get(handle)
            encoded = base64.b64encode(command.model_dump_json().encode()).decode()
            result = container.exec_run(["python", "-I", "-c", SUPERVISOR, encoded], demux=True)
            out, err = result.output
            if result.exit_code != 0:
                return ExecutionResult(
                    exit_code=result.exit_code,
                    stdout=(out or b"").decode(errors="replace"),
                    stderr=(err or b"").decode(errors="replace"),
                )
            payload = json.loads(out)
            payload["environment"] = {
                "image_id": container.image.id,
                "image": container.image.tags[0] if container.image.tags else container.image.id,
                "runtime": "docker",
                "platform": container.image.attrs.get("Architecture", "unknown"),
            }
            return ExecutionResult(**payload)

        return await asyncio.to_thread(execute)

    async def _python(self, handle: str, code: str, *args: str) -> bytes:
        result = await asyncio.to_thread(
            self.client.containers.get(handle).exec_run,
            ["python", "-I", "-c", code, *args],
            demux=True,
        )
        out, err = result.output
        if result.exit_code:
            raise RuntimeError(
                (err or out or b"Sandbox file operation failed").decode(errors="replace")
            )
        return out or b""

    async def upload(self, handle: str, path: str, data: bytes) -> None:
        path = safe_path(path)
        if len(data) > MAX_FILE:
            raise ValueError("Upload exceeds file limit")
        prefix = "import pathlib,sys,base64; p=pathlib.Path('/work')/sys.argv[1]; assert p.resolve().is_relative_to('/work') and not p.is_symlink(); "
        await self._python(
            handle, prefix + "p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'')", path
        )
        for start in range(0, len(data), 32768):
            await self._python(
                handle,
                prefix + "f=p.open('ab'); f.write(base64.b64decode(sys.argv[2])); f.close()",
                path,
                base64.b64encode(data[start : start + 32768]).decode(),
            )

    async def download(self, handle: str, path: str) -> bytes:
        return await self._python(
            handle,
            "import pathlib,sys; p=pathlib.Path('/work')/sys.argv[1]; assert p.resolve().is_relative_to('/work') and p.is_file() and not p.is_symlink(); assert p.stat().st_size <= 16777216; sys.stdout.buffer.write(p.read_bytes())",
            safe_path(path),
        )

    async def snapshot(self, handle: str) -> bytes:
        return await self._python(
            handle,
            """
import io,pathlib,sys,tarfile
root=pathlib.Path('/work')
files=list(root.rglob('*'))
assert all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in files)
assert sum(p.stat().st_size for p in files if p.is_file()) < 15000000
out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w') as archive:
 archive.add(root,arcname='work')
assert out.tell() <= 16777216
sys.stdout.buffer.write(out.getvalue())
""",
        )

    async def restore(self, handle: str, snapshot: bytes) -> None:
        with tarfile.open(fileobj=io.BytesIO(snapshot)) as archive:
            for item in archive:
                p = PurePosixPath(item.name)
                if (
                    p.is_absolute()
                    or ".." in p.parts
                    or p.parts[0] != "work"
                    or not (item.isfile() or item.isdir())
                ):
                    raise ValueError("Unsafe snapshot")
        import uuid

        path = f".restore-{uuid.uuid4().hex}.tar"
        await self.upload(handle, path, snapshot)
        await self._python(
            handle,
            """
import io,pathlib,shutil,sys,tarfile
root=pathlib.Path('/work')
data=(root/sys.argv[1]).read_bytes()
for p in root.iterdir():
 if p.is_dir() and not p.is_symlink(): shutil.rmtree(p)
 else: p.unlink()
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
 for member in archive:
  if member.name == 'work': continue
  member.name=member.name.removeprefix('work/')
  archive.extract(member,root,filter='data')
""",
            path,
        )

    async def destroy(self, handle: str) -> None:
        await asyncio.to_thread(self.client.containers.get(handle).remove, force=True)
