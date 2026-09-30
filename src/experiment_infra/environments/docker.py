import asyncio
import hashlib
import io
import json
import tarfile

from adapters.sandbox.docker import DockerSandboxBackend
from experiment_infra.core.schema import EnvironmentSpec


class DockerEnvironmentBackend:
    """Extend the existing Docker adapter; no new sandbox runtime."""

    def __init__(self, backend=None):
        self.backend = backend or DockerSandboxBackend()

    async def create(self, spec):
        if spec.backend != "docker" or not spec.image or spec.settings:
            raise ValueError(
                "Docker requires backend=docker, an image, and no unsupported settings"
            )
        return await self.backend.create(spec.model_dump())

    async def execute(self, handle, action):
        return await self.backend.exec(handle, action)

    async def upload(self, handle, path, data):
        await self.backend.upload(handle, path, data)

    async def download(self, handle, path):
        return await self.backend.download(handle, path)

    async def observe(self, handle):
        raw = await self.backend._python(
            handle,
            "import json,platform,sys; print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'implementation':sys.implementation.name}))",
        )
        data = json.loads(raw)
        container = await asyncio.to_thread(self.backend.client.containers.get, handle)
        image = container.image
        data.update(
            image_id=image.id,
            image_ref=(image.attrs.get("RepoDigests") or [image.id])[0],
            architecture=image.attrs.get("Architecture"),
        )
        host = container.attrs["HostConfig"]
        data["spec"] = EnvironmentSpec(
            backend="docker",
            image=data["image_ref"],
            cpu=host["NanoCpus"] / 1e9,
            memory_gb=host["Memory"] / (1024**3),
            dependencies={"python": data["python"]},
        ).model_dump(mode="json")
        data["fingerprint"] = hashlib.sha256(
            json.dumps(
                {key: data[key] for key in ("image_id", "architecture", "python")}, sort_keys=True
            ).encode()
        ).hexdigest()
        data["snapshot_mime_type"] = "application/x-tar"
        return data

    async def snapshot(self, handle):
        return await self.backend.snapshot(handle)

    async def restore(self, handle, snapshot):
        await self.backend.restore(handle, snapshot)

    @staticmethod
    def diff(snapshot_a, snapshot_b):
        def hashes(data):
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                return {
                    item.name: hashlib.sha256(archive.extractfile(item).read()).hexdigest()
                    for item in archive
                    if item.isfile()
                }

        a, b = hashes(snapshot_a), hashes(snapshot_b)
        return {
            "added": sorted(b.keys() - a.keys()),
            "removed": sorted(a.keys() - b.keys()),
            "modified": sorted(k for k in a.keys() & b.keys() if a[k] != b[k]),
        }

    async def destroy(self, handle):
        await self.backend.destroy(handle)
