import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path

from packages.core.models.schemas import Artifact


class LocalArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def path(self, artifact_id: str) -> Path:
        digest = artifact_id.removeprefix("sha256:")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid SHA-256 artifact ID")
        return self.root / "sha256" / digest[:2] / digest

    async def put(
        self,
        data: bytes,
        event_id: str,
        mime_type: str = "application/octet-stream",
        metadata: dict | None = None,
    ) -> Artifact:
        digest = hashlib.sha256(data).hexdigest()
        path = self.path(digest)

        def commit():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(tmp, 0o444)
                try:
                    os.link(tmp, path)  # atomic no-clobber publication
                except FileExistsError:
                    if path.read_bytes() != data:
                        raise ValueError("CAS corruption: refusing overwrite")
            finally:
                os.unlink(tmp)

        await asyncio.to_thread(commit)
        return Artifact(
            artifact_id=f"sha256:{digest}",
            sha256=digest,
            uri=f"cas://sha256/{digest}",
            mime_type=mime_type,
            size_bytes=len(data),
            created_by_event_id=event_id,
            metadata=metadata or {},
        )

    async def get(self, artifact_id: str) -> bytes:
        data = await asyncio.to_thread(self.path(artifact_id).read_bytes)
        if hashlib.sha256(data).hexdigest() != artifact_id.removeprefix("sha256:"):
            raise ValueError("CAS integrity check failed")
        return data

    async def exists(self, artifact_id: str) -> bool:
        try:
            await self.get(artifact_id)
            return True
        except FileNotFoundError:
            return False
