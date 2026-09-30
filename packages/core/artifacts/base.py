from typing import Protocol

from packages.core.models.schemas import Artifact


class ArtifactStore(Protocol):
    async def put(
        self,
        data: bytes,
        event_id: str,
        mime_type: str = "application/octet-stream",
        metadata: dict | None = None,
    ) -> Artifact: ...
    async def get(self, artifact_id: str) -> bytes: ...
    async def exists(self, artifact_id: str) -> bool: ...
