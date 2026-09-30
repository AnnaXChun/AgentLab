from typing import Protocol

from pydantic import Field

from packages.core.models.schemas import Schema


class Command(Schema):
    argv: list[str] = Field(min_length=1)
    environment: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    collect: list[str] = Field(default_factory=list)
    seed: int = 0


class ExecutionResult(Schema):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    environment: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)


class SandboxBackend(Protocol):
    async def create(self, config: dict) -> str: ...
    async def exec(self, handle: str, command: Command) -> ExecutionResult: ...
    async def upload(self, handle: str, path: str, data: bytes) -> None: ...
    async def download(self, handle: str, path: str) -> bytes: ...
    async def snapshot(self, handle: str) -> bytes: ...
    async def restore(self, handle: str, snapshot: bytes) -> None: ...
    async def destroy(self, handle: str) -> None: ...
