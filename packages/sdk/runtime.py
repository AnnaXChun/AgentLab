import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from sqlalchemy import create_engine

from adapters.sandbox.docker import DockerSandboxBackend
from adapters.storage.local import LocalArtifactStore
from domain_packs.loader import load_pack
from packages.core.events.store import EventStore
from packages.mcp_gateway.gateway import MCPGateway
from packages.retrieval.retriever import PostgresHybridRetriever
from packages.sdk.harness import Harness


def environment_metadata():
    try:
        version = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        version = os.getenv("HARNESS_CODE_VERSION", "uncommitted-v0")
    root = Path(__file__).resolve().parents[2]
    source = hashlib.sha256()
    files = [
        p
        for directory in ("packages", "adapters", "domain_packs", "migrations", "src")
        for p in (root / directory).rglob("*")
        if p.suffix in (".py", ".yaml", ".md")
    ]
    files += [root / "pyproject.toml", root / "uv.lock"]
    for path in sorted(files):
        if path.is_file():
            source.update(str(path.relative_to(root)).encode())
            source.update(path.read_bytes())
    return {
        "code_version": version,
        "source_sha256": source.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "scientific-harness",
                "pydantic",
                "sqlalchemy",
                "mcp",
                "docker",
                "pgvector",
                "pyarrow",
                "duckdb",
            )
        },
        "seed": 0,
    }


class Runtime:
    def __init__(self, database_url=None, artifact_root=None, *, initialize_sandbox=True):
        self.engine = create_engine(
            database_url
            or os.getenv(
                "DATABASE_URL", "postgresql+psycopg://harness:harness@localhost:55432/harness"
            ),
            pool_pre_ping=True,
        )
        self.store = EventStore(self.engine)
        self.artifacts = LocalArtifactStore(
            artifact_root or os.getenv("ARTIFACT_ROOT", ".artifacts")
        )
        self.sandbox = DockerSandboxBackend() if initialize_sandbox else None

    def components(self, pack):
        allowed = {name for server in pack.mcp_servers for name in server.get("tool_schemas", {})}
        retriever = PostgresHybridRetriever(
            self.engine, f"{pack.metadata['name']}:{pack.metadata['version']}"
        )
        return retriever, MCPGateway(pack.mcp_servers, allowed)

    async def create(self, goal, domain_pack="demo", agent_id="mock"):
        pack = load_pack(domain_pack)
        retriever, gateway = self.components(pack)
        return await Harness.create(
            self.store,
            self.artifacts,
            self.sandbox,
            retriever,
            gateway,
            pack,
            goal,
            agent_id,
            environment_metadata(),
        )

    def load(self, episode_id, branch_id=None):
        from packages.core.models.schemas import DomainPack

        episode = self.store.episode(episode_id)
        if episode.metadata.get("public_schema") == "2.0":
            raise ValueError("Use /experiments for v2 experiment partitions")
        pack = DomainPack.model_validate(episode.metadata["pack"])
        retriever, gateway = self.components(pack)
        return Harness(
            self.store,
            self.artifacts,
            self.sandbox,
            retriever,
            gateway,
            pack,
            episode_id,
            branch_id,
        )
