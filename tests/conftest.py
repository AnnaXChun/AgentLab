import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from adapters.storage.local import LocalArtifactStore
from packages.core.events.store import EventStore
from packages.core.models.database import Base


@pytest.fixture
def store():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    yield EventStore(engine)
    engine.dispose()


@pytest.fixture
def artifacts(tmp_path):
    return LocalArtifactStore(tmp_path / "artifacts")
