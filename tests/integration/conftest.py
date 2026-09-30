import os

import pytest
from sqlalchemy import text

from packages.sdk.runtime import Runtime


@pytest.fixture
def runtime(tmp_path):
    try:
        runtime = Runtime(artifact_root=tmp_path / "cas")
        runtime.sandbox.client.ping()
        with runtime.engine.connect() as connection:
            connection.execute(text("SELECT count(*) FROM retrieval_chunks"))
    except Exception as exc:
        if os.getenv("HARNESS_INTEGRATION") == "1":
            pytest.fail(f"Required integration infrastructure is unavailable: {exc}")
        pytest.skip(f"Run docker compose up to enable integration tests: {exc}")
    yield runtime
    runtime.engine.dispose()
