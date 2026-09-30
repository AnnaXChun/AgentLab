import pytest
from pydantic import ValidationError

from adapters.sandbox.docker import SUPERVISOR, safe_path
from packages.sandbox.base import Command


def test_paths_and_timeout():
    assert safe_path("result.json") == "result.json"
    for path in ("../secret", "/etc/passwd", "."):
        with pytest.raises(ValueError):
            safe_path(path)
    with pytest.raises(ValidationError):
        Command(argv=["python"], timeout_seconds=0)
    compile(SUPERVISOR, "supervisor", "exec")
