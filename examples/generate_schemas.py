import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_infra.core import schema as experiment_schemas
from packages.core.models import schemas
from packages.sdk.api import create_app

root = Path(__file__).resolve().parents[1] / "schemas"
root.mkdir(exist_ok=True)
for name in (
    "Episode",
    "Event",
    "Artifact",
    "Evidence",
    "Claim",
    "Verification",
    "EventEdge",
    "DomainPack",
    "AgentAction",
):
    (root / f"{name}.json").write_text(
        json.dumps(getattr(schemas, name).model_json_schema(), indent=2) + "\n"
    )
(root / "openapi.json").write_text(json.dumps(create_app().openapi(), indent=2) + "\n")

# V2 public entities are separate from the retained v1 transport/legacy entities.

v2 = root / "v2"
v2.mkdir(exist_ok=True)
for name in [
    *experiment_schemas.OBJECT_TYPES,
    "Action",
    "EnvironmentSpec",
    "DomainPack",
    "ReplayResult",
]:
    (v2 / f"{name}.json").write_text(
        json.dumps(getattr(experiment_schemas, name).model_json_schema(), indent=2) + "\n"
    )
