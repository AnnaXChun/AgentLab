from pathlib import Path

import yaml

from packages.core.models.schemas import DomainPack

ROOT = Path(__file__).parent


def load_pack(name: str) -> DomainPack:
    if not name.replace("_", "").isalnum():
        raise ValueError("Invalid pack name")
    path = ROOT / name / "manifest.yaml"
    data = yaml.safe_load(path.read_text())
    return DomainPack.model_validate(data)
