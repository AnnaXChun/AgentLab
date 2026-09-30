from collections.abc import Callable

MIGRATIONS: dict[str, Callable[[dict], dict]] = {}


def migrate(payload: dict) -> dict:
    seen = set()
    while payload.get("schema_version") != "1.0":
        version = payload.get("schema_version")
        if version in seen or version not in MIGRATIONS:
            raise ValueError(f"Unsupported schema_version: {version}")
        seen.add(version)
        payload = MIGRATIONS[version](dict(payload))
    return payload
