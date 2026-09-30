from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scientific-harness-demo")


@mcp.tool()
def load_dataset(candidate: str) -> dict[str, Any]:
    """Return a deterministic synthetic candidate dataset."""
    if candidate != "A":
        raise ValueError("Demo only includes candidate A")
    return {"candidate": candidate, "values": [0.18, 0.20, 0.22], "units": "dimensionless"}


@mcp.tool()
def run_mock_simulation(candidate: str, seed: int = 0) -> dict[str, Any]:
    """Deterministic synthetic simulation, not a physical scientific model."""
    if candidate != "A":
        raise ValueError("Unknown candidate")
    return {
        "candidate": candidate,
        "values": [0.18, 0.20, 0.22],
        "seed": seed,
        "model": "synthetic-v1",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
