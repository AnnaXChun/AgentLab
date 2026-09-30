import pytest
from jsonschema import ValidationError

from domain_packs.loader import load_pack
from packages.mcp_gateway.gateway import MCPGateway


@pytest.mark.parametrize("name", ["demo", "materials", "biology"])
async def test_pack_without_core_changes(name):
    pack = load_pack(name)
    gateway = MCPGateway(pack.mcp_servers, {"load_dataset", "run_mock_simulation"})
    result = await gateway.call("load_dataset", {"candidate": "A"})
    assert result["structuredContent"]["values"] == [0.18, 0.20, 0.22]


async def test_permissions_and_schema_before_execution():
    gateway = MCPGateway(load_pack("demo").mcp_servers, {"load_dataset"})
    with pytest.raises(PermissionError):
        await gateway.call("unauthorized", {})
    with pytest.raises(ValidationError):
        await gateway.call("load_dataset", {"candidate": 12})
