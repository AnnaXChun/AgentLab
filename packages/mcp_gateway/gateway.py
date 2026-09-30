import sys
from contextlib import asynccontextmanager
from typing import Any

from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPGateway:
    def __init__(self, servers: list[dict], allowed_tools: set[str]):
        self.servers = servers
        self.allowed_tools = allowed_tools

    @asynccontextmanager
    async def session(self, server):
        if server["transport"] != "stdio":
            raise ValueError("V0 supports MCP stdio transport")
        params = StdioServerParameters(
            command=sys.executable if server["command"] == "$PYTHON" else server["command"],
            args=server.get("args", []),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    def validate(self, tool_name: str, arguments: dict[str, Any]) -> dict:
        server = next(
            (server for server in self.servers if tool_name in server.get("tool_schemas", {})), None
        )
        if server is not None:
            Draft202012Validator(server["tool_schemas"][tool_name]).validate(arguments)
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"Tool is not allowed: {tool_name}")
        if server is None:
            raise ValueError(f"No configured schema/server for {tool_name}")
        return server

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> dict:
        server = self.validate(tool_name, arguments)
        async with self.session(server) as session:
            tools = (await session.list_tools()).tools
            remote = next((tool for tool in tools if tool.name == tool_name), None)
            if remote is None:
                raise ValueError(f"MCP server does not advertise {tool_name}")
            Draft202012Validator(remote.inputSchema).validate(arguments)
            result = await session.call_tool(tool_name, arguments)
            if result.isError:
                raise RuntimeError(str(result.content))
            return result.model_dump(mode="json", by_alias=True)
