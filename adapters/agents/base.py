from typing import Protocol

from packages.core.models.schemas import AgentAction


class AgentAdapter(Protocol):
    async def run_step(self, context: dict) -> AgentAction: ...
