from typing import Protocol

from experiment_infra.core.schema import Action


class AgentBackend(Protocol):
    async def next_action(self, state: dict) -> Action: ...
