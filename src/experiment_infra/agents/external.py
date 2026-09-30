from collections.abc import Awaitable, Callable

from experiment_infra.core.schema import Action


class ExternalAgentBackend:
    """Bridge a caller-supplied agent; no prompt, planner or model lives here."""

    def __init__(self, callback: Callable[[dict], Awaitable[Action]]):
        self.callback = callback

    async def next_action(self, state: dict) -> Action:
        return Action.model_validate(await self.callback(state))
