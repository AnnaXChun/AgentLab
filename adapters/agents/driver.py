from packages.core.models.schemas import EventType as T
from packages.provenance.tracing import span
from packages.sandbox.base import Command


async def drive_legacy_agent(harness, agent, max_steps=20):
    observation = {}
    for _ in range(max_steps):
        context = {
            "goal": harness.store.episode(harness.episode_id).goal,
            "observation": observation,
            "state": await harness.state(),
        }
        with span("AGENT", "agent.run_step", harness.episode_id):
            action = await agent.run_step(context)
            await harness.recorder.emit(
                T.AGENT_ACTION,
                input=action.model_dump(mode="json"),
                actor_type="agent",
                actor_id=harness.store.episode(harness.episode_id).agent_id,
            )
        args = action.arguments
        if action.kind == "finish":
            return observation
        if action.kind == "message":
            await harness.recorder.emit(
                T.AGENT_MESSAGE,
                output={"text": args["text"], "visible": True},
                actor_type="agent",
            )
        elif action.kind == "retrieve":
            observation = await harness.retrieve(**args)
        elif action.kind == "tool":
            observation = await harness.tool(**args)
        elif action.kind == "sandbox":
            observation = await harness.execute(Command.model_validate(args))
        elif action.kind == "claim":
            observation = await harness.claim(**args)
        elif action.kind == "verify":
            observation = await harness.verify(**args)
    raise RuntimeError("Agent step budget exhausted")
