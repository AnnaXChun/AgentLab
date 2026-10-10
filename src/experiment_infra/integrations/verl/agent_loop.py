"""The only GPU-facing module: loaded through official agent_loop_config_path."""

from pathlib import Path

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from experiment_infra.integrations.verl.loop import LoopLimits, run_loop
from experiment_infra.integrations.verl.session import make_session
from experiment_infra.integrations.verl.summary import summarize, write_sample


@register("experiment_runtime")
class ExperimentRuntimeAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        config = self.config.experiment_bridge
        if self.rollout_config.agent.num_workers != 1:
            raise ValueError(
                "V0.4 supports one local AgentLoop worker, not distributed environment workers"
            )
        if self.apply_chat_template_kwargs.get("enable_thinking") is not False:
            raise ValueError("V0.4 action span contract requires enable_thinking=false")
        session = make_session(kwargs["extra_info"], config.environment_timeout)
        try:
            trace, reward, failure, turns = await run_loop(
                self,
                session,
                kwargs["raw_prompt"],
                sampling_params,
                LoopLimits(
                    max_steps=config.max_steps,
                    max_response_length=self.rollout_config.response_length,
                    max_generation_tokens=config.max_generation_tokens,
                    allow_repair_once=config.allow_repair_once,
                ),
                {
                    "policy_id": config.policy_id,
                    "session_id": config.session_id,
                    "initial_checkpoint": config.initial_checkpoint,
                },
            )
            bundle = await session.runtime.export(
                trace.experiment_id, Path(config.output_dir) / "exports"
            )
            write_sample(
                config.output_dir, trace, reward, failure, kwargs["extra_info"], bundle["directory"]
            )
            summarize(config.output_dir)
            # Official in-loop reward extension. veRL skips RewardLoop when populated.
            return AgentLoopOutput(
                prompt_ids=trace.prompt_token_ids,
                response_ids=trace.response_token_ids,
                response_mask=trace.response_mask,
                response_logprobs=trace.response_logprobs,
                reward_score=reward.scalar_reward,
                num_turns=turns,
                metrics={},
                extra_fields={
                    "policy_trace_id": trace.policy_trace_id,
                    "reward_id": reward.reward_id,
                    "run_id": trace.run_id,
                    "failure": failure or "",
                    "min_global_steps": trace.training_step,
                    "max_global_steps": trace.training_step,
                    "policy_version": trace.policy_version,
                    "rollout_batch_id": trace.rollout_batch_id,
                    "turn_scores": [],
                    "tool_rewards": [],
                },
            )
        finally:
            session.runtime.storage.engine.dispose()
