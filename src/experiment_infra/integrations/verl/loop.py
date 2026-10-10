"""Dependency-free loop logic exercised locally with an explicit mock client.

Production hosts must supply the pinned veRL AgentLoopBase CT methods and client.
"""

import json
from dataclasses import dataclass
from uuid import uuid4

from jsonschema import ValidationError as SchemaError

from experiment_infra.core.schema import Action, PolicyStepSpan, PolicyTrace
from experiment_infra.integrations.verl.policy_trace import (
    identity_from_server,
    persist_trace,
    validate_merge,
    validate_on_policy_batch,
)


@dataclass(frozen=True)
class LoopLimits:
    max_steps: int = 8
    max_response_length: int = 4096
    max_generation_tokens: int = 512
    allow_repair_once: bool = False

    def __post_init__(self):
        if min(self.max_steps, self.max_response_length, self.max_generation_tokens) < 1:
            raise ValueError("Loop limits must be positive")


def parse_action(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    obj = json.loads(text, object_pairs_hook=unique)
    action = Action.model_validate(obj)
    if action.tool != "environment" or action.operation not in {"exec", "finish"}:
        raise ValueError("Policy can only execute environment actions or finish")
    if action.operation == "finish" and action.arguments:
        raise ValueError("finish takes no arguments")
    if action.operation == "exec":
        from packages.sandbox.base import Command

        if set(action.arguments) != {"command"}:
            raise ValueError("exec requires exactly command")
        Command.model_validate(action.arguments["command"])
    return action


async def run_loop(host, session, messages, sampling_params, limits, identity_config):
    messages = [dict(m) for m in messages]
    tokens = await host.ct_build_initial_tokens(messages)
    prompt_ids = list(tokens)
    mask, logprobs, spans = [], None, []
    failure, identity, repaired, turns = None, None, False, 0
    request_id = uuid4().hex
    try:
        while True:
            remaining = limits.max_response_length - len(mask)
            if remaining <= 0:
                failure = "BUDGET_EXCEEDED"
                break
            params = {**sampling_params, "max_tokens": min(remaining, limits.max_generation_tokens)}
            try:
                output = await host.server_manager.generate(
                    request_id=request_id, prompt_ids=list(tokens), sampling_params=params
                )
            except Exception:
                # Without any server identity/token result there is no trainable sample.
                if identity is None:
                    raise
                failure = "GENERATION_FAILURE"
                break
            current = identity_from_server(output.extra_fields, **identity_config)
            if identity is not None and identity != current:
                raise ValueError("Weight version changed within an online rollout")
            identity = current
            generated = list(output.token_ids)
            if not generated or len(generated) > remaining:
                raise ValueError("Inference server returned empty or over-budget tokens")
            merged, new_mask, new_probs = await host.ct_merge_assistant_token(
                tokens,
                generated,
                mask,
                ([] if logprobs is None and output.log_probs is not None else logprobs),
                assistant_logprobs=output.log_probs if output.log_probs is not None else None,
            )
            start, end = validate_merge(tokens, mask, merged, new_mask, generated)
            tokens, mask, logprobs = merged.token_ids, new_mask, new_probs
            spans.append(PolicyStepSpan(response_token_start=start, response_token_end=end))
            turns += 1
            if session.run is None:
                try:
                    await session.start(identity)
                except Exception:
                    if session.run is None:
                        raise
                    failure = "ENVIRONMENT_FAILURE"
                    break
            text = host.tokenizer.decode(generated, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": text})
            try:
                action = parse_action(text)
                session.adapter.validate_action(session.task, action)
            except (ValueError, KeyError, TypeError, SchemaError):
                if not limits.allow_repair_once or repaired:
                    failure = "FORMAT_FAILURE"
                    break
                repaired = True
                observation = {
                    "error": "FORMAT_FAILURE",
                    "instruction": "Return exactly one valid action JSON object.",
                }
            else:
                spans[-1] = spans[-1].model_copy(
                    update={"action_token_start": start, "action_token_end": end}
                )
                if action.operation == "finish":
                    break
                if len(session.policy_steps) >= limits.max_steps:
                    failure = "BUDGET_EXCEEDED"
                    break
                try:
                    step = await session.execute(action)
                except Exception:
                    failure = "ENVIRONMENT_FAILURE"
                    break
                spans[-1] = spans[-1].model_copy(update={"trajectory_step_id": step.step_id})
                observation = step.observation.model_dump(mode="json")
                if step.observation.exit_code != 0 or "snapshot_error" in step.state_after:
                    failure = "ENVIRONMENT_FAILURE"
            previous = list(messages)
            # JSON observations are appended as context by veRL's own token builder.
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps({"tool_observation": observation}, ensure_ascii=False),
                }
            )
            merged, new_mask, new_probs = await host.ct_merge_context_msg(
                previous, messages, tokens, mask, logprobs
            )
            start, end = validate_merge(tokens, mask, merged, new_mask)
            if end > limits.max_response_length:
                # Keep the whole observation in the environment trajectory; never truncate
                # existing policy tokens or save spans referring to discarded context.
                failure = failure or "BUDGET_EXCEEDED"
                break
            tokens, mask, logprobs = merged.token_ids, new_mask, new_probs
            if start != end:
                spans[-1] = spans[-1].model_copy(
                    update={"tool_response_token_start": start, "tool_response_token_end": end}
                )
            if failure:
                break
        reward, failure = await session.verify(failure)
        trace = PolicyTrace(
            **identity.model_dump(),
            experiment_id=session.experiment.experiment_id,
            run_id=session.run.run_id,
            trajectory_id=session.run.trajectory_id,
            trajectory_source=session.run.trajectory_source,
            prompt_token_ids=prompt_ids,
            response_token_ids=tokens[len(prompt_ids) :],
            response_mask=mask,
            response_logprobs=logprobs,
            step_spans=spans,
            sampling_params={
                **sampling_params,
                "max_generation_tokens": limits.max_generation_tokens,
            },
        )
        if trace.trajectory_source == "ONLINE_RL":
            validate_on_policy_batch([trace])
        await persist_trace(session.runtime, trace)
        return trace, reward, failure, turns
    finally:
        await session.close()
