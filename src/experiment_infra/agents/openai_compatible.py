"""One structured action per API call. No retries, planning, or hidden reasoning logs."""

import json
import os
from time import perf_counter
from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from experiment_infra.core.schema import Action
from packages.sandbox.base import Command


class AgentFormatError(ValueError):
    pass


class ContextBudgetExceeded(ValueError):
    pass


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tool: Literal["environment"]
    operation: Literal["exec", "finish"]
    arguments: dict
    expected_effect: str = Field(max_length=1000)
    rationale_summary: str | None = Field(default=None, max_length=1000)

    def action(self, seed):
        if self.operation == "finish":
            if self.arguments:
                raise AgentFormatError("finish arguments must be empty")
        else:
            if set(self.arguments) != {"command"}:
                raise AgentFormatError("exec requires exactly arguments.command")
            command = Command.model_validate(self.arguments["command"], strict=True)
            command = command.model_copy(update={"seed": seed})
            self.arguments = {"command": command.model_dump(mode="json")}
        return Action(
            tool=self.tool,
            operation=self.operation,
            arguments=self.arguments,
            expected_effect=self.expected_effect,
            reasoning_summary=self.rationale_summary,
        )


class OpenAICompatibleAgentBackend:
    """Chat Completions JSON mode plus mandatory local schema validation.

    Requests JSON-object mode; servers such as MLX may ignore the format hint.
    Local schema validation remains mandatory regardless of server capabilities.
    `state` contains task/policy/context/tools supplied by the external batch driver.
    """

    def __init__(
        self,
        *,
        base_url=None,
        api_key=None,
        model=None,
        temperature=0.0,
        seed=0,
        context_limit=120000,
        max_tokens=2048,
        timeout=120,
        client=None,
    ):
        self.base_url = base_url or os.getenv("MODEL_BASE_URL")
        self.api_key = api_key or os.getenv("MODEL_API_KEY")
        self.model = model or os.getenv("MODEL_NAME")
        local = urlparse(self.base_url or "").hostname in {"localhost", "127.0.0.1", "::1"}
        if not self.base_url or not self.model or (not self.api_key and not local):
            raise ValueError("Set MODEL_BASE_URL, MODEL_API_KEY and MODEL_NAME")
        if temperature < 0 or context_limit < 1 or max_tokens < 1:
            raise ValueError("Invalid model limits")
        self.temperature, self.seed = temperature, seed
        # UTF-8 bytes is an explicit portable input budget, not an estimated token count.
        self.context_limit, self.max_tokens = context_limit, max_tokens
        self.client = client or httpx.AsyncClient(timeout=timeout)
        self.owns_client = client is None
        self.last_call = None

    async def close(self):
        if self.owns_client:
            await self.client.aclose()

    async def next_action(self, state):
        self.last_call = None
        instructions = (
            "Return exactly one JSON action matching the schema. No markdown or private "
            "chain-of-thought. rationale_summary is optional, short, and factual. "
            "Use only the available tools. exec supports an argv array, files mapping for "
            "writing files before execution, collect list for output artifacts, and "
            "timeout_seconds (1..300). Use finish with empty arguments when done. "
            "Never call evaluators or access hidden evaluation data.\n"
            'The top-level tool is always "environment"; operation is "exec" or "finish". '
            'An exec action has arguments={"command":{"argv":[...]}}. '
            "Do not put the native tool name in the top-level tool field.\n"
            + json.dumps(AgentOutput.model_json_schema())
            + "\nCommand schema: "
            + json.dumps(Command.model_json_schema())
        )
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
        ]
        if len(json.dumps(messages, ensure_ascii=False).encode()) > self.context_limit:
            raise ContextBudgetExceeded("Input context exceeds configured UTF-8 byte budget")
        started = perf_counter()
        self.last_call = {
            "model": self.model,
            "usage": None,
            "latency_ms": None,
            "response_id": None,
            "action": None,
            "status": "PENDING",
        }
        try:
            response = await self.client.post(
                self.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + self.api_key} if self.api_key else {},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage") or {}
            self.last_call.update(
                response_id=body.get("id"),
                usage={
                    k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
                if usage
                else None,
            )
            choice = body["choices"][0]
            if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                raise AgentFormatError(
                    "Model response refused, truncated, or not a JSON completion"
                )
            parsed = AgentOutput.model_validate_json(choice["message"]["content"])
            action = parsed.action(self.seed)
            self.last_call.update(action=action.model_dump(mode="json"), status="OK")
            return action
        except (ValidationError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise AgentFormatError("Malformed model action: " + type(exc).__name__) from exc
        finally:
            self.last_call["latency_ms"] = (perf_counter() - started) * 1000
            if self.last_call["status"] == "PENDING":
                self.last_call["status"] = "ERROR"
