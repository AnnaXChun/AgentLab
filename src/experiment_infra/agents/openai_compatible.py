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


class NativeToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    tool: str = Field(min_length=1, max_length=200)
    arguments: dict

    def action(self, available, seed):
        from jsonschema import validate

        if self.tool == "finish":
            if self.arguments:
                raise AgentFormatError("finish arguments must be empty")
            return Action(tool="environment", operation="finish")
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in available["tools"]}
        if self.tool not in schemas:
            raise AgentFormatError("Unknown native tool")
        validate(self.arguments, schemas[self.tool])
        if available.get("python_workspace"):
            code = self.arguments["code"]
            command = {"argv": ["python", "-c", code], "seed": seed}
            if self.tool == "submit_python":
                command.update(argv=["python", "solution.py"], files={"solution.py": code})
            return Action(tool="environment", operation="exec", arguments={"command": command})
        return Action(
            tool="environment",
            operation="exec",
            arguments={
                "command": {
                    "argv": [
                        available["native_dispatch_prefix"],
                        self.tool,
                        json.dumps(self.arguments),
                    ],
                    "seed": seed,
                }
            },
        )


def compact_schema(value):
    if isinstance(value, dict):
        return {
            k: {name: compact_schema(schema) for name, schema in v.items()}
            if k in {"properties", "$defs", "definitions"}
            else v
            if k in {"default", "const", "enum", "examples"}
            else compact_schema(v)
            for k, v in value.items()
            if k != "title"
        }
    if isinstance(value, list):
        return [compact_schema(v) for v in value]
    return value


def native_output_schema(available):
    def inline(value, root):
        if isinstance(value, dict):
            if "$ref" in value:
                ref = value["$ref"]
                if not ref.startswith("#/"):
                    raise ValueError("Only local tool schema references are supported")
                target = root
                for part in ref[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
                return inline(target, root)
            return {
                k: inline(v, root) for k, v in value.items() if k not in {"$defs", "definitions"}
            }
        if isinstance(value, list):
            return [inline(v, root) for v in value]
        return value

    functions = [t["function"] for t in available["tools"]] + [
        {
            "name": "finish",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    ]
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "tool": {"const": f["name"]},
                    "arguments": inline(f["parameters"], f["parameters"]),
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            }
            for f in functions
        ]
    }


class OpenAICompatibleAgentBackend:
    """Chat Completions JSON mode plus mandatory local schema validation.

    JSON-object mode is the default; JSON Schema decoding is explicitly opt-in.
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
        self.response_format = os.getenv("MODEL_RESPONSE_FORMAT", "json_object")
        if self.response_format not in {"json_object", "json_schema"}:
            raise ValueError("MODEL_RESPONSE_FORMAT must be json_object or json_schema")
        self.owns_client = client is None
        self.last_call = None
        self.last_request = None

    async def close(self):
        if self.owns_client:
            await self.client.aclose()

    async def next_action(self, state):
        self.last_call = None
        self.last_request = None
        available = state.get("available_tools", {})
        native = isinstance(available, dict) and bool(
            available.get("native_dispatch_prefix") or available.get("python_workspace")
        )
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
        if native:
            instructions = (
                "Complete the user's task by calling the available tools, following the policy. "
                "Return exactly one JSON object with two fields: tool (a listed tool name) and "
                "arguments (an object with that tool's named parameters). No markdown or reasoning text. "
                'Shape: {"tool":"tool_name","arguments":{"parameter_name":"value"}}. '
                "Use the actual tool and parameter names from the definitions. Arguments are an object, "
                "never a string or an argv array. Inspect results and continue making progress. "
                "Do not repeat an identical query when its answer is already in the conversation. "
                "Use returned identifiers and observations to choose the next useful action. "
                'When the requested outcome is verified, return {"tool":"finish","arguments":{}}. '
                "Never call an evaluator or assume a tool succeeded without observing its result."
            )
            instructions += "\nCopy the tool name exactly from this list: " + json.dumps(
                [t["function"]["name"] for t in available["tools"]] + ["finish"]
            )
            if available.get("python_workspace"):
                instructions += (
                    "\nThe current directory is /work. Datasets are under benchmark/datasets (read-only). "
                    "There is initially NO solution.py. Use run_python to inspect inputs, then "
                    "submit_python to write and run your complete solution.py. Pass actual Python "
                    "source in arguments.code, not a command or a filename. Each call starts a new "
                    "Python process; variables do not persist, but files do. The evaluator runs "
                    "solution.py again in a fresh environment, so it must load the provided inputs "
                    "and produce the requested outputs itself. Never access hidden grading data."
                    "\nIMPORTANT: all paths shown in dataset_folder_tree are relative to "
                    "benchmark/datasets, NOT the workspace root. Use relative paths rooted at "
                    "benchmark/datasets for inputs and relative output paths, since the official "
                    "evaluator uses a different working directory. Create output directories as needed. "
                    "run_python does NOT save a submission. To solve the task, "
                    "call submit_python with the complete program, not run_python. Never finish "
                    "until submit_python has successfully produced the requested outputs."
                )
            state = {
                **state,
                "available_tools": [compact_schema(t["function"]) for t in available["tools"]],
            }
        elif isinstance(available, dict) and available.get("required_submission"):
            instructions += (
                "\nThe workspace initially contains no solution.py. You must create it. "
                "Use command.files with a mapping from relative path to complete Python source. "
                'File-write shape: {"files":{"example.py":"print(1)\\n"},"argv":["python","example.py"]}. '
                "This is an interface example, not the solution. Inspect the datasets, implement "
                "the task in solution.py, run it, and verify its outputs before finish. "
                "argv is a process argument list, not a shell: >, pipes and shell expansion do not work "
                "unless explicitly executed through a shell. Do not invent a pre-existing script or its flags."
            )
        history = state.get("history", [])
        public_state = {key: value for key, value in state.items() if key != "history"}
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(public_state, ensure_ascii=False)},
        ]
        for turn in history:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps(turn["action"], ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": "Tool result:\n"
                        + json.dumps(turn["observation"], ensure_ascii=False)
                        + "\nContinue from this result. Return the next action or finish if the task is complete.",
                    },
                ]
            )
        output_format = {"type": "json_object"}
        if self.response_format == "json_schema":
            schema = native_output_schema(available) if native else AgentOutput.model_json_schema()
            if not native:
                schema["properties"]["arguments"] = {
                    "type": "object",
                    "properties": {
                        "command": Command.model_json_schema(),
                    },
                    "additionalProperties": False,
                }
            output_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_action",
                    "strict": True,
                    "schema": schema,
                },
            }
        self.last_request = {
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "response_format": output_format,
            "model": self.model,
            "prompt_version": "0.3.2",
        }
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
                    "response_format": output_format,
                },
            )
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage") or {}
            self.last_call.update(
                server_fingerprint=body.get("system_fingerprint"),
                response_id=body.get("id"),
                usage={
                    k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
                if usage
                else None,
            )
            choice = body["choices"][0]
            self.last_call["finish_reason"] = choice.get("finish_reason")
            self.last_call["refused"] = bool(choice["message"].get("refusal"))
            if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                raise AgentFormatError(
                    "Model response refused, truncated, or not a JSON completion"
                )
            if native:
                from jsonschema.exceptions import ValidationError as ToolValidationError

                parsed = NativeToolOutput.model_validate_json(choice["message"]["content"])
                self.last_call["requested_tool"] = parsed.tool
                try:
                    action = parsed.action(available, self.seed)
                except ToolValidationError as exc:
                    self.last_call["validation_errors"] = [
                        {"path": list(exc.path), "type": exc.validator}
                    ]
                    raise AgentFormatError("Native tool arguments violate schema") from exc
            else:
                parsed = AgentOutput.model_validate_json(choice["message"]["content"])
                action = parsed.action(self.seed)
            self.last_call.update(action=action.model_dump(mode="json"), status="OK")
            return action
        except AgentFormatError as exc:
            self.last_call["format_error"] = str(exc)
            raise
        except (ValidationError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValidationError):
                self.last_call["validation_errors"] = [
                    {"path": [str(p)[:80] for p in e["loc"]], "type": e["type"]}
                    for e in exc.errors(include_input=False, include_context=False)[:10]
                ]
            raise AgentFormatError("Malformed model action: " + type(exc).__name__) from exc
        finally:
            self.last_call["latency_ms"] = (perf_counter() - started) * 1000
            if self.last_call["status"] == "PENDING":
                self.last_call["status"] = "ERROR"
