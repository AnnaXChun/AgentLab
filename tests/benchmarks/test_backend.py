import json

import httpx
import pytest

from experiment_infra.agents.openai_compatible import (
    AgentFormatError,
    AgentOutput,
    ContextBudgetExceeded,
    OpenAICompatibleAgentBackend,
)


def payload(**updates):
    return {
        "tool": "environment",
        "operation": "exec",
        "arguments": {"command": {"argv": ["python", "solution.py"]}},
        "expected_effect": "produce output",
        "rationale_summary": "Run the saved program",
        **updates,
    }


def test_rationale_summary_is_optional():
    value = payload()
    del value["rationale_summary"]
    assert AgentOutput.model_validate(value).action(0).reasoning_summary is None


@pytest.mark.parametrize(
    "value",
    [
        "run ls",
        "```json\n{}\n```",
        "{}",
        json.dumps(payload(operation="arbitrary")),
        json.dumps(payload(arguments={"command": {"argv": "ls"}})),
        json.dumps(payload(private_chain_of_thought="not allowed")),
        json.dumps(payload(operation="finish", arguments={"command": {"argv": ["ls"]}})),
    ],
)
async def test_malformed_response_is_not_retried(value):
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": value}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        agent = OpenAICompatibleAgentBackend(
            base_url="https://model.invalid/v1", api_key="test", model="test", client=client
        )
        with pytest.raises(AgentFormatError):
            await agent.next_action({})
        assert len(calls) == 1
        assert value not in json.dumps(agent.last_call)


async def test_valid_action_and_usage_and_http_failure():
    def reply(request):
        request_body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert request_body["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "id": "r1",
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(payload())}}
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        agent = OpenAICompatibleAgentBackend(
            base_url="https://model.invalid/v1",
            api_key="test",
            model="test",
            client=client,
            seed=42,
        )
        action = await agent.next_action({"task": "test"})
        assert action.arguments["command"]["seed"] == 42
        assert agent.last_call["usage"]["total_tokens"] == 20
        assert "test" not in agent.last_call.get("api_key", "")
        agent.context_limit = 1
        with pytest.raises(ContextBudgetExceeded):
            await agent.next_action({})


@pytest.mark.parametrize("finish,refusal", [("length", None), ("stop", "refused")])
async def test_refusal_truncation(finish, refusal):
    def reply(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": finish,
                        "message": {"content": json.dumps(payload()), "refusal": refusal},
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        agent = OpenAICompatibleAgentBackend(
            base_url="https://model.invalid", api_key="test", model="test", client=client
        )
        with pytest.raises(AgentFormatError):
            await agent.next_action({})


async def test_loopback_model_needs_no_api_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    def reply(request):
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(payload())}}
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        agent = OpenAICompatibleAgentBackend(
            base_url="http://127.0.0.1:8081/v1", model="local", client=client
        )
        assert (await agent.next_action({})).operation == "exec"
        for host in ("https://remote.invalid/v1", "http://localhost.remote.invalid/v1"):
            with pytest.raises(ValueError, match="MODEL_API_KEY"):
                OpenAICompatibleAgentBackend(base_url=host, model="remote", client=client)


@pytest.mark.parametrize(
    "arguments,valid", [({"phone_number": "555"}, True), ("555", False), ({"phone": "555"}, False)]
)
async def test_native_tool_objects_are_validated_and_encoded_once(arguments, valid):
    tools = {
        "native_dispatch_prefix": "tau2",
        "tools": [
            {
                "function": {
                    "name": "lookup",
                    "description": "Look up a record",
                    "parameters": {
                        "type": "object",
                        "properties": {"phone_number": {"type": "string"}},
                        "required": ["phone_number"],
                        "additionalProperties": False,
                    },
                }
            }
        ],
    }

    def reply(request):
        body = json.loads(request.content)
        assert "argv array" in body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({"tool": "lookup", "arguments": arguments}),
                        },
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        agent = OpenAICompatibleAgentBackend(
            base_url="http://localhost/v1", model="test", client=client
        )
        state = {"available_tools": tools}
        if valid:
            action = await agent.next_action(state)
            assert action.arguments["command"]["argv"] == ["tau2", "lookup", json.dumps(arguments)]
        else:
            with pytest.raises(AgentFormatError):
                await agent.next_action(state)
            assert agent.last_call["validation_errors"]
        assert agent.last_call["finish_reason"] == "stop"
        assert len(agent.last_request["messages"]) == 2


def test_schema_compaction_preserves_parameter_names():
    from experiment_infra.agents.openai_compatible import compact_schema

    schema = {
        "title": "Parameters",
        "properties": {"title": {"type": "string", "title": "Title"}},
        "required": ["title"],
    }
    assert compact_schema(schema) == {
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }


async def test_history_is_a_conversation_and_exact_request_is_recorded():
    history = [
        {
            "action": {"tool": "lookup", "arguments": {}},
            "observation": {"exit_code": 0, "stdout": "found record", "stderr": ""},
        }
    ]

    def reply(request):
        messages = json.loads(request.content)["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert json.loads(messages[2]["content"]) == history[0]["action"]
        assert "found record" in messages[3]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(payload(operation="finish", arguments={}))
                        },
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        agent = OpenAICompatibleAgentBackend(
            base_url="http://localhost/v1", model="test", client=client
        )
        await agent.next_action({"task": "test", "history": history})
        assert agent.last_request["messages"][3]["content"].startswith("Tool result:")


def test_python_submission_is_saved_without_modifying_model_source():
    from experiment_infra.agents.openai_compatible import NativeToolOutput

    source = "from pathlib import Path\nPath('output.txt').write_text('computed')\n"
    available = {
        "python_workspace": True,
        "tools": [
            {
                "function": {
                    "name": "submit_python",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                }
            }
        ],
    }
    action = NativeToolOutput(tool="submit_python", arguments={"code": source}).action(
        available, 42
    )
    assert action.arguments["command"] == {
        "argv": ["python", "solution.py"],
        "files": {"solution.py": source},
        "seed": 42,
    }


def test_decoding_schema_keeps_tools_paired_with_argument_types():
    from jsonschema import ValidationError as SchemaError
    from jsonschema import validate

    from experiment_infra.agents.openai_compatible import native_output_schema

    available = {
        "tools": [
            {
                "function": {
                    "name": "amount",
                    "parameters": {
                        "type": "object",
                        "$defs": {"Number": {"type": "number"}},
                        "properties": {"value": {"$ref": "#/$defs/Number"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                }
            }
        ]
    }
    schema = native_output_schema(available)
    validate({"tool": "amount", "arguments": {"value": 2.0}}, schema)
    validate({"tool": "finish", "arguments": {}}, schema)
    for value in [
        {"tool": "amount", "arguments": {"value": "2.0"}},
        {"tool": "unknown", "arguments": {}},
        {"tool": "finish", "arguments": {"value": 2}},
    ]:
        with pytest.raises(SchemaError):
            validate(value, schema)
