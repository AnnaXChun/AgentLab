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
