from packages.core.models.schemas import AgentAction


class MockAgentAdapter:
    """Returns declarative actions only; never performs I/O."""

    def __init__(self):
        self.step = 0

    async def run_step(self, context) -> AgentAction:
        step = self.step
        self.step += 1
        if step == 0:
            return AgentAction(
                kind="message",
                arguments={
                    "text": "I will compare candidate A's measured mean with the reference threshold."
                },
            )
        if step == 1:
            return AgentAction(
                kind="retrieve",
                arguments={"query": "candidate A property X mean threshold", "top_k": 3},
            )
        if step == 2:
            return AgentAction(
                kind="tool",
                arguments={"tool_name": "load_dataset", "arguments": {"candidate": "A"}},
            )
        if step == 3:
            return AgentAction(
                kind="tool",
                arguments={
                    "tool_name": "run_mock_simulation",
                    "arguments": {"candidate": "A", "seed": 0},
                },
            )
        if step == 4:
            data = context["observation"]["structuredContent"]
            code = (
                "import json,statistics,pathlib; data="
                + repr(data)
                + "; pathlib.Path('result.json').write_text(json.dumps({'candidate':data['candidate'],'mean':statistics.mean(data['values']),'seed':data['seed']}))"
            )
            return AgentAction(
                kind="sandbox",
                arguments={
                    "argv": ["python", "experiment.py"],
                    "files": {"experiment.py": code},
                    "collect": ["result.json"],
                    "seed": 0,
                },
            )
        if step == 5:
            return AgentAction(
                kind="claim",
                arguments={
                    "text": "Candidate A satisfies property X: mean < 0.5 (synthetic demo).",
                    "evidence_refs": list(context["state"]["evidence"]),
                    "artifact_refs": [context["observation"]["files"]["result.json"]],
                },
            )
        if step == 6:
            claim = context["observation"]
            return AgentAction(
                kind="verify",
                arguments={
                    "claim_id": claim["claim_id"],
                    "verifier_type": "numeric",
                    "artifact_id": claim["artifact_refs"][0],
                    "criteria": {"field": "mean", "operator": "lt", "threshold": 0.5},
                },
            )
        return AgentAction(kind="finish")
