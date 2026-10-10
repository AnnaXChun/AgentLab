"""Real Docker/runtime smoke with SCRIPTED tokens. This is not a veRL training run."""

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from experiment_infra.integrations.verl.loop import LoopLimits, run_loop
from experiment_infra.integrations.verl.session import make_session
from experiment_infra.integrations.verl.summary import write_sample


class ScriptedHost:
    def __init__(self):
        self.actions = [
            json.dumps(
                {
                    "tool": "environment",
                    "operation": "exec",
                    "arguments": {
                        "command": {"argv": ["python", "-c", "print(open('inputs.json').read())"]}
                    },
                }
            ),
            json.dumps(
                {
                    "tool": "environment",
                    "operation": "exec",
                    "arguments": {
                        "command": {
                            "argv": [
                                "python",
                                "-c",
                                "import json; x=json.load(open('inputs.json')); json.dump({'sum':sum(x['numbers'])},open('answer.json','w'))",
                            ]
                        }
                    },
                }
            ),
            '{"tool":"environment","operation":"finish","arguments":{}}',
        ]
        self.calls = 0
        self.server_manager = self
        self.tokenizer = SimpleNamespace(decode=lambda ids, **_: self.actions[ids[0] - 100])

    async def ct_build_initial_tokens(self, messages):
        return [1, 2, 3]

    async def generate(self, **kwargs):
        i = self.calls
        self.calls += 1
        return SimpleNamespace(
            token_ids=[100 + i, 999],
            log_probs=[-0.3, -0.2],
            extra_fields={"min_global_steps": 0, "max_global_steps": 0},
        )

    async def ct_merge_assistant_token(self, tokens, generated, mask, probs, assistant_logprobs):
        result = SimpleNamespace(
            token_ids=tokens + generated, inserted_token_ids=[], removed_prefix_token_count=0
        )
        return result, mask + [1] * len(generated), (probs or []) + assistant_logprobs

    async def ct_merge_context_msg(self, previous, updated, tokens, mask, probs):
        result = SimpleNamespace(
            token_ids=tokens + [44, 45], inserted_token_ids=[], removed_prefix_token_count=0
        )
        return result, mask + [0, 0], (probs or []) + [0.0, 0.0]


async def main(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    info = {"benchmark": "toy", "task_id": "sum-0", "split": "train"}
    session = make_session(info, 30)
    session.source = "OFFLINE"
    try:
        trace, reward, failure, _ = await run_loop(
            ScriptedHost(),
            session,
            [],
            {},
            LoopLimits(),
            {
                "policy_id": "SCRIPTED-NOT-TRAINING",
                "session_id": "local-docker-smoke",
                "initial_checkpoint": "NONE-SCRIPTED",
            },
        )
        bundle = await session.runtime.export(trace.experiment_id, root / "exports")
        write_sample(root, trace, reward, failure, info, bundle["directory"])
        assert trace.trajectory_source == "OFFLINE"
        report = {
            "kind": "SCRIPTED_RUNTIME_SMOKE_NOT_TRAINING",
            "real_docker": True,
            "real_policy_generation": False,
            "policy_update_verified": False,
            "reward": reward.scalar_reward,
            "failure": failure,
            "run_id": trace.run_id,
            "policy_trace_id": trace.policy_trace_id,
            "response_ids": trace.response_token_ids,
            "response_mask": trace.response_mask,
            "step_spans": [s.model_dump(mode="json") for s in trace.step_spans],
            "reward_result": reward.model_dump(mode="json"),
        }
        (root / "local_smoke.json").write_text(json.dumps(report, indent=2))
        if failure or reward.scalar_reward != 1:
            raise RuntimeError(f"Docker smoke failed: {failure}")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in {"step_spans", "reward_result", "summary"}
                },
                indent=2,
            )
        )
    finally:
        session.runtime.storage.engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    asyncio.run(main(parser.parse_args().output_dir))
