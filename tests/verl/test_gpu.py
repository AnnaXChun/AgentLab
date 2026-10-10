"""Opt-in real official veRL training; never replace this with mocked optimizers."""

import os
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.verl_integration,
    pytest.mark.skipif(
        os.getenv("VERL_INTEGRATION") != "1", reason="Requires explicit Linux/NVIDIA GPU run"
    ),
]


def test_official_agent_loop_output_and_training(tmp_path):
    from experiment_infra.integrations.verl.launch import preflight

    preflight()
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

    from experiment_infra.integrations.verl.agent_loop import ExperimentRuntimeAgentLoop

    assert issubclass(ExperimentRuntimeAgentLoop, AgentLoopBase)
    output = AgentLoopOutput(prompt_ids=[1], response_ids=[2], response_mask=[1], metrics={})
    assert output.as_dict()["response_mask"].tolist() == [1]
    model = os.environ["VERL_TEST_MODEL"]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiment_infra.integrations.verl.launch",
            "--model",
            model,
            "--output-dir",
            str(tmp_path / "training"),
        ],
        check=True,
    )
