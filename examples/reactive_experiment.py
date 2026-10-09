"""A deterministic external client that reacts to recorded execution feedback.

Run ``python -m examples.reactive_experiment`` with the normal runtime's database
and Docker environment available. Contract tests inject a memory-only Environment
backend; neither the agent nor the runtime executes Python on the host.

This is a small rule-based client, not a model or proof of model reasoning.
``coverage='complete'`` attests only that the driver recorded all input it passed
to this callback. It cannot attest to an arbitrary external model's hidden input.
"""

# ruff: noqa: E402 -- also support python examples/reactive_experiment.py
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from experiment_infra.agents.external import ExternalAgentBackend
from experiment_infra.core.schema import Action
from experiment_infra.runtime.executor import ExperimentRuntime

INITIAL_ARGV = ["python", "-c", "print(total)"]
CORRECTED_ARGV = ["python", "-c", "total = sum([1, 2, 3]); print(total)"]
INSTRUCTIONS = (
    "Print the sum of 1, 2, and 3. Try the existing total first. "
    "If execution reports that total is undefined, define it before printing."
)


async def next_action(payload: dict) -> Action:
    """Choose from actual delivered observations, without a queue or hidden counter."""
    observations = payload["feedback"]["observations"]
    if not observations:
        argv = INITIAL_ARGV
        effect = "Try printing the existing total"
    else:
        last = observations[-1]
        if last["exit_code"] != 0 and "NameError: name 'total' is not defined" in last["stderr"]:
            argv = CORRECTED_ARGV
            effect = "Define the missing total identified by the previous observation"
        else:
            raise ValueError("No correction is defined for this feedback")
    return Action(
        tool="environment",
        operation="exec",
        arguments={"command": {"argv": list(argv)}},
        expected_effect=effect,
    )


async def drive_one_decision(runtime, experiment_id, run_id, agent):
    """Freeze the exact input before handing a fresh decoded copy to the agent."""
    feedback = await runtime.get_feedback(experiment_id, run_id)
    payload = {"instructions": INSTRUCTIONS, "feedback": feedback}
    source_refs = [observation["event_id"] for observation in feedback["observations"]]
    source_refs += [result["verification_id"] for result in feedback["verifier_feedback"]]
    context = await runtime.capture_decision_context(
        experiment_id, run_id, payload, coverage="complete", source_refs=source_refs
    )
    chosen = await agent.next_action(context.payload)
    action = Action(
        **chosen.model_dump(exclude={"decision_context_id"}),
        decision_context_id=context.context_id,
    )
    return await runtime.execute(experiment_id, action, run_id=run_id)


async def run_demo(runtime=None, *, environment_spec=None):
    runtime = runtime or ExperimentRuntime()
    experiment = await runtime.create_experiment(
        "React to a failed command and print the sum of 1, 2, and 3",
        agent_backend="external-feedback-reactive-client",
        environment_spec=environment_spec,
    )
    eid = experiment.experiment_id
    run = await runtime.start_run(eid, seed=0)
    agent = ExternalAgentBackend(next_action)
    steps = []
    try:
        # The driver has a safety bound. The callback chooses each command solely
        # from its payload, and the driver stops as soon as execution succeeds.
        for _ in range(3):
            step = await drive_one_decision(runtime, eid, run.run_id, agent)
            steps.append(step)
            if step.observation.exit_code == 0:
                break
        if len(steps) != 2 or steps[0].observation.exit_code == 0:
            raise RuntimeError("The demo did not exercise failure followed by correction")
        if steps[-1].observation.exit_code != 0 or steps[-1].observation.stdout.strip() != "6":
            raise RuntimeError("The feedback-selected correction did not produce 6")
    finally:
        finished = await runtime.finish_run(run.run_id)
    await runtime.finish_experiment(eid)
    summary = {
        "experiment_id": eid,
        "run_id": run.run_id,
        "trajectory_id": run.trajectory_id,
        "recovered": True,
        # Current run semantics retain FAILED when any command failed, even if a
        # later action recovered. Do not relabel that durable history as success.
        "run_status": finished.status,
        "actions": [step.action.arguments["command"]["argv"] for step in steps],
        "exit_codes": [step.observation.exit_code for step in steps],
        "stdout": steps[-1].observation.stdout,
        "decision_context_ids": [step.action.decision_context_id for step in steps],
        "correction_source_observation_id": steps[0].observation.event_id,
        "coverage": "complete (caller-attested)",
    }
    return summary


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_demo()), indent=2))
