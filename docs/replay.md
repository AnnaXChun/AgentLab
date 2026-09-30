# Exact and branch execution replay

## Exact replay

`await runtime.replay(run_id)` starts a new Run with the original initial snapshot,
pinned Docker image reference, actual image ID/architecture/Python checks, resource
limits, configuration, seed and target code_version. The run.started event additionally records the actual current
executor source hash and dependency versions, so using a newer execution engine is
visible rather than falsely relabelled as the old engine. It submits each recorded action in its
original order to the same execution interface. Source text in actions and versioned
workspace snapshots are immutable, so current host example files are not consulted.
All replay actions, observations, artifacts and snapshots are new append-only records.

The result contains original/new step IDs, expected/actual artifact hashes, exit codes,
stdout/stderr equality and comparison policy. General commands compare logs, exit codes
and collected file hashes. A declared test command compares exit code and artifacts;
its stderr may include test-runner wall-clock timing, so log equality is separately
reported instead of making nondeterministic duration text a scientific criterion.
MCP/RAG calls run again and their structured outputs are compared. Nondeterminism or
changed external responses are reported as mismatches, never silently overwritten.

A replay's `matched` concerns these explicit recorded comparisons. It is not a claim
of process-memory identity, arbitrary distributed-state restoration or universal
scientific reproducibility. Test and metric evidence are not automatically copied
into a replay Run; an external client must explicitly request verification there.

## Branch replay

```python
branch = await runtime.branch(run_id, from_step=1, config={"N": 4000})
new_step = await runtime.execute(experiment_id, new_action, run_id=branch.run_id)
```

Step indices are one-based. A step UUID may be supplied instead. The branch restores
the state **after** that step into a fresh environment. Config overrides are written
as a new artifact version and become part of the branch's initial snapshot. The
source code and other files come from the selected workspace snapshot. The new Run
has parent_run_id and parent_step_id, and its steps have an independent branch_id.
The original Run and Trajectory are unchanged; repeated runs may test the same
Hypothesis.

Snapshot failures are visible in state_after/environment_diff. Their actions remain
in the trajectory, but the runtime rejects branching from an unrestorable step.
Later actions may repair the workspace and create a new valid snapshot.

`CounterfactualLink` records original/counterfactual step IDs, declared intervention,
context-reference changes, outcomes and optional delta. Enums reserve context_remove,
context_mask, artifact_remove, tool_result_replace, action_replace and state_restore.
This release actually restores snapshots and executes caller-chosen alternate actions.
It does not automatically perform context interventions or infer causal credit.

## Export boundaries

`trajectory.jsonl` holds one typed Trajectory per run. `artifacts.jsonl` and
`evidence.jsonl` retain immutable identity/provenance. SFT is state/context→action;
RL rows include state, action, observation, reward_signals, next_state and terminal.
No hidden chain-of-thought is required, and no rewards are fabricated.

Export includes canonical events, provenance graph and the CAS closure. A manifest
pins exported JSONL file hashes, the event cutoff and all artifact hashes. Export
completion is appended after the cutoff; it cannot describe itself inside the file
already written. This version does not yet import a portable export into a new
Postgres instance for execution replay. Replay uses the persistent live store.
