# Feedback-driven experiment milestone

This additive v2 slice keeps action selection outside the runtime. The existing toy
script and legacy records remain readable and executable; they do not acquire a
fictional observation history or evaluator trust after the fact.

## Decision inputs

1. The external client obtains `get_feedback(experiment_id, run_id)`: current state,
   prior observations and verification results already produced for that run.
2. It assembles its actual input, including prompt/tool text and any other context,
   then calls `capture_decision_context(..., payload, coverage="complete")` before
   invoking its backend with the returned `context.payload`.
3. It submits the chosen Action with `decision_context_id=context.context_id`.

DecisionContext stores canonical JSON, SHA-256, source references, run and preceding
step. Its identity cannot be rewritten in the append-only ledger. Payload strings
and tuple references avoid mutable nested model state; `.payload` returns a fresh
copy. Cross-run and stale step bindings are rejected. Separate contexts are separate
versions; no previous record is replaced. HTTP equivalents are POST
`/experiments/{id}/decision-contexts` and GET `/experiments/{id}/feedback?run_id=...`.

Coverage is **client-attested completeness**, not proof that a remote model saw the
payload or that an arbitrary client disclosed all inputs. IDs cannot authenticate
hidden client prompts. Context source references validate existing provenance, not
truth of arbitrary text. The runtime never manufactures missing inputs from later
ledger state. Capture before invocation is the documented client protocol; no
security boundary prevents a malicious SDK host fabricating a payload.

## Evidence and authority

Structured experiments declare typed criteria and verifier policies; hypotheses
name their criterion IDs. Bound claims identify the hypothesis and complete criterion
set, with the registered statement. Evidence is checked against hypothesis, criterion,
policy digest, source run and immutable result. Missing/insufficient evidence remains
inconclusive. Legacy free-text aggregation retains an explicit `unbound_legacy` label.

Built-in metric checks read agent-produced metrics (`agent_reported`), and built-in
test checks inspect runtime command logs (`runtime_observed`). Neither is protected
scientific evaluation. A host may inject a HostEvaluator callback that evaluates
artifacts against separately owned fixtures/logic. Public HTTP requests cannot
register evaluator code or promote trust using artifact metadata. Host ownership is
a deployment responsibility, not a sandbox, attestation service, or proof of scientific
validity. A callback that simply accepts agent metrics is not an independent evaluator.

## Training exports

Canonical trajectories/events remain lossless. Training rows include run purpose,
lineage group, split and weight. Exact replay is excluded by default; opting in uses
an explicit replay weight. Default SFT includes only captured contexts whose client
marked coverage complete. Partial/unknown contexts require opt-in and remain labeled;
missing payload remains null, never reconstructed. RL states are captured decision
inputs, next state is the next captured input or null, and unmeasured rewards stay
empty. Verifier feedback attached after execution is outcome metadata, never silently
inserted into the earlier policy state.

All descendant experiments and their runs/branches/replays use the root experiment
as deterministic split group. Consumers must retain the same split seed/fraction and
respect the emitted group; exports cannot prevent downstream users mixing datasets
with incompatible split policies or falsely declaring independent experiments.

## Scope

The reactive demo is a deterministic interface/causality check: a failure observation
changes the next caller-selected action. It is not an LLM evaluation, scientific
benchmark, planner, training loop, or demonstration of autonomous research quality.
Docker/Postgres integration requires the documented infrastructure; in-memory tests
prove contract behavior only.

### Boundary and legacy provenance

Run records predating purpose capture load with `purpose="unknown"`. They remain in
canonical exports and are excluded from training by default; opt-in preserves the
unknown label. This avoids silently treating an old replay as a fresh experiment.
RL `episode_boundary` (and compatibility `terminal`) means only that the recorded
run has ended. A known final execution timeout gives `truncated=true`; otherwise
closed-run `terminated`/`truncated` stay null because manual `finish_run` does not
establish an MDP termination cause. Intermediate transitions have both flags false.

### Run the reactive client

With the normal database and Docker infrastructure available:

```bash
uv run python -m examples.reactive_experiment
```

For the contract-only fake environment, with no host command execution:

```bash
OTEL_TRACES_SAMPLER=always_on uv run pytest -q tests/contract/test_decision_context.py
```

The demo reports `recovered=true` after the observation-selected correction prints
6. Existing run status semantics intentionally retain `FAILED` because the first
command failed; recovery does not erase the failed observation.

## Benchmark runner integration (0.3.1)

The benchmark runner records a DecisionContext before each model call. For the
built-in OpenAI-compatible backend it captures the complete JSON request body,
including system/user messages, response schema instructions and generation
settings. Transport URL, authorization header and API key are not part of this
payload. The frozen request is submitted directly rather than rebuilding prompts
after capture. Completeness describes this client's request, not the provider's
hidden system context or actual model consumption.

Custom backend inputs are captured with unknown coverage because the driver cannot
observe any additional prompts the backend adds. Default SFT excludes those rows.
Executed model actions bind their context ID and retain the existing token usage.
Finish decisions and failed generation attempts retain context IDs in the task
journal without inventing successful execution steps. Existing resumable journals,
no-retry behavior, native evaluation receipts and calibration/model separation are
preserved. Native receipts remain in the legacy unbound assessment path; they do not
silently acquire structured host-evaluator trust from this merge.

Benchmark aggregate RL files project onto the model-action sequence separately from
canonical audit trajectories. Their `boundary_scope` is `model_action_sequence`;
completed journals mark the final retained actor row as the episode boundary even
when a later native evaluator action ends the canonical trajectory. Next input comes
from the next captured decision, including an explicit finish or failed generation
attempt when present. Finish records no invented execution and keeps natural
termination flags unknown (`agent_finish`); step budgets and known execution timeouts
mark truncation. Evaluator actions remain in the untouched canonical bundle.
