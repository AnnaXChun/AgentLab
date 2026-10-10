# V0.4 development preview: veRL online environment bridge

The bridge connects the official veRL AgentLoop to ExperimentRuntime and records
policy tokens, environment steps and verified rewards. Local adapter tests and a
real Docker/runtime smoke pass. **Real GPU training has not been executed.** No
Linux/NVIDIA server was available during local validation; the user now plans
to continue on a GPU host. See [GPU handoff](../HANDOFF.md). Policy updates, a
trained checkpoint, and before/after benchmark success rates remain unverified. This is
`0.4.0.dev0`, not completion of the V0.4 online-GRPO acceptance criteria.

## Official API and dependency boundary

Pinned upstream: veRL `0.10.0.dev`, commit
[`55a06a164d3d308b407fa9063bcfef25a41f0ea4`](https://github.com/volcengine/verl/tree/55a06a164d3d308b407fa9063bcfef25a41f0ea4).
No upstream source or optimizer was modified. The checked-out upstream tree is
local/ignored; the optional dependency uses the exact Git commit.

Production extension points:

- `AgentLoopBase.run(sampling_params, **kwargs)` and `AgentLoopOutput`, registered
  as `experiment_runtime` through `agent.agent_loop_config_path`.
- `self.server_manager.generate`: uses veRL's inference server, original
  `TokenOutput.token_ids`, `log_probs`, and reported weight version.
- `ct_build_initial_tokens`, `ct_merge_assistant_token`, `ct_merge_context_msg`:
  upstream Continuous Token handles prompt and observation tokenization and mask
  alignment. Generated text is decoded for parsing, never re-tokenized by this
  bridge.
- `AgentLoopOutput.reward_score`: the official precomputed in-loop reward
  extension. Upstream `_agent_loop_postprocess` bypasses RewardLoop when this
  field is populated. This avoids running the verifier twice. No custom trainer,
  reward model or advantage function is installed.
- `verl.trainer.main_ppo`: unchanged vanilla GRPO trainer. Standard configuration
  selects `algorithm.adv_estimator=grpo`, async rollout mode, and raw chat input.

The distribution is now named `experiment-infra` (previously
`scientific-harness`); Python imports remain `experiment_infra`. Source-install
commands, without assuming a PyPI publication:

```bash
python -m pip install .                 # core: no veRL, Ray or Torch
python -m pip install '.[verl]'          # adapter dependencies, Python 3.12
python -m pip install '.[verl,benchmarks]' # also tau2/SAB adapter dependencies
```

After publication these correspond to `pip install experiment-infra` and
`pip install 'experiment-infra[verl]'`. A clean base-only virtual environment was
installed and verified to import Runtime/PolicyTrace/RewardResult with no
`verl`, `ray`, or `torch` installed. GPU backend installation is separate: the
training server needs the pinned upstream-compatible Linux NVIDIA/FSDP/vLLM
stack. The extra and full GPU backend combination have **not** been installed
or exercised on a GPU server. macOS runs schema, bridge, mock and Docker tests;
the training launcher explicitly rejects macOS before creating any output.

## Implementation map

New bridge modules under `src/experiment_infra/integrations/verl/`:

| File | Responsibility |
| --- | --- |
| `__init__.py` | Pin; no eager optional imports |
| `agent_loop.py` | Official veRL subclass, output and in-loop reward wiring |
| `loop.py` | Strict action loop and token bookkeeping, tested with mock client |
| `policy_trace.py` | CT merge validation, policy identity, on-policy isolation, persistence |
| `session.py` | Runtime lifecycle, toy adapter, deterministic failure verification |
| `reward_bridge.py` | Verification → Evidence → RewardResult |
| `dataset.py` | Fresh benchmark task inputs in Parquet, disjoint train/test IDs |
| `summary.py` | Per-rollout records, verified bundle audit, JSON/CSV batch statistics |
| `launch.py` | GPU preflight and official trainer subprocess |
| `audit_training.py` | Actual parameter-value comparison and training acceptance |
| `local_smoke.py` | Explicitly scripted tokens with real Docker, never training |

Other new files: `configs/verl/{agent_loops,_grpo_base,toy_grpo,benchmark_grpo}.yaml`,
`tests/verl/{test_bridge,test_gpu}.py`, and generated PolicyTrace/RewardResult
JSON schemas. Core changes add schemas, immutable ledger objects and export
files, with optional identity/source fields on Run/Trajectory. Old records
load as `OFFLINE`. Exact replay and branch runs explicitly use `REPLAY` and
`BRANCH`. Core imports none of veRL, Ray, distributed Torch or CUDA.

## Token and policy identity contract

`PolicyTrace` contains identity (`policy_id`, `policy_version`, `checkpoint_id`,
`training_step`, `rollout_batch_id`), experiment/run/trajectory IDs, exact prompt
and response token IDs, response mask/logprobs, sampling parameters, timestamp,
and `step_spans`. Its schema remains additive to public schema version 2.0.

Each `PolicyStepSpan` stores a half-open response interval `[start,end)`, an
optional action interval, optional observation interval, and the corresponding
TrajectoryStep ID. Every policy token belongs to a recorded generation interval.
Action intervals contain only mask=1; observation intervals contain only mask=0.
Finish decisions have generation/action spans and no environment-step ID.
Malformed generations have no action span. Driver-owned evaluation is present
in the environment trajectory but is not attributed to policy tokens.

V0.4 requires thinking disabled and a complete JSON action as each generated
turn. Thus the action span is the whole structured-action generation, including
any server-supplied end token. It does not claim a separately inferred substring
span inside natural-language reasoning. Duplicate JSON keys, markdown fences and
prose fail strict parsing. `allow_repair_once` defaults to false; when enabled,
one explicit error observation and one additional generation are allowed.

The bridge retains the exact masks/logprobs returned by official CT methods.
Qwen boundary newlines and appended observation/context tokens have mask=0.
A CT merge that rewrites a previous prefix is rejected; this preview supports
append-only builders such as the pinned Qwen builder. No guessed offsets,
retroactive re-tokenization, or unrecorded token truncation is permitted. An
observation exceeding the response budget remains in the environment record;
it is not appended to the policy stream and its token span is null.

Policy identity comes from the inference server's `min_global_steps` and
`max_global_steps`, which must be equal and nonnegative. Missing versions or a
version change within a rollout fail closed. Initial checkpoint identity is a
hash of actual tensors; later checkpoint identities are session-scoped logical
actor/global-step references. `training_step` means the version of generating
weights, not the upcoming optimizer-step number. In this synchronous trainer,
all rollout groups for that weight version belong to the session/version batch.
Only one AgentLoop worker is supported. No separate-async training or distributed
environment workers are implemented.

`validate_on_policy_batch` rejects OFFLINE, REPLAY, BRANCH, scripted tokens and
mixed identities/batches. Training datasets contain task prompts only; they do
not ingest historical trajectory/SFT files. Validation uses disjoint task IDs
and is summarized separately from training. The trainer controls which outputs
are validation versus training; the bridge does not optimize policy weights.

## Reward and failures

`RewardResult` records scalar reward, named reward signals, verification IDs,
evidence IDs, source, scope and metadata. Main GRPO reward is 1 only when all
selected deterministic verifications pass, otherwise 0. Native benchmark scores
are also saved as `benchmark_score` without altering the official evaluator.

Successful rewards link to immutable VerificationResult and Evidence, then to
runtime-produced artifacts/observations. Format, environment, budget and
verifier failures produce a terminal zero-reward rollout and a bridge-owned
failure receipt with failed verification/evidence. These are explicitly
`bridge/terminal_failure` evidence, not fabricated official benchmark results.
A failure before any inference tokens/weight identity, storage corruption, or
policy/token integrity violation aborts rather than manufacturing a trainable
sample. Cleanup/storage failures may also surface; recovery is not implemented.

The summary auditor validates canonical file/CAS hashes, exact exported
trace/reward identity and verification/evidence inputs before counting valid
reward provenance. Raw logs retain the official trainer's available loss, KL,
reward and gradient metrics; no placeholder numbers are reported.

## Local validation and sample

```bash
uv sync --extra benchmarks
.venv/bin/pytest tests/verl -q
.venv/bin/python -m experiment_infra.integrations.verl.local_smoke \
  --output-dir results/verl-local-smoke-NEW
```

The last command uses existing PostgreSQL and Docker (`docker compose up -d`).
It reads `inputs.json`, writes `answer.json`, invokes a driver-owned deterministic
verifier and exports canonical provenance. Tokens are **scripted test IDs**, not
Qwen or veRL generations. The resulting Run/Trajectory/PolicyTrace are OFFLINE
and cannot enter GRPO.

Observed local sample on 2026-10-10:

```text
Task: sum-0; initial policy: NONE-SCRIPTED (no checkpoint)
Prompt IDs: [1,2,3]
Response IDs: [100,999,44,45,101,999,44,45,102,999]
Mask:         [1,  1, 0, 0,  1,  1, 0, 0,  1,  1]

Action #1 [0,2) → read inputs → observation [2,4), mask=0
Action #2 [4,6) → write answer → observation [6,8), mask=0
Finish    [8,10)
Official runtime evaluation receipt → PASSED Evidence → RewardResult=1
GRPO batch / backprop / policy update: not executed
```

Local detailed IDs, PolicyStepSpans and RewardResult are in
`results/verl-local-smoke-final-20261010/local_smoke.json`; canonical exports
are alongside. This sample demonstrates runtime integration, not the requested
real training sample. The real sample remains a GPU acceptance requirement.

The tau2 dataset bridge has prepared 8 official train tasks and 5 official test
tasks at `results/verl-tau2-inputs/`, with disjoint task IDs and no hidden criteria
in prompts. No benchmark RL rollout or before/after training evaluation has run.

## GPU commands and acceptance

Prepare a Linux/NVIDIA host with Python 3.12, the pinned upstream-compatible
vLLM/FSDP dependencies, Docker, PostgreSQL, the local HF model directory and the
project installed with `.[verl,benchmarks]`. The launcher verifies the veRL Git
commit in package metadata. Commands below are supplied for GPU validation;
they have not yet been executed on such a host.

```bash
python -m experiment_infra.integrations.verl.launch \
  --config configs/verl/toy_grpo.yaml \
  --model /models/Qwen3-1.7B \
  --output-dir results/verl-toy-grpo

# Only after toy acceptance passes:
python -m experiment_infra.integrations.verl.launch \
  --config configs/verl/benchmark_grpo.yaml \
  --model /models/Qwen3-1.7B \
  --output-dir results/verl-tau2-grpo

VERL_INTEGRATION=1 VERL_TEST_MODEL=/models/Qwen3-1.7B \
  python -m pytest tests/verl/test_gpu.py -q
```

Default configs use Qwen3-1.7B, batch size 4, rollout n=4, three GRPO optimization
steps, learning rate 1e-6, before-training validation, per-step validation and
checkpoint saving (including HF weights). Hydra overrides configure these
parameters, generation limits, temperature, task counts and environment timeout.
Each invocation uses a new session and output directory; no implicit resume.

Outputs include `initial_policy.json`, `trainer.log`, `launch.json`, checkpoints,
per-rollout samples/canonical bundles, and `verl_training_summary.json/csv`.
Statistics include task/rollout/success/failure counts, reward, action/policy/tool
lengths, failure classes, valid traces and verified reward provenance, grouped
by policy version and split.

Acceptance reads the actual initial and final safetensors, compares tensor names,
shapes and values after dtype normalization, rejects nonfinite parameters, and
requires changed parameters plus different hashes. At least two saved optimizer
steps and distinct before/after evaluation versions are required. Only then is
`policy_update_evidence.json` written, with before/after rates, training reward
curve, rollout counts and checkpoint hashes. All-zero/equal-reward GRPO groups
can produce no learning; the audit rejects an unchanged policy rather than
changing the reward to force an apparent update.

## Recorded checks (2026-10-10)

- Full suite with live infrastructure: **102 passed, 4 skipped**. Three skips
  are the existing isolated-MLX tests; one is the opt-in GPU training test.
- Final bridge suite after metadata review: **20 passed, 1 GPU test skipped**.
- Both Hydra configurations composed against the pinned official config tree.
- The exact upstream `AgentLoopOutput` Pydantic class schema accepted the bridge
  fields in an isolated schema check. This does not certify a full installed
  AgentLoop worker or GPU backend.
- Real Docker scripted smoke: reward 1; OFFLINE source; exact mask/span and
  canonical provenance exported. No optimization occurred.
- Clean base-only package installation/import; Ruff, formatting, whitespace and
  lock consistency checks passed.

## Remaining acceptance and next stage

Verified locally: base-package independence, official output Pydantic-schema compatibility, schema round trip, masks/spans,
reward provenance, source/version isolation, malformed action and runtime failure
handling, and real Docker execution with scripted tokens.

Still required: optional adapter/backend installation and official AgentLoop
execution on Linux GPU; real toy GRPO updates; saved changed checkpoints; real
tau2 subset GRPO; before/after metrics; one real training-sample trace. No
improvement, GPU memory requirement, or training completion is claimed yet.

For a later Branch Replay RL phase, retain the original snapshot/step and policy
version, attach each alternative to a new BRANCH run and independently verified
reward, and define its sampling/credit rule separately. Vanilla GRPO should
continue rejecting those branch samples until that algorithm is explicitly
implemented and validated. No branch training, token credit, reward model,
custom advantage, multi-agent RL, UI, Slurm or Kubernetes is included here.
