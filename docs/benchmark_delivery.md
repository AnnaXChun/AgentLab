# AgentLab 0.3 benchmark delivery — 2026-09-30

This release integrates real benchmark environments and native evaluation with the existing runtime. Per the user's latest instruction, generation-model credentials and real-model baseline measurements are deferred. Reference-fixture calibration is labelled `environment_calibration`; it is not model performance or paper baseline data.

## 1. Benchmark choice

ScienceAgentBench supplies scientific programming tasks and task-specific official verification. The verified annotation file and upstream evaluator revision are pinned. The second benchmark is tau2 telecom **solo**, an official multi-step tool-use protocol with local persistent state and native evaluation. It requires no simulated-user model. Its results are not comparable to the conversational tau2 leaderboard. Pins, source links and rationale are in [benchmark_selection.md](benchmark_selection.md).

## 2. Installed environment

- Host: macOS arm64; Python 3.12 virtual environment, locked dependencies via `uv sync --extra benchmarks`.
- Docker: Colima profile/context `scientific-harness` / `colima-scientific-harness`, 4 CPUs, 12 GiB RAM, 80 GiB disk. PostgreSQL/pgvector and API are running through Compose; API reports 0.3.0.
- SAB upstream checkout: `.benchmarks/ScienceAgentBench`; annotation: `.benchmarks/scienceagentbench-data/verified.parquet`.
- Official encrypted `benchmark_verified.zip` downloaded from the authors' SharePoint and extracted locally to `.benchmarks/scienceagentbench-data/benchmark` (955 archive entries; approximately 4.1 GB uncompressed). No extracted data is committed.
- Official base image, built with upstream Conda configuration: `sab.base.arm64:latest`, image ID `sha256:3f109b92b0ed7394c0c3aa5e627a9ab521a8a18784004f4579e7b18632025c0c`.
- Tau2 checkout: `.benchmarks/tau2-bench`; 114 telecom base tasks available. Installed package source is checked against the pinned checkout.
- Both adapters passed five create/snapshot/restore/destroy checks without model calls.

Reproduction commands and path overrides are in [benchmarks.md](benchmarks.md). `.env.example` contains placeholders. Environment variables must be exported explicitly; no secrets file is loaded implicitly.

## 3. Thin adapter and external runner

The adapter loads tasks, provides public context/tools, prepares an environment and invokes native evaluation. A separate sequential runner calls the agent and `runtime.execute`. Tasks have isolated environments and atomic journals. A failure ends that task and the batch continues. Resume skips terminal tasks and records an interrupted task without retrying its pending action. No planner, automatic recovery, training or new core schema is introduced.

## 4. Model backend

`OpenAICompatibleAgentBackend` uses `MODEL_BASE_URL`, `MODEL_API_KEY` and `MODEL_NAME`. Each Chat Completions call requests a JSON object and undergoes strict local Pydantic validation. Only `environment.exec` and `finish` are accepted. Commands support argv, file writes and collected outputs; tau2 uses a native-tool dispatch envelope rather than a shell. Malformed, refused and truncated responses fail without repair calls. Only a short rationale summary is accepted; raw hidden reasoning is not retained. Actual API usage is stored when available; otherwise token counts remain null. The portable input-context limit is explicitly UTF-8 bytes.

## 5. Task and run mapping

The public task instruction/ticket becomes `Experiment.goal`. Each Run records the benchmark revision, task ID, split, model/backend, temperature, seed, step/context limits, runtime version and Git commit. Snapshots bind the environment image/data fingerprint. Model input context, structured response metadata, command logs and produced files enter the existing ArtifactStore and provenance graph. Hidden evaluation criteria and reference solutions are excluded from real-agent context.

## 6. Official evaluation to Evidence

The driver invokes an evaluation gateway after the action loop. SAB uses the pinned upstream `build_base_images` / `run_instances` functions and reads the actual per-instance result; a missing result is an evaluator failure, never an invented score. Tau2 uses `evaluate_simulation(ALL, solo_mode=True)`. Native scores are wrapped by a receipt-hash verifier into VerificationResult → Evidence → Claim. Only native full success supports the completion claim. Native logs and result files are retained as linked artifacts. The SAB bridge makes input mounts read-only without changing scoring functions.

## 7. Environment calibration results

The first four selected tasks use official reference actions/programs; the fifth deliberately omits required work. The intended failure tests the native rejection path. All generation-model call counts are zero. Real-model success rate is **not measured**.

| Calibration metric | ScienceAgentBench | tau2 telecom-solo |
| --- | ---: | ---: |
| Tasks attempted | 5 | 5 |
| Native reference successes | 4 | 4 |
| Deliberate native failures | 1 | 1 |
| Fixture success rate (not model accuracy) | 80% | 80% |
| Native results / linked Evidence | 5 / 5 | 5 / 5 |
| Valid trajectories / replay eligible | 5 / 5 | 5 / 5 |
| Artifact / Evidence coverage | 100% / 100% | 100% / 100% |
| Replays completed / matched | 2 / 2 | 2 / 2 |
| Average executed agent steps | 1.0 | 1.6 |
| Average runtime, including sampled replay | 167.89 s | 4.89 s |
| Generation-model calls | 0 | 0 |
| API tokens | null | null |

Selected directories: `results/environment-scienceagentbench-v2` and `results/environment-tau2-v2`. SAB IDs are `5`, `16`, `40`, `41`, `92`; ID 92 deliberately omits the expected output. Its native result reports `valid_program=0`, `success_rate=0`, with “The program does not save its output correctly.” All four reference programs receive native success. Tau2 uses the first five tasks in official `base` order; the fifth executes no tool calls.

SAB success/failure example run IDs: `7eef56f9-b337-4952-8b1c-3f5b02e99def` and `0bdb2014-cd85-46c2-b4a6-d7ef345e4db2`. Find their trajectories, evidence IDs, snapshot IDs and export paths in the selected `benchmark_runs.jsonl`. The combined local table is `results/benchmark_summary.json/csv`; `results/installed_environment.json` records installed pins and image identity.

## 8. Failure taxonomy

Structured signals map to `AGENT_REASONING_FAILURE`, `TOOL_EXECUTION_FAILURE`, `ENVIRONMENT_FAILURE`, `FORMAT_FAILURE`, `TIMEOUT`, `BUDGET_EXCEEDED`, `VERIFIER_FAILURE` or `UNKNOWN`. A normal native rejection without a more specific runtime error uses `AGENT_REASONING_FAILURE`; the deliberate fixture failure does not diagnose any model. Missing model configuration fails before creating a batch. Evaluator exceptions remain distinct from native unsuccessful scores.

Earlier SAB integration attempts exposed a missing output directory and Python import shadowing; both were fixed. Their local records remain available as integration failures and are excluded from the selected final calibration table.

## 9. Replay

Eligible trajectories have an initial snapshot and snapshots for every recorded step. The runner samples the first eligible success and failure for exact stored-action replay in fresh environments, without querying a model. Tau2 replay uses local native state transitions. SAB reruns numerical code and the official evaluator; package downloads and stochastic task code limit stronger reproducibility claims. Completion and match rates are reported separately; eligibility alone is not a replay guarantee.

## 10. Exports and examples

Each selected output directory contains `benchmark_runs.jsonl`, `failures.jsonl`, `trajectory.jsonl`, `artifacts.jsonl`, `evidence.jsonl`, `benchmark_summary.json/csv` and canonical CAS bundles under `exports/`. Aggregated calibration `sft.jsonl` / `rl_transition.jsonl` are empty; the canonical audit bundles still contain reference fixtures and must not be used for training. Model-baseline training exports preserve compatibility and exclude evaluator/replay steps.

`results/benchmark_summary.json` and `.csv` combine the explicitly selected calibration batches. All results remain local because they contain benchmark text and reference material subject to upstream redistribution restrictions. The public repository contains the integration code, tests and this report.

A failure row includes task/run/trajectory IDs, failure stage/type, failed step/action, last valid snapshot, state before failure, context references and artifact references. A native rejection can have no failed execution step: the commands ran, but the official task criteria were unmet.

## 11. Limits and verification scope

Final validation: `HARNESS_INTEGRATION=1 .venv/bin/pytest -q` — **70 passed** (57.70 s), including live PostgreSQL/Docker integration. Ruff lint/format checks pass. Compose services are healthy and the API OpenAPI version is 0.3.0. The final full test run was performed after benchmark containers had exited; the existing legacy cleanup assertion assumes no other sandbox is running on the host. Real Docker submission checks also verified both absent and present program handling.

- Real-model end-to-end behavior and vanilla model failure distribution await user credentials. Mock structured-output tests do not replace those measurements.
- Only the selected SAB tasks have been calibrated end to end. The base environment does not supply every optional task library. Official GPT visual grading is not enabled; unsupported evaluation dependencies fail explicitly.
- The SAB agent only sees read-only datasets and has no network. The native upstream evaluator reruns the submitted code with hidden grading data according to its official protocol; this is not a separate tamper-proof scoring service.
- Run SAB batches sequentially: upstream instance image tags are shared. The task runner lock protects one batch directory, not multiple processes using different directories.
- Snapshots exclude the data mount and currently support less than 15 MB of regular workspace files. Larger outputs can make runs ineligible for replay.
- Valid trajectory means canonical-schema round-trip validity. Artifact coverage measures observed persisted file versions, not deleted transient files or external services.
- These calibration runs were made during integration against the previous Git HEAD plus working-tree changes. Recorded runtime source/dependency fingerprints and local artifacts identify the execution; they are not clean-commit paper experiments.

## 12. Branch recovery handoff

A future experiment can select `failures.jsonl`, restore the recorded snapshot and use the existing branch API for an alternative action. Preserve the original trajectory and evaluate each branch independently with the same pinned native verifier. Failure data and references are ready; alternative-action generation and recovery policies are intentionally outside this release.
