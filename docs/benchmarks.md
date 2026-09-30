# Running benchmarks

Install Python 3.12 dependencies and start existing infrastructure:

```bash
uv sync --extra benchmarks
source .venv/bin/activate
docker compose up -d --wait
```

Keep the benchmark checkouts and data outside version control:

```bash
mkdir -p .benchmarks
git clone https://github.com/sierra-research/tau2-bench .benchmarks/tau2-bench
git -C .benchmarks/tau2-bench checkout 5bfa7e37b36656b37dc6d022156be6563c1007f3
git clone https://github.com/OSU-NLP-Group/ScienceAgentBench .benchmarks/ScienceAgentBench
git -C .benchmarks/ScienceAgentBench checkout c26e151ed601ba109dc4d35e057ff8e73fec469d
mkdir -p .benchmarks/scienceagentbench-data
curl -Lf https://huggingface.co/datasets/osunlp/ScienceAgentBench/resolve/9c6e96c9e74572e979b0930ee735041cef528cb7/data/verified-00000-of-00001.parquet -o .benchmarks/scienceagentbench-data/verified.parquet
```

Get the full `benchmark_verified.zip` from the official link in [benchmark_selection.md](benchmark_selection.md), then:

```bash
unzip -q -P scienceagentbench ~/Downloads/benchmark_verified.zip -d .benchmarks/scienceagentbench-data
python -m experiment_infra.benchmarks.prepare_sab
```

The official base image is large (Conda, Torch, TensorFlow, RDKit). Allocate sufficient Docker memory and disk. Paths can be overridden with TAU2_ROOT, SAB_ROOT, SAB_DATA, SAB_ANNOTATIONS, SAB_AGENT_IMAGE and SAB_EVALUATOR_PYTHON. See `.env.example`; no key or environment file is loaded implicitly by AgentLab.

Environment checks make no model calls:

```bash
python -m experiment_infra.benchmarks.run --benchmark tau2 --limit 5 --check-environment
python -m experiment_infra.benchmarks.run --benchmark scienceagentbench --limit 5 --check-environment
```

After setting MODEL_BASE_URL, MODEL_API_KEY and MODEL_NAME in the shell:

```bash
python -m experiment_infra.benchmarks.run --benchmark scienceagentbench --limit 5
python -m experiment_infra.benchmarks.run --benchmark tau2 --limit 5
```

Default output directories are `results/scienceagentbench` and `results/tau2`. Each contains `benchmark_summary.json/csv`, `benchmark_runs.jsonl`, `failures.jsonl`, canonical exports and immutable CAS bundles. Use `--output-dir` for a distinct experiment. `--task-id` is repeatable. Selection is the first N tasks in official split order unless IDs are supplied; there is no success-based selection.

Default model runs also refresh the combined `results/benchmark_summary.json` (array of benchmark rows) and `.csv`. For explicitly named experiment directories, combine selected batches with `python -m experiment_infra.benchmarks.report DIRECTORY_A DIRECTORY_B --output-dir results`. The report command rejects mixing calibration and model-baseline batches.

Other controls: `--seed`, `--temperature`, `--max-steps`, `--context-limit`, `--replay-samples`. Context limit is measured in UTF-8 input bytes, explicitly not a provider token estimate. Known provider usage is saved; unknown usage/cost stays null.

`--resume` validates the batch manifest and skips terminal tasks, including failures. An interrupted task is marked interrupted and finalized from its recorded state; its pending action is not replayed or retried. A new attempt needs a new output directory. A filesystem lock prevents concurrent writers to a batch. This does not add distributed worker recovery.

## Boundaries and mapping

Adapters only load task inputs, configure environments/tools and invoke evaluators. The external runner calls `agent.next_action(state)`, then one `runtime.execute` action at a time. Model response parsing uses JSON-object mode plus mandatory strict local Pydantic validation; no prose-to-command extraction or repair requests. Supported actions are `environment.exec` and `finish`. Exec argv/files cover read/write/search with existing Docker commands. Tau2 exec is a dispatch envelope `["tau2", native_tool_name, JSON_arguments]`, never a shell.

Experiment goal is the public task goal/ticket. Run.config contains benchmark version, split, task ID, model, temperature, seed, step/context limits, runtime version and Git commit. Initial snapshot records the environment fingerprint. Core schemas and database migrations remain unchanged. The only runtime extension passes optional token usage into the existing TrajectoryStep field.

Official evaluation runs through a driver-owned gateway action. The runtime produces its immutable observation/log. NativeResultVerifier checks that receipt's exact content hash and wraps the native metric in VerificationResult → Evidence → Claim. Native success alone can support the task-completed claim. An evaluator crash yields no successful claim. Model prompts contain no evaluator invocation tool.

Model input context and structured response metadata are stored as separate artifacts linked to each model action. Raw reasoning fields and malformed free-form replies are not retained. Successful and failed commands have log artifacts and workspace snapshots where supported.

## Metrics

- Success rate: native full successes / attempted tasks; missing evaluations remain failures in this denominator and are also counted separately.
- Valid trajectory rate: deserializable canonical trajectories / attempts, including evaluator-only trajectories when the model failed before a first action.
- Artifact coverage: observed produced workspace file versions captured in snapshots / observed produced versions. Tau2 measures native state checkpoints. It does not claim coverage of external services or deleted transient files.
- Evidence coverage: native results linked to Evidence / available native results. `native_results_available` also exposes evaluator coverage.
- Replay eligibility: initial snapshot exists and every recorded step has a snapshot. This is a precondition, not a replay guarantee.
- Replay samples: first N eligible native successes and failures, exact stored actions in fresh environments, no new model decisions. Report completion and match separately. Tau2 native tools are local deterministic interactions; SAB's native evaluator may need network/model downloads and task-specific stochastic numerical code.

Canonical bundles contain infrastructure/evaluator steps and all CAS bytes. Aggregated SFT/RL files include only model execution steps, excluding evaluator and replay actions. Calibration batches leave these aggregated training files empty. Their canonical bundles still contain reference programs for audit and are tagged `environment_calibration`; do not use benchmark test/oracle data in training. No private chain-of-thought or artificial rewards are exported.

Failures preserve stage/type/failed step, last valid snapshot, failed action, state before failure, context and artifact references. Types: AGENT_REASONING_FAILURE, TOOL_EXECUTION_FAILURE, ENVIRONMENT_FAILURE, FORMAT_FAILURE, TIMEOUT, BUDGET_EXCEEDED, VERIFIER_FAILURE, UNKNOWN. AGENT_REASONING_FAILURE means a completed native evaluation failed without a more specific runtime signal; it is not an LLM diagnosis.

## Recovery handoff

Next-stage recovery can select a failure row, locate its last valid snapshot and use the existing branch API to submit an alternative action. Keep original runs immutable and score branches independently with the same native evaluator. This release only saves that input data; it does not generate alternatives or train a policy.

## Environment calibration

```bash
python -m experiment_infra.benchmarks.smoke --benchmark tau2 --limit 5 --output-dir results/environment-tau2
python -m experiment_infra.benchmarks.smoke --benchmark scienceagentbench \
  --task-id 5 --task-id 16 --task-id 40 --task-id 41 --task-id 92 \
  --limit 5 --output-dir results/environment-scienceagentbench
```

The first four tasks use official reference actions; the fifth deliberately finishes immediately. This verifies both native pass/fail paths. It is not a model success-rate measurement. ScienceAgentBench calibration analogously uses selected reference programs and a deliberately missing output. Preserve the exact IDs and runtime images in reports.

The SAB native harness executes the submitted program again and uses its own hidden evaluation data. The bridge mounts native benchmark/scorer/prediction inputs read-only while keeping native result directories writable; scoring functions are unchanged. Task images are rebuilt for each submission. Run SAB evaluation batches sequentially because upstream image names are shared across batches. The agent container has no network and only sees the dataset mount. The official base image does not include every task-specific library; visual tasks requiring the official GPT judge are not enabled in this release. Workspace snapshots support less than 15 MB of regular files, excluding the dataset mount. Larger outputs require a later snapshot-storage extension.
