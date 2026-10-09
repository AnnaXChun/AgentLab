# Qwen3-1.7B full telecom-solo base evaluation

On 2026-10-09, local Qwen3-1.7B completed all 114 tasks in the pinned tau2
telecom-solo `base` split. It passed 13 tasks (11.40%). Each task was attempted
once with a fixed configuration; no failed task was retried in this batch.
All 114 attempts produced an official native evaluation.

| Task category | Attempts | Successes | Success rate |
| --- | ---: | ---: | ---: |
| Mobile data | 36 | 1 | 2.78% |
| Phone service | 29 | 11 | 37.93% |
| MMS | 49 | 1 | 2.04% |
| Total | 114 | 13 | 11.40% |

The 101 failures comprise 96 `AGENT_REASONING_FAILURE` and 5
`TOOL_EXECUTION_FAILURE` outcomes. The former means that official completion
criteria were not met without a more specific runtime error; it is a result
classification, not a diagnosis of the model's reasoning. No format, environment,
timeout, budget or verifier failure was recorded.

Successes were concentrated in phone-service tasks. The full result therefore
supports a working execution/recording pipeline, with limited task-solving
performance from this 1.7B configuration.

## Scope and reproducibility

This covers all 114 official `base` task IDs, including the upstream train/test
subsets. It does not cover the 2,285-task `full` split, other tau2 domains or
ScienceAgentBench. Prior debugging cases are included, so this is a development
evaluation rather than a held-out estimate.

- AgentLab runtime: 0.3.3; clean source at launch,
  `355849a56d496873364397d698627869798819e7`.
- tau2: `5bfa7e37b36656b37dc6d022156be6563c1007f3`, telecom-solo protocol,
  unmodified official evaluator, `ALL` evaluation mode.
- Qwen3-1.7B official BF16 weights:
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- MLX 0.32.3, mlx-lm 0.31.3, Outlines 1.3.3; local inference, no cloud API key.
- Temperature 0.6, top-p 0.8, top-k 20, seed 0; at most 25 actions and 4,096
  response tokens per call. Thinking enabled with a 1,024-token thinking budget;
  final structured actions use JSON Schema constrained decoding.
- Input limit: 120,000 UTF-8 bytes. Exact requests, including system/user prompts,
  history and decoding configuration, are stored in the successful bundles.

During evaluation, only the offline reader and its test were changed to expose
the final `finish` decision and distinguish driver-owned evaluation steps.
Inference, runner and scoring configuration remained fixed.
See [the run command](benchmarks.md#success-only-delivery).

## Saved successful trajectories

Local output root: `results/qwen-full-tau2-20261009/`.

- `successes/index.html`: offline reader and 13 successful detail pages.
- `successful_trajectories.zip`: standalone delivery, 9,148,401 bytes, 886 files.
- `successful_runs.jsonl`: complete successful-task journals.
- `success_audit.json`: final per-run integrity and replay checks.
- `benchmark_summary.json/csv`: full 114-task statistics, retaining failures in
  the denominator.

Archive SHA-256:
`85c43e24fdbcfb5e51c31265e2b4fa2ea9bd1bd96750fa236735a4a023d878d4`.

Extract the ZIP and open `successes/index.html`. Its relative links work without
the original database or inference server. The archive includes exact prompts,
structured model returns including termination, all actions and observations,
before/after states, snapshot payloads, native evaluation, evidence, claims,
events, provenance and deterministic replay. Private model reasoning is not
saved. Absolute paths inside raw records are original-run provenance metadata.

All 13 successful runs have a `SUPPORTED` Claim linked to `PASSED` Evidence.
Every success replay completed and matched, and direct inspection of the replay
evaluator observations confirmed native score 1 and success true for all 13.
Every canonical manifest file hash and CAS payload hash passed verification;
all step snapshot references resolved to included payloads. The archive CRC and
all HTML link targets were checked.

The success-only aggregates contain 26 canonical trajectories (13 original,
13 replay), 1,060 artifact records, 13 evidence records, and 131 model execution
steps in each SFT/RL aggregate. The canonical bundles also retain driver and
replay steps for audit. No failed-task full bundle or journal is included in the
ZIP; failed outcomes remain in overall statistics. The runtime's append-only
internal audit ledger retains its original history. Generated data and task text
remain local and ignored by Git.

## Validation

- `HARNESS_INTEGRATION=1 .venv/bin/pytest -q`: 82 passed, 3 skipped.
- The 3 MLX-specific tests passed separately in `.local-model/venv`.
- Ruff lint, Ruff formatting and Git whitespace checks passed.
- The benchmark process exited successfully after all 114 task journals reached
  `complete`; their unique IDs match the official base split exactly.
