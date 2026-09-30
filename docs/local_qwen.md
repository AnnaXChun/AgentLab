# Local Qwen3-1.7B on Apple Silicon

AgentLab 0.3.1 can use a model running on the same machine without a cloud API key or paid inference endpoint. MLX runs on the Mac GPU; the benchmark environments continue to use Docker/native tools. A loopback HTTP interface connects the existing runner to the model.

## Installed and verified on 2026-09-30

- Apple M5 Pro, 48 GiB unified memory.
- Official `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Original BF16 weights, no quantization: 4,063,515,592 weight-file bytes (about 3.78 GiB), plus tokenizer and runtime/cache memory.
- `mlx-lm==0.31.3`, `mlx==0.32.3`; all inference dependencies pinned in `examples/local_qwen/requirements.txt` and installed separately under `.local-model/venv`.
- Local URL: `http://127.0.0.1:8081/v1`. No API key required. Cloud endpoints still require their configured credentials.
- Thinking disabled with `enable_thinking=false`. Server defaults: temperature 0.7, top-p 0.8, top-k 20. The runner's `--temperature` overrides that server default; use 0.7 explicitly to reproduce these checks.
- Weights are cached in the local Hugging Face cache. The serving process uses `HF_HUB_OFFLINE=1`; model inference does not contact a cloud model provider. Initial setup requires network access to download dependencies and weights.

The machine has ample memory for this model alongside the configured 12 GiB Colima VM. Model size and task-solving ability are separate: sufficient hardware does not make a 1.7B model reliable at complex scientific or tool-use tasks.

## Prepare and start

From the repository root:

```bash
uv sync --extra benchmarks
.venv/bin/python examples/local_qwen/manage.py prepare
.venv/bin/python examples/local_qwen/manage.py serve
```

`serve` runs in the foreground, binds only to loopback and loads the exact downloaded snapshot. Keep that terminal open; Ctrl-C stops it. No login/startup service is installed. The initial session's model server is running separately in the task terminal.

In another terminal, from the same repository:

```bash
source .env.local-qwen
.venv/bin/python examples/local_qwen/smoke.py
```

The generated `.env.local-qwen` configures the local URL, empty API key and absolute pinned model path. It is ignored by Git. Setup also writes `.local-model/model-manifest.json`. Model caches, virtual environments and generated results are excluded from both Git and the Docker build context.

The connectivity smoke makes one real model call, executes the generated Python action inside Docker, verifies stdout, finalizes the run and exports artifacts. The validated run `f1f149fc-5a3f-4c15-bd49-9802d102555f` computed the sum of integers 1–10 and printed `55`. Its model call used 605 prompt tokens and 78 output tokens, taking approximately 1.46 seconds including prompt processing. This is one warm connectivity measurement, not a throughput benchmark. Results are in `results/local-qwen-connectivity/check.json` and its canonical export directory.

## Run benchmark tasks

```bash
source .env.local-qwen
.venv/bin/python -m experiment_infra.benchmarks.run \
  --benchmark tau2 --limit 5 --max-steps 10 --temperature 0.7 \
  --output-dir results/qwen3-tau2-new

.venv/bin/python -m experiment_infra.benchmarks.run \
  --benchmark scienceagentbench \
  --task-id 5 --task-id 16 --task-id 40 --task-id 41 --task-id 92 \
  --limit 5 --max-steps 10 --temperature 0.7 \
  --output-dir results/qwen3-sab-new
```

Use new output directories for independent attempts; `--resume` does not retry terminal failures. Both commands query the local model. SAB visual tasks still need the upstream official visual judge configuration and are outside these numeric-task checks.

## Actual integration smoke results

The first real-model batches used temperature 0.7, seed 0, a 10-step limit, 2,048 maximum response tokens and a 120,000-byte input budget. They are distinct from the earlier reference-fixture calibration; no reference solutions were supplied to the model.

| Result | tau2 telecom-solo | ScienceAgentBench |
| --- | ---: | ---: |
| Tasks attempted | 5 | 5 |
| Successful tasks | 0 | 0 |
| Available native evaluations | 5 | 0 |
| Format failures | 5 | 2 |
| Tool execution failures | 0 | 3 |
| Canonical trajectories | 5 | 5 |
| Linked native Evidence | 5 | 0 |
| Sampled replay completed / matched | 1 / 1 | 1 / 0 |

Tau2 native evaluation rejected the unchanged task states. The SAB model failed to produce usable `solution.py` submissions; the official harness produced no per-instance results, recorded as evaluator errors in addition to the original model/tool failures. SAB's zero successful-attempt count must not be treated as five completed native scores. Its replay completed the recorded operations but did not match; that discrepancy remains visible in the export.

Batches: `results/qwen3-1.7b-tau2` and `results/qwen3-1.7b-scienceagentbench`. Combined table: `results/local-qwen/benchmark_summary.json/csv`. These are integration runs during development, not a final clean-commit paper baseline. The action prompt was clarified during integration and its optional rationale field corrected afterward. Earlier connectivity attempts and benchmark failures are retained in the event store; no automated retry or recovery was added.

MLX 0.31.3 accepts but does not enforce the `response_format` JSON hint. AgentLab enforces its action schema locally and stops invalid actions before tool execution. No free-form answer or shell command is guessed from malformed output. The model is installed and executable; these task failures reflect the tested small model/prompt/tool interface rather than insufficient host memory.

Sources: [official model](https://huggingface.co/Qwen/Qwen3-1.7B), [MLX LM](https://github.com/ml-explore/mlx-lm). Local setup follows the Qwen model card's non-thinking sampling guidance.

Final verification: **72 tests passed** with `HARNESS_INTEGRATION=1 .venv/bin/pytest -q` (55.99 seconds), including live Docker/PostgreSQL checks. Ruff lint and formatting checks pass. Tests cover keyless loopback requests, continued credential requirements for remote hosts, and the optional rationale field.
