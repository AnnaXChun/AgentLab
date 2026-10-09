# Local Qwen3-1.7B on Apple Silicon

AgentLab 0.3.2 runs the original Qwen3-1.7B locally through MLX and Outlines. It has a verified tau2 telecom-solo success with native reward 1 and matching deterministic replay. See [the adaptation report](qwen_adaptation.md) for the successful trajectory and retained failures.

## Environment

- Apple M5 Pro, 48 GiB unified memory; Docker uses a separate 12 GiB Colima VM.
- Official `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, original BF16 weights (about 3.78 GiB).
- `mlx==0.32.3`, `mlx-lm==0.31.3`, `outlines==1.3.3`. Dependencies are pinned in `examples/local_qwen/requirements.txt` and isolated under `.local-model/venv`.
- Loopback service `http://127.0.0.1:8081/v1`, no API key. The service is single-request, nonstreaming and intended for local benchmark use.
- Thinking enabled by `manage.py serve`, bounded to `min(1024, max_tokens // 2)` generated tokens before final JSON. Reasoning is discarded; only final structured actions and token counts are returned. Total completion usage includes reasoning.
- Outlines constrains final output to the tool's JSON Schema; the client independently validates tool names and argument types. This prevents malformed actions, not task-solving mistakes.
- Top-p 0.8, top-k 20; reproduction commands explicitly set temperature 0.6 and seed 0.

Initial setup downloads dependencies and weights. Serving uses `HF_HUB_OFFLINE=1` and needs no cloud inference provider.

## Prepare and start

```bash
uv sync --extra benchmarks
.venv/bin/python examples/local_qwen/manage.py prepare
.venv/bin/python examples/local_qwen/manage.py serve
```

Keep the serving terminal open. Ctrl-C stops it; no startup service is installed. Setup writes ignored `.env.local-qwen` and `.local-model/model-manifest.json`, including model revision and serving configuration. The environment file sets `MODEL_RESPONSE_FORMAT=json_schema`, the local URL and model path. No environment file is loaded implicitly.

In another terminal:

```bash
source .env.local-qwen
.venv/bin/python examples/local_qwen/smoke.py
```

The connectivity smoke asks the model to generate a simple Python program, executes it in Docker, verifies its output and exports the run. It is separate from benchmark success.

## Reproduce the successful benchmark case

Install the pinned [benchmark environments](benchmarks.md), start the model service, then:

```bash
source .env.local-qwen
.venv/bin/python -m experiment_infra.benchmarks.run \
  --benchmark tau2 \
  --task-id '[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]' \
  --limit 1 --max-steps 25 --max-response-tokens 4096 \
  --temperature 0.6 --seed 0 \
  --output-dir results/qwen-success-new
```

The task ID selects the official task; it is not included in the model-visible state. The model receives the public ticket, official policy/tool descriptions, step budget and prior tool results. It receives no reference action sequence or hidden evaluator criteria.

Use a fresh output directory for each attempt. `--resume` skips terminal failures; it does not retry them. Default replay executes saved actions in a fresh environment, without asking the model again. Fresh model generation and deterministic action replay are different checks.

For a broader exploratory batch use `--benchmark tau2 --limit 5`; ScienceAgentBench is selected with `--benchmark scienceagentbench`. SAB numeric tasks need the full official bundle; visual judging has additional upstream requirements. A selected successful case is not a benchmark-wide accuracy estimate.

## Prompt and failure diagnostics

Each model call saves the exact system/user/history messages, response schema, temperature, seed, token limit and `prompt_version` as a `model_request` artifact, linked through the task row's `request_refs`. The implementation is in `src/experiment_infra/agents/openai_compatible.py`. Credentials and private reasoning are not stored.

The native action interface is `{"tool":"name","arguments":{...}}`. Tau2 parameters are serialized into its dispatch command by the adapter. For SAB, `run_python` inspects data and `submit_python` saves the model's complete source as `solution.py` unchanged. Relative input/output paths allow execution in both the agent workspace and the official evaluator's workspace.

The initial v0.3.1 batches (`results/qwen3-1.7b-tau2` and `results/qwen3-1.7b-scienceagentbench`) had no successes. Tau2 initially failed because its arguments were encoded incorrectly, even when the selected tool was appropriate. SAB included missing submissions and truncated/invalid replies. The native interface, conversational history, constrained decoding and explicit submission guidance address those integration faults. Remaining legal-action failures are retained and evaluated normally.

Validation commands:

```bash
HARNESS_INTEGRATION=1 .venv/bin/pytest -q
.local-model/venv/bin/python -m unittest discover -s tests/local_qwen -v
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

The MLX boundary tests run in the isolated environment and skip in ordinary non-MLX CI. Do not run Docker integration tests concurrently with SAB evaluations: a legacy integration check requires no remaining sandbox containers.

Sources: [official Qwen model](https://huggingface.co/Qwen/Qwen3-1.7B), [MLX LM](https://github.com/ml-explore/mlx-lm), [Outlines](https://github.com/dottxt-ai/outlines).
