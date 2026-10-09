# Qwen3-1.7B benchmark adaptation — 0.3.2

Audited on 2026-10-09. A real local Qwen3-1.7B completed an official tau2 telecom-solo task: native reward **1.0**, linked PASSED Evidence, SUPPORTED Claim, and matching deterministic replay. This fulfills the single successful trajectory integration target. It is a selected success obtained during iterative development, not a held-out accuracy result or a claim that the full benchmark passes.

## Verified trajectory

The successful run in `results/qwen-adaptation-11-tau` contains eight model-generated tool executions followed by model `finish`. The canonical trajectory contains those eight executions plus the driver-owned native evaluation step. Exact requests are stored for all nine model calls, including finish.

| Object | Identifier |
| --- | --- |
| Experiment | `e8fd1b96-9ec4-48d7-a8bd-4e9327dde358` |
| Run | `4a28974c-92da-48ea-ac36-9cf05f1b369a` |
| Trajectory | `9c461a53-fd29-4dd1-914a-fbe9fefe4947` |
| Evidence | `b4d2ce20-4198-4292-a7d8-f091bd1c11d4` |
| SUPPORTED Claim | `62b286b5-c119-42ba-9b78-e3072feae4a2` |
| Replay run | `85fa9bf3-22ab-44b7-9945-c4537ba8f390` |

The model queried customer/data status, refueled data, tested speed, inspected network status, toggled airplane mode and mobile data, tested speed again, and finished. The native evaluator was `tau2.evaluator.evaluator.evaluate_simulation:ALL` at upstream commit `5bfa7e37b36656b37dc6d022156be6563c1007f3`. Both official environment assertions passed. This task's upstream reward basis is `ENV_ASSERTION`; its separate database-match diagnostic was false and remains in the raw result. We report the unchanged official reward, not a custom interpretation of that diagnostic.

Replay completed and matched all nine canonical steps, including native evaluation. The export contains seven hash-verified JSONL files, a manifest and all referenced CAS payloads. The full bundle is under:

```text
results/qwen-adaptation-11-tau/exports/
  e8fd1b96-9ec4-48d7-a8bd-4e9327dde358/
  920d5aae-2dcb-460c-ae12-3beb88026b26/
```

The two-task batch had one success and one failure. Earlier attempts remain in their own directories and the append-only ledger. No reference action sequence or gold program was passed to the model. Model-visible state contains only step, maximum steps and seed, alongside the official public task, policy and tools; task IDs encoding initial conditions remain outside prompts.

## Changes that enabled the run

1. Native tool objects replace model-authored nested command serialization. Tool names and argument types are validated before execution. The canonical Action/runtime contract is unchanged.
2. History is an alternating conversation of actions and tool observations. Internal object IDs/configuration no longer crowd the task prompt.
3. An MLX/Outlines loopback server constrains final JSON against paired tool/argument schemas. It uses original BF16 Qwen weights and allows model-native thinking before the final action.
4. Each request saves exact messages, decoding schema and sampling settings; response diagnostics include finish reason, validation failure location and server fingerprint. Private reasoning and malformed free-form replies are not retained.
5. SAB exposes `run_python` and `submit_python`; the latter writes model code unchanged. Relative dataset/output paths are now requested for compatibility with the native evaluator's working directory. Missing optional task-image metadata no longer masks native evaluator failures.

The successful configuration uses model revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, MLX 0.32.3, mlx-lm 0.31.3, Outlines 1.3.3, seed 0, temperature 0.6, top-p 0.8, top-k 20, 4,096 maximum completion tokens and 25 tool-step budget. Reproduction commands are in [local_qwen.md](local_qwen.md).

During final review the thinking limit was corrected to count tokens from MLX's first processor callback: MLX prefills most prompt tokens before that callback. The earlier calculation subtracted the full prompt length and could exceed its intended reasoning budget. The final default enables thinking and caps it at `min(1024, max_tokens // 2)` before constrained final output. Isolated MLX tests cover natural termination, forced termination and variable prefill lengths. The successful historical run used fewer than 1,024 completion tokens in every call, so this correction does not retroactively change its observed actions.

## Retained adaptation attempts

| Directory suffix | Outcome |
| --- | --- |
| `01` | Tau2 repeated queries, then format failure after 23 actions |
| `02` | Tau2 unknown tool name after 8 actions |
| `03-sab` | Missing solution script |
| `04-tau`, `06-tau` | Incorrect native argument type |
| `05-sab` | Invalid JSON before first action |
| `07-sab` | Legal JSON, wrong dataset path |
| `08-tau` | 17 legal actions; native reward 0; matching replay |
| `09-sab` | Model-generated array operation failed on incompatible shapes |
| `10-sab` | Submission failed; native evaluator also reported an absolute-path error |
| `11-tau` | First task failed; second succeeded with reward 1 and matching replay |

Directories share the prefix `results/qwen-adaptation-`. ScienceAgentBench has no verified real-model success in these attempts. It has a separately validated environment calibration; that does not count as model success. No automatic recovery, training, planner or multi-agent execution was added. A failing run still terminates; a new development attempt uses a new directory.

The historical successful run records runtime 0.3.1 and base commit `707f5395c06d5bd5503697d6359fceebbf1c240d`, with the then-uncommitted 0.3.2 adaptation and prompt version 0.3.2. It is explicitly a development run. Benchmark exports, task text, generated programs and weights stay local and ignored by Git.

## Fresh 0.3.2 confirmation

After fixing the reasoning budget and persisting the default serving configuration, a new model run in `results/qwen-v032-confirmation` also succeeded on 2026-10-09. The model generated the same eight tool actions and finish; native reward was 1.0. This was fresh inference, followed by a separate matching action replay.

| Object | Identifier |
| --- | --- |
| Experiment | `ede7b9ae-486f-411f-b067-82ee9d272355` |
| Run | `48a0a57d-4be0-482a-aebf-f4ebd0bcb12d` |
| Trajectory | `ccff17a1-e4df-4663-a3cd-75489b312d93` |
| PASSED Evidence | `b931645e-df4b-441a-aa57-d02d9ae9a2ed` |
| SUPPORTED Claim | `c32d4921-6125-49aa-8ff2-e32497de31a9` |
| Matching replay | `d730abf2-2b12-429c-9b4a-c3a7c90bb455` |

The audit verified all seven export files, all 46 CAS payload hashes, nine exact request artifacts, Evidence→Claim linkage, and the nine-step canonical trajectory. Aggregate usage was 94,388 tokens including repeated prompt context and model reasoning. There were no model, evaluation or finalization errors. `audit.json` provides a compact local record; `prompts/system_prompt.txt` and `prompts/user_prompt.json` contain the exact first-call prompts. All nine calls remain available in the canonical artifact bundle.

The confirmation records runtime 0.3.2 and ran before this release was committed; its base commit still points to 0.3.1. It validates the working tree being released, not a clean-commit accuracy baseline.

Final validation: **80 passed, 3 skipped** in the main suite with `HARNESS_INTEGRATION=1` (79.90 seconds). The three skipped tests require MLX and separately **all passed** under `.local-model/venv/bin/python`. Ruff lint, formatting and diff-whitespace checks passed. The rebuilt API is healthy and reports version 0.3.2; the loopback model service is ready.
