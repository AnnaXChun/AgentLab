# Benchmark selection and protocols

AgentLab 0.3 uses two pinned public benchmarks. Real model runs are deferred until the user configures MODEL_BASE_URL / MODEL_API_KEY / MODEL_NAME. Environment calibration uses reference fixtures and is never reported as model performance.

| Benchmark | Pin | Input split | Protocol and primary score |
| --- | --- | --- | --- |
| [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) | `c26e151ed601ba109dc4d35e057ff8e73fec469d` | HF `verified`, revision `9c6e96c9e74572e979b0930ee735041cef528cb7` | Agent writes a self-contained Python submission; official task evaluator determines `success_rate` |
| [tau2-bench](https://github.com/sierra-research/tau2-bench) | `5bfa7e37b36656b37dc6d022156be6563c1007f3` (package 1.0.1) | telecom `base`, 114 tasks | Official **solo** ticket protocol, native assistant/user tools and `evaluate_simulation(ALL, solo_mode=True)`; primary score `reward == 1` |

ScienceAgentBench provides scientific tasks, datasets and task-specific scoring scripts. The verified release corrects false negatives; it must not be mixed with the older validation release. Its full encrypted bundle is distributed by the authors through [SharePoint](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/chen_8336_osu_edu/IQB870QrmuqwS5Ck33cHpJfkAVt3LsMeariREIwP3AT7byA?e=3ckueC). Download/unzip with the authors' password `scienceagentbench`. The official README forbids redistributing extracted benchmark data: `.benchmarks/`, results, oracle fixtures and exported task text remain local and ignored by Git.

Tau2 telecom is a multi-step customer-support tool benchmark with persistent subscriber/device state. The official solo protocol exposes a ticket and both sets of tools; no simulated user LLM is needed. This is an explicitly identified official ablation, **not the conversational tau2 protocol or its leaderboard score**. Task IDs and splits are unmodified; unsupported splits fail rather than silently dropping tasks. No task solution, expected action list, private user scenario or evaluation criteria is included in the model prompt. Reference actions are used only by the separately labelled `smoke` command.

The latest tau2 import graph loads some audio support even in text mode. The optional dependency group includes `websockets`, `scipy`, and `pydub` to make these imports work; it does not enable voice or user simulation.

ScienceAgentBench has visualization tasks whose *official* grader itself uses GPT-4o. Numeric/programmatic calibration tasks avoid that additional API dependency. We do not replace official scoring with our own LLM judge. Any unsupported evaluation dependency is reported as an evaluator error; no missing native result is replaced with an invented zero or success.

Both protocols use a plain external action loop. No planner, multi-agent coordination, retries, automatic recovery or training is implemented. A command error stops that task. Scores must be interpreted under these declared interaction limits.
