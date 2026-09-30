# Agent Experiment Infrastructure 架构收敛交付报告

验证日期：2026-09-30（Asia/Shanghai）。版本：0.2.0。此次在现有 scientific-harness 上增量实现，保留旧存储、旧数据和旧测试，没有推倒重写。

完整 vertical slice 已通过真实 PostgreSQL、Docker 和独立 verifier 验证；最终 **50 passed，0 skipped**。`python examples/toy_experiment/run.py` 与 Compose 路径均已运行成功。当前 API 与 Postgres healthy，Alembic 为 `0003`。

## 1. 当前 repository 架构分析

首先交付的 [architecture_refactor.md](architecture_refactor.md) 记录了原实现的职责和改造方案。原系统没有独立 Task/Experiment/Run，Episode 承担运行单元；Environment 只有 sandbox 接口；Evidence 主要是检索片段；replay 只做事件折叠；`Harness.run_agent` 同时驱动 Agent 和执行动作。

本次直接复用了：SQLAlchemy/Postgres append-only EventStore、SHA-256 LocalArtifactStore、DockerSandboxBackend、MCP SDK Gateway、Postgres FTS/pgvector、FastAPI、OTel/OpenInference 和旧 exporter。

实际解耦：

- 外部 `AgentBackend.next_action(state)` 决策；`ExperimentRuntime.execute(...)` 只执行一个显式动作。
- `Harness.run_agent` 已删除；旧 demo 的循环移至 `adapters/agents/driver.py`，新 demo 的策略只在 examples 内。
- 新核心以 Experiment → Hypothesis → Runs → Action/Observation → Artifact → Verification/Evidence → Claim → Trajectory 组织。
- RAG 返回 context，不自动变成实验事实；MCP/RAG 与环境动作统一产生 Action、Observation、Artifact 和 TrajectoryStep。
- Environment 观察提供 provider spec/fingerprint/snapshot MIME，executor 不解析 Docker 镜像内部信息。测试用内存 backend 证明可替换；生产仅实现 Docker。
- 没有 planner、persona、通用推理框架、multi-agent 或训练子系统。

新代码主要在 `src/experiment_infra/`；旧基础实现留在原位置，避免无收益搬迁。

## 2. 实际修改文件列表

原目录尚无 Git commit，因此列表按本轮开始时的文件内容哈希比较，未虚构 Git diff。完整机器可读清单：[changed_files.json](changed_files.json)。忽略缓存、虚拟环境、CAS 和运行输出。

修改已有文件（17）：

```text
ARCHITECTURE.md
Dockerfile
GOALS.md
Makefile
README.md
adapters/sandbox/docker.py
examples/generate_schemas.py
examples/run_demo.py
packages/core/models/schemas.py
packages/sdk/api.py
packages/sdk/harness.py
packages/sdk/runtime.py
pyproject.toml
schemas/Event.json
schemas/openapi.json
tests/integration/test_live.py
uv.lock
```

新增文件（59，含生成的 v2 JSON Schemas）：

```text
README.v0.md
adapters/agents/driver.py
docs/architecture_refactor.md
docs/changed_files.json
docs/provenance.md
docs/refactor_delivery.md
docs/replay.md
docs/schema.md
examples/toy_experiment/algorithms.py
examples/toy_experiment/benchmark.py
examples/toy_experiment/run.py
examples/toy_experiment/test_algorithms.py
migrations/versions/0003_experiments.py
schemas/v2/Action.json
schemas/v2/ActionEvent.json
schemas/v2/Artifact.json
schemas/v2/Claim.json
schemas/v2/CounterfactualLink.json
schemas/v2/DomainPack.json
schemas/v2/EnvironmentSpec.json
schemas/v2/Evidence.json
schemas/v2/Experiment.json
schemas/v2/Hypothesis.json
schemas/v2/ObservationEvent.json
schemas/v2/ReplayResult.json
schemas/v2/Run.json
schemas/v2/Snapshot.json
schemas/v2/Trajectory.json
schemas/v2/TrajectoryStep.json
schemas/v2/VerificationResult.json
src/experiment_infra/__init__.py
src/experiment_infra/agents/__init__.py
src/experiment_infra/agents/base.py
src/experiment_infra/agents/external.py
src/experiment_infra/core/__init__.py
src/experiment_infra/core/database.py
src/experiment_infra/core/ledger.py
src/experiment_infra/core/schema.py
src/experiment_infra/domains/__init__.py
src/experiment_infra/domains/science_demo/__init__.py
src/experiment_infra/domains/science_demo/pack.py
src/experiment_infra/environments/__init__.py
src/experiment_infra/environments/base.py
src/experiment_infra/environments/docker.py
src/experiment_infra/exporters/__init__.py
src/experiment_infra/exporters/canonical.py
src/experiment_infra/exporters/rl.py
src/experiment_infra/exporters/sft.py
src/experiment_infra/integrations/__init__.py
src/experiment_infra/runtime/__init__.py
src/experiment_infra/runtime/api.py
src/experiment_infra/runtime/executor.py
src/experiment_infra/runtime/replay.py
src/experiment_infra/verifiers/__init__.py
src/experiment_infra/verifiers/base.py
src/experiment_infra/verifiers/metric.py
src/experiment_infra/verifiers/test.py
tests/integration/test_experiments.py
tests/unit/test_experiment_models.py
```

删除文件：无。旧 README 保存在 [README.v0.md](../README.v0.md)。旧 `/episodes` 写接口拒绝新 Experiment 分区，防止借旧 Claim 语义绕过验证。

## 3. 新 schema

定义：[core/schema.py](../src/experiment_infra/core/schema.py)。字段说明：[schema.md](schema.md)。JSON Schema：[schemas/v2](../schemas/v2)。

已实现 schema_version=2.0 的 Experiment、Hypothesis、Run、Action、ActionEvent、ObservationEvent、Artifact、Evidence、Claim、Snapshot、VerificationResult、TrajectoryStep、Trajectory、CounterfactualLink、EnvironmentSpec、DomainPack、ReplayResult。

关键语义：

- 一个 Hypothesis 可绑定多个 Run，每次 run 有 config/seed/environment snapshot/code version/resource spec。
- Artifact 对象 ID 表示不可变版本；SHA-256 表示字节内容；修改产生新 ID/hash，并记录 parent_artifact_ids 和 producer run/step。
- Evidence 必须引用 Artifact、Run、Observation 和独立 VerificationResult。
- Claim 状态只有 SUPPORTED / REFUTED / INCONCLUSIVE / UNVERIFIED。无 evidence 只能 UNVERIFIED；全部成功 verifier evidence 才能 SUPPORTED。Agent 无法通过 API 指定状态。
- TrajectoryStep 完整保存 state/context/action/observation/artifacts/evidence/state_after/verifier feedback/reward/usage/parent/branch；空 reward 和未知计量不伪造。
- CounterfactualLink 预留六种 intervention，实际运行 snapshot branch + 新动作；不计算 credit assignment。

旧事件 envelope 保持 1.0，新的 first-class objects 为 2.0。两者分层，旧科学 Claim 的 SUPPORTED/VERIFIED 不会自动转换成新支持事实。

## 4. Migration 情况

新增 [0003_experiments.py](../migrations/versions/0003_experiments.py)：

| 表 | 结构 |
| --- | --- |
| experiment_objects | object_id PK；experiment_id FK；kind；created_event_id FK |
| provenance_edges | source_id + target_id + relation 复合 PK；created_event_id FK |

事件和 index/edge 在同一事务中提交；scope/cycle 验证失败会回滚。新表复用 append-only 触发器，拒绝 UPDATE/DELETE/TRUNCATE。实体版本仍来自 canonical events，不另建可变事实源。

已在现有 `0002` 数据库上执行 upgrade 到 `0003`；Compose 启动命令自动执行 migration。没有改写旧事件、旧 artifact 或旧 Claim。

## 5. Toy experiment 执行结果

实际执行：

```bash
source .venv/bin/activate
python examples/toy_experiment/run.py
```

该入口只需要已经启动的 PostgreSQL 与 Docker，无模型账号。

本次 stdout：

```text
Experiment created 93c5fde6-6070-483a-89c7-f8ab88600cdb
Hypothesis registered 4954f582-3877-4ff1-9742-86ba89f38c50
Run #1 executed
Artifacts stored
Evidence generated
Claim verified (SUPPORTED)
Trajectory recorded
Replay succeeded 903531be-0c21-4e1f-84da-dde46f61feb8
Branch created from step 1 ce1d72eb-196e-49a1-ac23-c64a735fe6f2
Run #2 executed
Provenance graph validated
Exports generated:
  trajectory.jsonl
  artifacts.jsonl
  evidence.jsonl
  sft.jsonl
  rl_transition.jsonl
  events.jsonl
  provenance.jsonl
Summary: /Users/bytedance/Desktop/Own/scientific-harness/outputs/experiments/93c5fde6-6070-483a-89c7-f8ab88600cdb/014f1174-7829-429a-927e-133c25e947e1/summary.json
```

这是确定性的 addition-count 实验，不把机器上的噪声计时包装成确定性结论。

| 配置 | A additions | B additions | A/B 相同结果 |
| --- | ---: | ---: | ---: |
| N=2000 | 2000 | 1 | 2,001,000 |
| N=4000 | 4000 | 1 | 8,002,000 |

原始 run 的 MetricVerifier 与 TestVerifier 都通过，Claim 为 SUPPORTED；branch 的 metric evidence 同样支持新 Claim。TestVerifier 消费真实 `python -m unittest` 执行产生的 runtime log，不接受 Agent 自写的“测试成功”日志。

当前产物：[summary.json](../outputs/experiments/93c5fde6-6070-483a-89c7-f8ab88600cdb/014f1174-7829-429a-927e-133c25e947e1/summary.json)。本次导出 3 条 trajectories、25 个 artifact 版本、3 条 evidence，canonical export 截止包含 124 个 events。

## 6. Pytest 结果

```text
HARNESS_INTEGRATION=1 .venv/bin/pytest -q
50 passed, 1 warning in 38.53s
```

没有 skip。唯一 warning 是 Starlette TestClient 对 httpx 的上游弃用提示。`ruff check`、`ruff format --check` 均通过。

覆盖：原 24 项 V0 回归；新 schema serialization；无证据 Claim 拒绝；跨实验 evidence 拒绝；不可变 artifact 版本和 parent hash；provenance cycle 的事务回滚；claim 追溯到 code/config/run/observation；真实 deterministic replay；随机输出 mismatch；branch 后原 trajectory 不变；snapshot 失败仍保留 trajectory、恢复后可继续；新 API 全链路；标准 MCP/RAG Action/Observation/Artifact；替换 environment 不需要 Docker；内部控制 Python 不受工作区同名 json.py 干扰。

## 7. Replay / branch replay 结果

原 run：`a63d2aaf-9269-4581-929f-893694122ae8`。
Exact replay 新 run：`903531be-0c21-4e1f-84da-dde46f61feb8`，matched=`true`。

| 原始 step | 比较规则 | 匹配 |
| --- | --- | --- |
| 1 | `exit_code_artifacts_stdout_stderr` | true |
| 2 | `declared_test_exit_code_and_artifacts` | true |

普通动作比较 exit code、采集文件 SHA-256 和 stdout/stderr。声明的 test 命令比较 exit code/artifact，test runner 的计时文本单独报告，不作为科学判据。

Branch run：`ce1d72eb-196e-49a1-ac23-c64a735fe6f2`。从 step 1 的后置 snapshot 新建容器，继承源码并以新 config artifact 把 N 改成 4000。原 trajectory 序列化在 branch 前后完全一致。

本次 Claim 的 provenance 查询包含 22 个相关节点，覆盖 Hypothesis、Experiment、Run、Action、Observation、code/config/metric/log Artifact、Verification 和 Evidence。查询接口与详情见 [provenance.md](provenance.md)。

Replay 记录目标 code_version，并在 run.started 额外记录当前 executor 源码哈希及实际依赖版本，区分实验复用的版本和本次执行引擎版本。完整语义见 [replay.md](replay.md)。

## 8. Export 样例

目录：`outputs/experiments/93c5fde6-6070-483a-89c7-f8ab88600cdb/014f1174-7829-429a-927e-133c25e947e1`。已逐一验证 manifest 记录的文件 SHA-256 和全部 CAS closure 内容哈希。

| 文件 | JSONL 行数 | 字节 |
| --- | ---: | ---: |
| [trajectory.jsonl](../outputs/experiments/93c5fde6-6070-483a-89c7-f8ab88600cdb/014f1174-7829-429a-927e-133c25e947e1/trajectory.jsonl) | 3 | 34,560 |
| [artifacts.jsonl](../outputs/experiments/93c5fde6-6070-483a-89c7-f8ab88600cdb/014f1174-7829-429a-927e-133c25e947e1/artifacts.jsonl) | 25 | 13,923 |
| [evidence.jsonl](../outputs/experiments/93c5fde6-6070-483a-89c7-f8ab88600cdb/014f1174-7829-429a-927e-133c25e947e1/evidence.jsonl) | 3 | 1,923 |
| [sft.jsonl](../outputs/experiments/93c5fde6-6070-483a-89c7-f8ab88600cdb/014f1174-7829-429a-927e-133c25e947e1/sft.jsonl) | 5 | 11,073 |
| [rl_transition.jsonl](../outputs/experiments/93c5fde6-6070-483a-89c7-f8ab88600cdb/014f1174-7829-429a-927e-133c25e947e1/rl_transition.jsonl) | 5 | 31,323 |
| [events.jsonl](../outputs/experiments/93c5fde6-6070-483a-89c7-f8ab88600cdb/014f1174-7829-429a-927e-133c25e947e1/events.jsonl) | 124 | 256,449 |
| [provenance.jsonl](../outputs/experiments/93c5fde6-6070-483a-89c7-f8ab88600cdb/014f1174-7829-429a-927e-133c25e947e1/provenance.jsonl) | 1 | 110,374 |

SFT 第一行的字段节选（实际文件保留完整 state、代码和参数）：

```json
{
  "schema_version": "2.0",
  "input": {
    "state": {
      "config": {
        "N": 2000
      },
      "snapshot_id": "961ae978-dbba-410f-812e-cb859555c7d5"
    },
    "context_refs": []
  },
  "output": {
    "action": {
      "tool": "environment",
      "operation": "exec",
      "arguments": {
        "command": {
          "argv": [
            "python",
            "benchmark.py"
          ]
        }
      }
    }
  }
}
```

Branch RL transition 节选：

```json
{
  "schema_version": "2.0",
  "run_id": "ce1d72eb-196e-49a1-ac23-c64a735fe6f2",
  "step_id": "04ae6f54-a380-4452-9060-c0adad573fb1",
  "reward_signals": {},
  "terminal": true,
  "state": {
    "config": {
      "N": 4000
    }
  },
  "action": {
    "tool": "environment",
    "operation": "exec"
  },
  "observation": {
    "exit_code": 0,
    "created_artifacts": [
      "1757d4b6-9830-45f6-9fd3-ef59997dff8b",
      "ec724550-2dea-41cd-8090-4642b3967b76"
    ]
  },
  "next_state": {
    "snapshot_id": "ac7c8e04-7b82-4352-903c-7616522489ac"
  }
}
```

真实 evidence 示例：

```json
{
  "schema_version": "2.0",
  "evidence_id": "6ea30df0-c837-4c05-b872-f0afd23e4956",
  "experiment_id": "93c5fde6-6070-483a-89c7-f8ab88600cdb",
  "hypothesis_id": "4954f582-3877-4ff1-9742-86ba89f38c50",
  "source_artifact_ids": [
    "a0733600-ae65-4eb4-9e10-ca6df771c05c"
  ],
  "source_run_ids": [
    "a63d2aaf-9269-4581-929f-893694122ae8"
  ],
  "source_observation_ids": [
    "2cde73b3-9b14-4ec1-94af-cff0971326f0"
  ],
  "metric": "operations_b",
  "value": 1,
  "uncertainty": null,
  "verifier": "metric",
  "verification_id": "8221196b-b447-42ce-8687-5919bf9fa6a3",
  "verification_status": "PASSED",
  "metadata": {
    "criteria": {
      "field": "operations_b",
      "operator": "lt",
      "baseline_field": "operations_a"
    }
  }
}
```

SFT 数据从 state/context→action 构造，没有隐藏思维链。RL transition 的 reward_signals 为 {}；verifier feedback 另外记录，不伪造 reward。

## 9. 目前仍未实现的部分

- LLM/Human/Statistical verifier 只有 protocol；没有真实模型 backend 或服务集成。外部 client 可通过 SDK/HTTP 接入。
- CounterfactualLink 有 contract 和 state branch；没有自动 context_remove/mask、完整因果 credit assignment、reward 学习。
- 没有分布式 execution lease、崩溃恢复 worker、跨工具/DB exactly-once 事务；中断留下可见 intent，需要显式恢复。
- Export bundle 已带 CAS 和完整事件，但没有把 bundle 导入新数据库并跨机器重跑的工具。
- 当前工作动作词汇为 CLI-style exec 加 MCP/RAG；只有 Docker production environment。未来 Browser/Slurm/Robot 需要 adapter 实现，不在本阶段实现。
- Snapshot 为工作区文件系统，不含进程内存、外部数据库或物理设备状态。特殊文件/符号链接不可快照，失败会保留轨迹并禁止从该步 branch。
- 验证器核验声明的指标/测试规则，不自动证明任意自然语言 Claim 的语义蕴含；领域判据质量仍需 domain policy。
- Budget/constraints 是元数据，除现有 Docker 限制外不做完整预算管理；token/compute/cost 未测量字段为 null。
- API 单 worker 串行处理；直接 SDK 的生命周期调用要求调用者串行化。事件投影和 graph traversal 以正确性为主，尚未针对大型数据优化。

按要求没有实施 RL/SFT 训练、GRPO/ORSD、大 benchmark、multi-agent、planner、UI、Kubernetes、Slurm、physical lab 或新向量数据库。

## 10. 下一阶段最值得做的 3 个功能

1. **可靠执行恢复**：持久 execution lease、幂等键、未完成 intent reconciliation，以及明确的失败/重试协议。
2. **可移植 replay bundle**：验证、导入 artifact/event closure，固定环境/依赖，跨主机重跑并产生可解释差异报告。
3. **更强证据契约**：把 Claim 的结构化判据、Hypothesis 期望和 domain verifier policy 绑定，增加多次运行的统计证据；再据此做 context counterfactual 数据质量评估。
