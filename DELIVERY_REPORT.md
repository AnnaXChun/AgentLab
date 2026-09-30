# scientific-harness V0 交付报告

验证日期：2026-09-30（Asia/Shanghai）。项目位于本机 `scientific-harness/`，已初始化 Git，未提交或发布远程仓库。许可证为 Apache-2.0。

V0 的完整本地链路已实测通过：Compose 启动及自动 migration、真实 PostgreSQL/pgvector、真实 MCP stdio、真实 Docker 命令、证据绑定 Claim、三个分支的 replay，以及五种数据导出。测试结果为 **24 passed**，没有 skip。

## 1. 当前 repo tree

省略虚拟环境、缓存、Git 内部文件和生成数据；`.artifacts/` 与 `outputs/` 是保留在本机的运行产物。

```text
scientific-harness/
├── .github/
│   └── workflows/
│       └── tests.yml
├── adapters/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── mock.py
│   ├── sandbox/
│   │   ├── __init__.py
│   │   └── docker.py
│   ├── storage/
│   │   ├── __init__.py
│   │   └── local.py
│   └── __init__.py
├── domain_packs/
│   ├── biology/
│   │   ├── __init__.py
│   │   ├── manifest.yaml
│   │   └── reference.md
│   ├── demo/
│   │   ├── __init__.py
│   │   ├── manifest.yaml
│   │   ├── reference.md
│   │   └── server.py
│   ├── materials/
│   │   ├── __init__.py
│   │   ├── manifest.yaml
│   │   └── reference.md
│   ├── __init__.py
│   └── loader.py
├── examples/
│   ├── canonical_event.json
│   ├── generate_schemas.py
│   ├── replay.py
│   └── run_demo.py
├── migrations/
│   ├── versions/
│   │   ├── 0001_events.py
│   │   └── 0002_retrieval.py
│   └── env.py
├── packages/
│   ├── core/
│   │   ├── artifacts/
│   │   │   ├── __init__.py
│   │   │   └── base.py
│   │   ├── claims/
│   │   │   └── __init__.py
│   │   ├── episodes/
│   │   │   └── __init__.py
│   │   ├── events/
│   │   │   ├── __init__.py
│   │   │   ├── recorder.py
│   │   │   └── store.py
│   │   ├── evidence/
│   │   │   └── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── database.py
│   │   │   ├── migrations.py
│   │   │   └── schemas.py
│   │   ├── verification/
│   │   │   ├── __init__.py
│   │   │   └── service.py
│   │   └── __init__.py
│   ├── exporters/
│   │   ├── __init__.py
│   │   └── trajectory.py
│   ├── mcp_gateway/
│   │   ├── __init__.py
│   │   └── gateway.py
│   ├── provenance/
│   │   ├── __init__.py
│   │   ├── replay.py
│   │   └── tracing.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── sandbox/
│   │   ├── __init__.py
│   │   └── base.py
│   ├── sdk/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── harness.py
│   │   └── runtime.py
│   └── __init__.py
├── schemas/
│   ├── AgentAction.json
│   ├── Artifact.json
│   ├── Claim.json
│   ├── DomainPack.json
│   ├── Episode.json
│   ├── Event.json
│   ├── EventEdge.json
│   ├── Evidence.json
│   ├── Verification.json
│   └── openapi.json
├── tests/
│   ├── contract/
│   │   ├── test_domains_mcp.py
│   │   ├── test_export.py
│   │   ├── test_replay_branches.py
│   │   └── test_safety.py
│   ├── integration/
│   │   ├── conftest.py
│   │   └── test_live.py
│   ├── unit/
│   │   ├── test_retrieval.py
│   │   ├── test_sandbox.py
│   │   ├── test_schemas.py
│   │   ├── test_storage.py
│   │   ├── test_tracing.py
│   │   └── test_verification.py
│   └── conftest.py
├── .dockerignore
├── .gitignore
├── ARCHITECTURE.md
├── DELIVERY_REPORT.md
├── Dockerfile
├── GOALS.md
├── LICENSE
├── Makefile
├── README.md
├── alembic.ini
├── docker-compose.yml
├── pyproject.toml
└── uv.lock
```

## 2. 已实现 architecture

- `AgentAdapter` 返回声明式 `AgentAction`；MockAgent 不执行 I/O，不依赖模型厂商或 Agent framework。
- `Harness` 统一调度 ingestion、retrieval、MCP、sandbox、artifact、claim、verification、checkpoint、fork、rollback、export。
- 外部实验操作先提交 intent event，再执行，再记录 result/failure；16 KiB 以上的 payload 使用 CAS 引用。
- `DomainPack` 注入镜像、MCP server/schema、检索配置和 verifier 列表。demo/materials/biology 使用同一个 core；后两者为 mock pack。
- PostgreSQL append-only events 为 canonical truth。OTel/OpenInference 仅生成观测镜像。
- Artifact 使用原子 no-clobber 发布、SHA-256 路径及读取校验。代码、输入、结果和快照都可引用。
- Claim 经过 `UNVERIFIED → SUPPORTED → VERIFIED`；验证失败进入 `CONTRADICTED`。成功状态只能由 verifier service 写入。
- Replay 根据事件历史和 artifact 重建分支状态，不调用外部工具；fork/rollback 追加历史，不改旧分支。
- Public schema 为 `1.0`；提供 JSON Schema/OpenAPI、显式 event migration registry 和 Alembic SQL migrations。

完整设计见 [ARCHITECTURE.md](ARCHITECTURE.md)。`core/episodes`、`core/evidence`、`core/claims` 保留模块边界，V0 实体定义集中在 `core/models/schemas.py`。

## 3. Database schema

| 表 | 关键字段与约束 |
| --- | --- |
| `episodes` | `episode_id` PK、`payload` JSONB；用行锁串行化同 episode 的 event append |
| `events` | `event_id` PK、`episode_id` FK、`sequence`、`branch_id`、`event_type`、`timestamp`、`payload` JSONB；`(episode_id, sequence)` UNIQUE |
| `event_edges` | `(source_event_id, target_event_id, relation)` 复合 PK；两端 FK 指向 events；`schema_version` |
| `retrieval_chunks` | `(namespace, evidence_id)` PK；document/hash/source/title/chunk/content；`embedding vector(64)`；生成列 `search tsvector`；GIN 与 HNSW index |
| `alembic_version` | 当前版本 `0002` |

Migration `0001` 创建 canonical store、pgvector extension、不可 UPDATE/DELETE/TRUNCATE 的数据库触发器；`0002` 创建检索索引。Artifact/Evidence/Claim/Verification 以版本化记录嵌入 canonical event，replay 生成投影，避免第二个可变事实来源。

## 4. Canonical event schema 示例

下面是本次真实 demo 的 sandbox result，完整示例另存于 [examples/canonical_event.json](examples/canonical_event.json)。`state_*_hash` 表示分支事件前缀的历史状态哈希；`parent_event_id` 关联对应的执行请求。

```json
{
  "schema_version": "1.0",
  "episode_id": "4f148354-4db9-4340-8a2e-49d7f999f856",
  "event_id": "df693287-f29b-432d-8928-1474719d0e44",
  "sequence": 25,
  "timestamp": "2026-09-30T06:43:31.169470Z",
  "trace_id": "e05860a6f5060c9316001edcdab769eb",
  "span_id": "6733ccdaab707408",
  "branch_id": "74a84daa-087e-4401-ac68-410f8f3740b4",
  "parent_event_id": "d49ec2b6-e30f-4904-96b7-8ef0cf812889",
  "actor_type": "harness",
  "actor_id": "scientific-harness",
  "event_type": "sandbox.result",
  "input": {
    "request_event_id": "d49ec2b6-e30f-4904-96b7-8ef0cf812889"
  },
  "output": {
    "files": {
      "result.json": "sha256:6045b051265b7063147064437479e9259d624f474a3c0eb7a82c6560f3971195"
    },
    "stderr": "",
    "stdout": "",
    "exit_code": 0,
    "timed_out": false,
    "environment": {
      "image": "python:3.12-slim",
      "runtime": "docker",
      "image_id": "sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f",
      "platform": "arm64"
    },
    "schema_version": "1.0",
    "input_artifacts": {
      "experiment.py": "sha256:64271fdf197e07f235641c974e8cd8c3e5d2f93afefe1dc8378b42da6dd5ef99"
    },
    "stderr_truncated": false,
    "stdout_truncated": false
  },
  "artifact_refs": [
    "sha256:6045b051265b7063147064437479e9259d624f474a3c0eb7a82c6560f3971195",
    "sha256:64271fdf197e07f235641c974e8cd8c3e5d2f93afefe1dc8378b42da6dd5ef99"
  ],
  "evidence_refs": [],
  "state_before_hash": "f578cfe64a6e8a2f7e3668f087a53b589cf267bfb9b8142627eeee961820eba0",
  "state_after_hash": "a3033e2820d405316bda142a4709dcdde179bd39f00fcbd92ec1c78be5927cec",
  "status": "SUCCESS",
  "usage": {
    "schema_version": "1.0",
    "wall_time_ms": 372.5622709998788,
    "cpu_seconds": null,
    "gpu_seconds": null,
    "tokens": null,
    "estimated_cost_usd": null
  },
  "metadata": {}
}
```

## 5. DomainPack manifest 示例

真实可运行的 [demo manifest](domain_packs/demo/manifest.yaml)：

```yaml
schema_version: '1.0'
api_version: sciharness/v1
kind: DomainPack
metadata:
  name: demo
  version: 0.1.0
environment:
  image: python:3.12-slim
mcp_servers:
  - name: demo-tools
    transport: stdio
    command: $PYTHON
    args: ['-m', 'domain_packs.demo.server']
    tool_schemas:
      load_dataset:
        type: object
        properties:
          candidate: {type: string}
        required: [candidate]
        additionalProperties: false
      run_mock_simulation:
        type: object
        properties:
          candidate: {type: string}
          seed: {type: integer}
        required: [candidate]
        additionalProperties: false
retrieval:
  sources: [local_docs]
  documents: [reference.md]
validators: [numeric, execution, schema]
resources:
  cpu: 1
  memory_gb: 1
network:
  policy: disabled
```

`$PYTHON` 在 Gateway 打开 stdio 会话时解析，manifest 可以跨主机/容器保存及加载。materials/biology 使用相同的配置结构，不需要修改 core。

## 6. Demo 执行命令

```bash
cd scientific-harness
make demo
```

已实测等价命令：

```bash
docker compose up -d --build --wait
docker compose exec -T scientific-harness-api python examples/run_demo.py
```

本机开发模式：

```bash
uv sync --python 3.12
uv run python examples/run_demo.py
HARNESS_INTEGRATION=1 uv run pytest -q
```

API：`http://127.0.0.1:8000`，Postgres：`127.0.0.1:55432`。当前两个 Compose service 均 healthy。安装并使用了独立的 Colima `scientific-harness` profile；Python 3.12 和依赖由 uv 管理。

## 7. Demo 完整 trajectory 摘要

实际 episode：`4f148354-4db9-4340-8a2e-49d7f999f856`。状态：`COMPLETED`。

1. 创建 goal，记录 DomainPack、Python/依赖版本、seed、源码内容哈希。
2. 将 demo Markdown 文档存入 CAS 并 ingestion 到 Postgres FTS + pgvector。
3. MockAgent 发出可见说明和 retrieval action，检索返回稳定 Evidence ID。
4. Gateway 分别调用真实 MCP `load_dataset` 和 `run_mock_simulation`。
5. MockAgent 生成 `experiment.py`；Harness 将代码存入 CAS、上传 Docker 后执行，收集 `result.json`。
6. 创建 Claim：先 UNVERIFIED，解析 Evidence/Artifact 后 SUPPORTED。
7. NumericVerifier 从不可变 `result.json` 读取 mean，实测 `0.2 < 0.5`，记录成功 Verification 并升级为 VERIFIED。
8. 创建主分支 checkpoint，保存 Docker 工作区 tar artifact。
9. fork branch A，恢复 checkpoint，执行故意抛出异常的实验，记录失败和 traceback。
10. fork branch B，恢复同一 checkpoint，读取已有 `result.json` 并生成 `recovered.json`；通过 retry edge 关联 branch A 的失败，记录成功。
11. 保存 branch B checkpoint；主分支 rollback 到原 checkpoint，物理恢复 sandbox 并追加 rollback event。
12. 清理三个分支的容器；追加 `episode.completed`。
13. Replay 重建三个分支；导出 canonical history、Parquet、SFT、RL-style 和 failure-recovery。

本次共有 **70 个 live events**。导出截止 `export.started`，保存 **69 个 events**；最后的 `export.completed` 在文件写完后追加到 live log。

| Event type | 数量 |
| --- | ---: |
| `episode.started` | 1 |
| `retrieval.ingest` | 1 |
| `artifact.created` | 12 |
| `retrieval.ingested` | 1 |
| `agent.action` | 8 |
| `agent.message` | 1 |
| `retrieval.query` | 1 |
| `retrieval.result` | 1 |
| `tool.call` | 2 |
| `tool.result` | 2 |
| `sandbox.command` | 14 |
| `sandbox.result` | 14 |
| `claim.created` | 1 |
| `claim.updated` | 1 |
| `verification.started` | 1 |
| `verification.result` | 1 |
| `checkpoint.created` | 2 |
| `branch.created` | 2 |
| `rollback.created` | 1 |
| `episode.completed` | 1 |
| `export.started` | 1 |
| `export.completed` | 1 |

其中 `artifact.created` 含 intent 和成功记录，因此不是 artifact 数量。成功 artifact 写入 6 次：文档、Python 源码、结果、主分支快照、恢复结果、分支快照。

本次分支 ID：

```json
{
  "main": "74a84daa-087e-4401-ac68-410f8f3740b4",
  "branch_a": "a0216f7b-e19c-4174-8b54-cf7d8f29654c",
  "branch_b": "deea5deb-00f1-4a60-a0c4-130791484195"
}
```

## 8. Export 文件示例

本机目录：`outputs/4f148354-4db9-4340-8a2e-49d7f999f856/e4920f52-8f4a-4656-a8fa-8e0bbba8d3af`。

| 文件 | 字节 | JSONL 行数 |
| --- | ---: | ---: |
| [raw.jsonl](outputs/4f148354-4db9-4340-8a2e-49d7f999f856/e4920f52-8f4a-4656-a8fa-8e0bbba8d3af/raw.jsonl) | 81,367 | 69 |
| [events.parquet](outputs/4f148354-4db9-4340-8a2e-49d7f999f856/e4920f52-8f4a-4656-a8fa-8e0bbba8d3af/events.parquet) | 37,168 | — |
| [sft.jsonl](outputs/4f148354-4db9-4340-8a2e-49d7f999f856/e4920f52-8f4a-4656-a8fa-8e0bbba8d3af/sft.jsonl) | 8,528 | 3 |
| [rl.jsonl](outputs/4f148354-4db9-4340-8a2e-49d7f999f856/e4920f52-8f4a-4656-a8fa-8e0bbba8d3af/rl.jsonl) | 8,229 | 7 |
| [failure_recovery.jsonl](outputs/4f148354-4db9-4340-8a2e-49d7f999f856/e4920f52-8f4a-4656-a8fa-8e0bbba8d3af/failure_recovery.jsonl) | 1,976 | 1 |
| [edges.jsonl](outputs/4f148354-4db9-4340-8a2e-49d7f999f856/e4920f52-8f4a-4656-a8fa-8e0bbba8d3af/edges.jsonl) | 11,828 | 72 |
| [manifest.json](outputs/4f148354-4db9-4340-8a2e-49d7f999f856/e4920f52-8f4a-4656-a8fa-8e0bbba8d3af/manifest.json) | 693 | — |
| [demo_summary.json](outputs/4f148354-4db9-4340-8a2e-49d7f999f856/e4920f52-8f4a-4656-a8fa-8e0bbba8d3af/demo_summary.json) | 3,087 | — |

`raw.jsonl` 是完整 canonical events；Parquet 为 normalized event table；SFT 只含 visible message/tool-call/tool-result/observation；RL-style 的 reward 为 null；failure-recovery 包含 1 个显式失败→修正→成功 pair。`artifacts/` 保存离线 replay 需要的 CAS 闭包。

已验证脱离 DB/Docker 的 replay：

```bash
uv run python examples/replay.py outputs/4f148354-4db9-4340-8a2e-49d7f999f856/e4920f52-8f4a-4656-a8fa-8e0bbba8d3af
# 69 events, 3 branches
```

## 9. Pytest 与分阶段验证

最终真实环境测试命令：`HARNESS_INTEGRATION=1 .venv/bin/pytest -q`。

```text
24 passed, 1 warning in 10.59s
```

唯一 warning 是 FastAPI/Starlette TestClient 对 httpx 的上游弃用提示，不影响测试结果。真实集成测试没有跳过。`ruff check` 和 `ruff format --check` 通过。

| Phase | 完成后累计测试数 |
| --- | ---: |
| 1 schema/skeleton | 2 |
| 2 event/artifact storage | 4 |
| 3 sandbox interface | 5 |
| 4 真实 MCP / domain isolation | 9 |
| 5 retrieval | 10 |
| 6 verifier / payload spill | 12 |
| 7 replay / branch / rollback | 13 |
| 8 五种 exporter / DuckDB | 14 |
| 9 tracing | 15 |
| 10 API / 真实 Postgres + Docker / 完整 demo / safety | 24 |

测试覆盖数据库拒绝历史修改、并发 append 顺序、CAS 防覆盖/篡改、MCP allowlist/schema、Docker 超时/环境变量/上传下载/快照恢复、三 pack 不改 core、事件完整性、跨分支 artifact 拒绝、Claim 不可伪造状态、数值/执行/schema verifier、导出与离线完整性检查。

## 10. 当前已知限制

- 单用户可信本地部署，API 只有一个 worker；SDK 调用者要串行化同一 episode 的操作。Docker socket 与 MCP host 插件需要信任，尚无多租户隔离。
- Python AgentAdapter 是可信插件接口，无法在同一 Python 进程中阻止恶意插件访问宿主机。生成的实验代码进入 Docker。
- intent 与外部系统之间没有分布式事务；崩溃可能留下未完成 intent 和容器。不会悄悄重复执行；V1 需要 leases/idempotency/recovery worker。
- Replay 是历史重建，未实现跨机器的自动重新执行。记录镜像摘要、代码、依赖和 seed，但不承诺跨硬件位级重现。
- 数值/schema/execution verifier 验证显式判据，不自动证明自然语言 claim 与证据的语义蕴含关系；领域 pack 仍需提供科学有效性规则。
- 64 维 feature-hash embedding 是确定性基线；Docling/PDF 解析未实现，只保留接口。
- Sandbox 镜像必须包含 Python 3.12；文件和快照上限 16 MiB；stdout/stderr 各 16 MiB，超出显式标记 truncation。更大型科学计算属于后续工作。
- SFT 导出为各分支可见执行片段；RL-style 无训练和 reward；未实现独立 preference exporter。
- Usage 可空字段和 budget event type 已预留，未实现 token/成本/GPU 计量与完整预算控制。
- OTel/OpenInference SDK spans 已验证；未部署 collector、Phoenix 或产品 UI。

## 11. V1 推荐优先级

1. 持久化 execution lease、幂等键、崩溃恢复和未完成 intent 检查。
2. Claim criterion 的结构化绑定、领域 verifier policy 与证据相关性验证。
3. 固定镜像/依赖、显式 re-execution 模式及可复现性差异报告。
4. 外部 AgentAdapter、S3/远程 sandbox 的 adapter conformance suite。
5. 真实 embedding、Docling ingestion、检索质量评估与投影性能优化。
6. Preference/failure-recovery 数据质量、reward provenance 和训练消费接口；训练系统继续放在 core 之外。
