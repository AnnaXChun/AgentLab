# GPU 接续交接（2026-10-10）

当前版本为 **0.4.0.dev0，veRL 在线桥接开发预览**。实现代码基线为
`11578ff4b8bfee3f3b4bd820faadc549fb113129`，已上传 `origin/main`。
本文件随后的文档提交记录交接状态。新机器应拉取最新 main。

用户已说明下一阶段有 GPU，希望新 Codex 对话接续开发。GPU 型号、数量、显存和
驱动版本尚未提供，需要在新环境实测。此前验证在 Mac 上完成，**尚未执行真实
GPU 训练，也没有已训练 checkpoint 或训练前后提升结果**。

## 新对话先读什么

1. [AGENTS.md](AGENTS.md)：仓库规则；每个完成版本检查后 commit 并 push origin/main，禁止 force push。
2. 本文件及 [veRL 实现与验收报告](docs/verl_integration.md)。
3. [训练配置](configs/verl/_grpo_base.yaml)、[启动器](src/experiment_infra/integrations/verl/launch.py)、
   [训练审计](src/experiment_infra/integrations/verl/audit_training.py)。
4. [benchmark 环境](docs/benchmarks.md) 与 [历史 Qwen 基线](docs/qwen_full_benchmark.md)。

项目是 Agent Experiment Infrastructure，保留 Experiment / Run / Artifact /
Evidence / Claim / Trajectory 架构。当前工作范围是完成已请求的 V0.4 官方 veRL
在线 GRPO 接入与验收；不扩展通用 Agent、多 Agent、Branch RL、自定义 advantage、
reward model、自动恢复或 UI。训练循环由官方 veRL 拥有，core 不依赖 Torch/Ray/CUDA。

## 已完成的代码与验证

- PolicyTrace / PolicyStepSpan / PolicyIdentity / RewardResult，以及 OFFLINE、ONLINE_RL、REPLAY、BRANCH 来源隔离。
- 官方 AgentLoop 桥接、原始推理 token/logprob、Continuous Token 合并、精确 mask/span、严格 JSON action。
- 确定性 verifier → Evidence → RewardResult；导出哈希和 provenance 审计。
- toy 与 tau2 新任务输入数据桥接；原生 GRPO 配置、启动器、实际 safetensor 参数变更审计。
- 全套本地测试：**102 passed, 4 skipped**（3 个隔离 MLX 测试、1 个 GPU 测试跳过）。
- 最终桥接测试：**20 passed, 1 GPU test skipped**；Ruff、格式、lock 和 whitespace 检查通过。
- 两个 Hydra 配置与固定版本官方配置树组合成功；core 独立安装可用，不安装 veRL/Ray/Torch。
- 上游 AgentLoopOutput 仅做过隔离 Pydantic schema 检查，尚非完整 GPU 进程验证。
- 真实 Docker smoke 读输入、写答案、验证得分 1；token 是脚本测试 ID，来源 OFFLINE，**没有优化或训练**。
- tau2 数据桥接已在本机生成互不重叠的 8 个 train / 5 个 test 输入，不含隐藏判定条件。

历史离线 Qwen3-1.7B 基线：tau2 telecom-solo **base 114 题，13 成功（11.40%）**，
13 条成功轨迹均通过官方 replay 和完整性检查。这不是完整 2285 题，也不是 SAB 或 RL
结果；此前有调试接触，不能称为独立 held-out 结果。历史离线/回放轨迹禁止进入 on-policy GRPO。

## GPU 环境接续步骤

以下是待在 GPU 主机验证的操作，不能视作已经跑通的安装组合。

```bash
git clone https://github.com/AnnaXChun/AgentLab.git
cd AgentLab
git switch main
git pull --ff-only origin main
nvidia-smi
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[verl,benchmarks]'
```

1. 记录 Linux、GPU/显存、驱动/CUDA、Python、Torch、vLLM、Ray、veRL 版本。
   按固定上游版本安装兼容的 Linux/NVIDIA vLLM/FSDP 后端；extra 只声明适配层依赖，
   并不保证安装出完整兼容的 GPU 栈。先验证 CUDA 和官方 AgentLoop 导入。
2. 启动 Docker/PostgreSQL：`docker compose up -d --build --wait`；按 README 验证
   runtime、迁移与 Docker 访问。运行本地桥接测试和 scripted smoke，结果仍标记 OFFLINE。
3. 下载官方 Qwen3-1.7B HF safetensors 模型到 `/models/Qwen3-1.7B`（可改为真实路径）。
   不使用 Mac MLX 量化文件代替训练权重。准备 tau2 时按下列固定 SHA 下载上游，
   默认位置 `.benchmarks/tau2-bench`，也可显式设置 `TAU2_ROOT`；详见 benchmark 文档。
4. 先运行真实 toy GRPO，解决安装/API/显存问题，直到通过参数变更验收。
5. toy 通过后运行 tau2 小子集在线 GRPO，再报告真实训练前后评估与完整训练样本。

固定依赖：

| 组件 | revision |
| --- | --- |
| veRL `0.10.0.dev` | `55a06a164d3d308b407fa9063bcfef25a41f0ea4` |
| tau2 | `5bfa7e37b36656b37dc6d022156be6563c1007f3` |
| Qwen/Qwen3-1.7B BF16 | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

启动器检查 veRL 的 `direct_url.json` 中 VCS commit。应通过固定 Git 依赖安装；
本地 editable 安装若没有该 metadata，会被拒绝。不要直接删掉版本校验来绕过问题。

```bash
python -m experiment_infra.integrations.verl.launch \
  --config configs/verl/toy_grpo.yaml \
  --model /models/Qwen3-1.7B \
  --output-dir results/verl-toy-grpo

# toy 验收通过后：
python -m experiment_infra.integrations.verl.launch \
  --config configs/verl/benchmark_grpo.yaml \
  --model /models/Qwen3-1.7B \
  --output-dir results/verl-tau2-grpo

# 此测试会另行启动真实训练，不只是检查导入：
VERL_INTEGRATION=1 VERL_TEST_MODEL=/models/Qwen3-1.7B \
  python -m pytest tests/verl/test_gpu.py -q
```

每次使用新的输出目录；目录存在即拒绝启动，resume 默认关闭。启动器自动生成新的
任务输入 Parquet，日志写入输出目录 `trainer.log`。默认 batch=4、rollout n=4、
3 个优化步、lr=1e-6、每步评估和保存 checkpoint；先训练前评估。单 AgentLoop worker，
thinking 关闭。tau2 上下文较长，按实测显存调整合法配置并记录差异；当前没有显存需求实测。

## 必须保留的接口约束

- 使用官方 `AgentLoopBase.run`、`server_manager.generate`、Continuous Token API 和
  `AgentLoopOutput.reward_score`；不以独立 OpenAI 推理服务替代训练 rollout。
- 生成 token 不重新 tokenize；observation mask=0。当前仅支持 append-only CT，前缀重写必须报错。
- action span 是严格 JSON 整个生成区间（含服务端结束 token），不是从 reasoning 猜测的子串。
- policy version 来自推理服务 `min_global_steps == max_global_steps`，缺失/混合版本中止。
  `training_step` 指生成时的权重版本，不是下一次优化步。
- 仅当前策略的新 ONLINE_RL 样本可进入训练；脚本、离线、回放、分支和旧策略样本必须拒绝。
- reward 来自真实 verifier/evidence；完整性、存储或生成前身份错误必须显式失败。
  当前不实现恢复；清理失败也可能向上抛出。

## GPU 验收与交付

- [ ] 真正安装并调用官方 AgentLoop，完成多步 toy GRPO。
- [ ] global step 推进；至少两个已保存优化步；初始/最终参数名称和 shape 一致，最终值有限。
- [ ] 实际参数值和哈希发生变化，且原始模型文件未被训练过程覆盖。
- [ ] `policy_update_evidence.json` 审计通过；不能以“生成了 checkpoint 文件”代替更新证明。
- [ ] tau2 小子集真实在线 GRPO，包含不同策略版本的 before/after 评估。
- [ ] 保存 reward 曲线、有效 rollout 数、成功率、失败分类、checkpoint 哈希及训练配置/版本。
- [ ] 展示一条真实训练样本：prompt/response token、mask、step spans、环境动作/观测、
  Evidence、RewardResult、生成权重版本和后续更新 checkpoint 的可核验引用。
- [ ] 更新 `docs/verl_integration.md` 与本文件，检查、提交、推送代码和可公开进度报告。

同组 reward 全相等可能没有 GRPO 更新。应报告和分析真实结果，不能篡改 reward、
放宽 verifier 或伪造样本来让审计通过。不预先承诺成功率提升。

## 原机器证据与迁移边界

代码和报告在 GitHub；以下生成物仅在原 Mac，**不在仓库，也不会随 clone 迁移**：

- `results/verl-local-smoke-final-20261010/local_smoke.json` 及旁边 canonical exports。
- `results/verl-tau2-inputs/`：本地数据桥接输出，可在新主机重新生成。
- `results/qwen-full-tau2-20261009/successful_trajectories.zip`：13 条历史成功轨迹。
- `/tmp/agentlab-v04-final-suite.log`、`/tmp/agentlab-v04-final-bridge.log`、
  `/tmp/agentlab-v04-config-final.log`：临时验证日志，可能被系统清理。

权重、benchmark bundles/任务正文/参考程序、凭证、数据库、CAS 和完整轨迹导出不提交
Git。需要历史原始证据时另行从原机器迁移，并遵守上游数据条款；GPU 接续可重新生成
环境与新 rollout，无须把旧训练数据导入。成功完整轨迹应保存在结果目录；失败保留汇总
统计，遵循现有 success-only 交付机制。训练验收所需的内部审计证据仍需保留。
