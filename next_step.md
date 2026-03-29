# imaging-101 Progress Summary & Next Steps

**Last Updated**: 2026-03-28

---

## 1. 项目现状 (Current Status)

### 1.1 仓库结构

经过整理，仓库目前结构清晰：

```
imaging-101/
├── CLAUDE.md                  # Agent 指引文件
├── README.md                  # 项目概述
├── config_llm.yaml            # LLM API 配置
├── next_step.md               # 本文件
│
├── evaluation_harness/        # 🧠 核心评估框架
│   ├── agent.py               #   ReAct agent（单循环）
│   ├── multi_agent.py         #   Multi-Agent pipeline（Plan→Architect→Code→Judge）
│   ├── agents/                #   各角色 agent 实现
│   ├── scorer.py              #   评估打分器（NRMSE, NCC, PSNR, SSIM, MSE + 可视化）
│   ├── visualizer.py          #   📊 可视化模块（对比图、残差图、指标卡、剖面图）
│   ├── runner.py              #   运行调度器（react / multi_agent）
│   ├── llm_client.py          #   LLM API 客户端（支持重试、计数）
│   └── ...
│
├── tasks/                     # 📋 11 个成像任务
│   ├── eht_black_hole/        #   ✅ 参考任务（已完成清洗）
│   ├── eht_black_hole_UQ/     #   ✅ 参考任务（已完成清洗）
│   ├── eht_black_hole_original/  # ✅ 已评估（端到端）
│   ├── eht_black_hole_dynamic/
│   ├── eht_black_hole_feature_extraction_dynamic/
│   ├── eht_black_hole_tomography/
│   ├── hessian_sim/
│   ├── light_field_microscope/
│   ├── reflection_ODT/
│   ├── single_molecule_light_field/
│   └── SSNP_ODT/
│
├── scripts/                   # 🔧 运行脚本
│   ├── run_end2end_gemini.sh
│   ├── run_function_evals.sh
│   ├── run_comparison.sh
│   ├── run_preprocessing_eval_gemini.sh
│   └── compare_frameworks.py
│
├── docs/                      # 📄 文档和报告
│   └── COMPARISON_REPORT_GEMINI25PRO.md  # 最新对比报告
│
├── results/                   # 📊 评估结果 JSON
├── logs/                      # 📝 日志
│   ├── eval_runs/             #   评估运行日志
│   └── interactions/          #   Agent 交互日志
│
├── skills/                    # 📚 共享技能库
├── overnight/                 # 🌙 过夜运行脚本
└── archive/                   # 🗄️ 归档（旧报告/脚本/日志）
```

### 1.2 评估框架能力

已实现两种 Agent 框架的端到端评估：

| 框架 | 架构 | 描述 |
|------|------|------|
| **ReAct** | 单循环 | Action→Observation→Thought 循环，每轮1次 LLM 调用 |
| **Multi-Agent** | 多角色协作 | Planner→Architect→Coder→Judge 流水线，含 Critic 审核 |

**评估指标**（5 个图像质量度量 + 资源用量 + 可视化）：
- NRMSE (↓), NCC (↑), PSNR (↑), SSIM (↑), MSE (↓)
- LLM 调用次数、wall time、token 用量
- **可视化输出**（每次评估自动生成）：
  - `comparison.png` — GT vs 重建结果 vs 差异图（三栏对比）
  - `residual.png` — 残差图 + 残差分布直方图
  - `metrics_card.png` — 指标可视化卡（水平柱状图，颜色编码质量）
  - `cross_section.png` — 水平/垂直剖面对比（GT vs 重建，含差异区域填充）
  - 所有图片保存在 `results/figures/<task>_<framework>_<model>/` 目录下
  - `reconstruction.npy` 和 `ground_truth.npy` 同时保存，确保可复现

### 1.3 已完成的评估结果

在 `eht_black_hole_original` 任务上的端到端评估（v2，公平对比）：

| 指标 | Claude ReAct | Gemini ReAct | Gemini Multi-Agent |
|------|-------------|-------------|-------------------|
| **模型** | Claude-4.6-opus | gemini-2.5-pro | gemini-2.5-pro |
| **NRMSE** (↓) | 1.6258 | 1.0546 | **0.7383** ★ |
| **NCC** (↑) | 0.0727 | 0.0701 | **0.7013** ★ |
| **PSNR** (↑) | N/A | 17.37 | **20.46** ★ |
| **SSIM** (↑) | N/A | -0.0034 | **0.5450** ★ |
| **LLM Calls** | ~40 | 73 | 99 |
| **Wall Time** | 3.3h | 12min | 88min |
| **Tokens** | 1.48M | 754K | 1.29M |

**核心发现**：
1. **Multi-Agent 在所有质量指标上全面胜出**：NRMSE 降低 30%，NCC 提升 10 倍，SSIM 从 0 提升到 0.55
2. Multi-Agent 重建结果捕捉了黑洞的基本结构（NCC=0.70），而 ReAct 的结果几乎没有结构信息
3. Multi-Agent 的优势来自：结构化问题分解、Critic 审核捕获错误、Judge 诊断失败原因、Re-planning 策略调整
4. 但 Multi-Agent 时间成本更高（88min vs 12min），主要由于 gemini-2.5-pro 的 thinking tokens 较多

---

## 2. 已知问题 & 技术债务

| 问题 | 严重程度 | 描述 |
|------|---------|------|
| Multi-Agent 进程被 kill | 🟡 中 | v2/v3 运行被 SIGKILL，原因不明（非 OOM），v4 在前台终端中运行成功 |
| Gemini thinking tokens 占比高 | 🟡 中 | gemini-2.5-pro 约 90% tokens 为 reasoning tokens，成本效率不理想 |
| ReAct parser 对 Gemini 格式的适配 | 🟢 已修 | 已添加 fallback regex 处理 Gemini 的 inline Action 输出 |
| `.gitignore` 存在合并冲突 | 🟢 已修 | 已清理并重写为干净版本 |
| Claude 的 ReAct 结果不公平 | 🟡 中 | Claude 运行时使用的是修复前的 parser，可能浪费了大量 FORMAT_ERROR 迭代 |
| 仅在 1 个任务上测试 | 🔴 高 | 目前只在 `eht_black_hole_original` 上运行了端到端评估 |
| 可视化模块已就绪 | 🟢 已完成 | `evaluation_harness/visualizer.py` 已集成到 scorer 中，每次端到端评估自动生成 4 种对比图并保存到 `results/figures/` |

---

## 3. Next Steps

### 3.1 🔥 高优先级

#### A. 扩展到更多任务

目前只评估了 1/11 个任务。下一步应在更多任务上运行对比：

```bash
# 推荐优先级（从已清洗的任务开始）
1. eht_black_hole_UQ        # 已完成清洗，不确定性量化
2. eht_black_hole           # 已完成清洗
3. hessian_sim              # Hessian 成像模拟
4. light_field_microscope   # 光场显微镜
5. reflection_ODT           # 反射光学衍射断层成像
```

每个任务运行 2 个配置（ReAct + Multi-Agent），收集统计显著的结果。

#### B. 重跑 Claude ReAct（修复后版本）

在 parser 修复后重新运行 Claude-4.6-opus 的 ReAct 评估，确保公平对比：

```bash
cd scripts/
# 修改 run_end2end_gemini.sh 中的 MODEL 为 cds/Claude-4.6-opus
# 运行评估
```

#### C. Claude Multi-Agent 评估

运行 Claude-4.6-opus 作为 Multi-Agent 的 backbone，形成 2×2 对比矩阵：

| | ReAct | Multi-Agent |
|---|---|---|
| **Claude-4.6-opus** | ✅（需重跑） | ⬜ |
| **gemini-2.5-pro** | ✅ | ✅ |

这样可以分离 **框架效果** vs **模型效果**。

### 3.2 🟡 中优先级

#### D. 优化 Multi-Agent 效率

- **缓存 system prompt**：重复的系统提示可以通过 prefix caching 优化

#### E. 添加更多评估指标

- **可视化对比** ✅ 已完成：每次端到端评估现在自动生成 4 种可视化图（对比图、残差图、指标卡、剖面图），保存在 `results/figures/` 中，并在评估报告 JSON 中记录图片路径
- **多次运行对比图**：使用 `generate_comparison_chart()` 可生成多框架/多模型的柱状对比图
- **代码质量度量**：pylint score、代码行数、函数覆盖率
- **鲁棒性测试**：多次运行取平均值和标准差（目前每个配置只有 1 次运行）
- **中间过程分析**：记录每个 macro-iteration 的质量变化曲线

### 3.3 🟢 低优先级

#### G. 新任务清洗

继续清洗未完成的任务（按 CLAUDE.md 中的标准流程）：
- `eht_black_hole_dynamic` — 动态黑洞成像
- `eht_black_hole_tomography` — 黑洞断层成像
- `single_molecule_light_field` — 单分子光场成像
- `SSNP_ODT` — 半稀疏相位 ODT
- `reflection_ODT` — 反射 ODT

#### H. Docker 沙箱

目前使用 `LocalRunner`（直接在本地运行），应实现 `DockerRunner` 以提供隔离的评估环境，避免 agent 代码影响宿主系统。

#### I. 自动化 CI

- 设置 overnight batch run，对所有任务 × 所有配置进行评估
- 自动生成对比报告
- 结果上传至 HuggingFace（`huggingface_upload.py` 已有但未接入流程）

---

## 4. 推荐的下一步行动

**建议立即执行的 3 件事**：

1. **在 `eht_black_hole_UQ` 上运行 ReAct + Multi-Agent 对比**（2-3 小时）
   - 验证 Multi-Agent 优势是否在不同任务上一致
   - 使用 gemini-2.5-pro 作为 backbone
   - **可视化模块已就绪**，运行后自动生成对比图

2. **重跑 Claude ReAct 评估**（0.5-1 小时）
   - 使用修复后的 parser
   - 100 iterations，与 Gemini 对比
   - 评估完成后自动保存可视化图片

3. **写论文/报告的 benchmark 对比表**
   - 整合所有评估结果
   - 使用 `generate_comparison_chart()` 生成多框架/多模型的柱状对比图
   - 利用各次评估自动生成的 `comparison.png`、`residual.png`、`cross_section.png` 等图片

### 4.1 可视化模块使用说明

评估框架现已内置自动可视化功能（`evaluation_harness/visualizer.py`）：

- **自动触发**：每次端到端评估完成后，`scorer.py` 自动调用 `visualizer.py` 生成图片
- **输出目录**：`results/figures/<task_name>_<framework>_<model>/`
- **生成的文件**：
  - `reconstruction.npy` — agent 重建结果（从沙箱复制，确保可复现）
  - `ground_truth.npy` — 参考真值（从任务目录复制）
  - `<label>_comparison.png` — GT vs 重建 vs 差异图（三栏对比）
  - `<label>_residual.png` — 残差图 + 残差分布直方图
  - `<label>_metrics.png` — 指标可视化卡
  - `<label>_cross_section.png` — 水平/垂直剖面对比
- **评估报告 JSON** 中新增 `visualization_paths` 字段，记录所有生成图片的绝对路径
- **CLI 输出** 会显示生成的图片数量和路径

---

*详细对比报告见 `docs/COMPARISON_REPORT_GEMINI25PRO.md`*
*评估结果 JSON 文件在 `results/` 目录*
*评估框架代码在 `evaluation_harness/`*
