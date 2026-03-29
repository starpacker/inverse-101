# Multi-Agent Framework — 迭代改进 Prompt

> **目标**: 不断修改 multi-agent framework，提升其在 `eht_black_hole_original` 上的指标，逼近 ground-truth method 的效果  
> **红线**: 所有改动必须 general-purpose，禁止 task-specific tuning  
> **Git 备份**: `https://github.com/starpacker/inverse-101` (branch: `main`)  
> **最后更新**: 2026-03-29

---

## 0. 硬性约束（违反任何一条 = 作废）

1. **禁止 Cheating**: 不得在 prompt 中注入 task-specific 的算法选择、超参数、数据格式、数学公式。所有 prompt 规则必须对任意 computational imaging 任务通用。
2. **禁止信息泄漏**: 不得读取 `evaluation/reference_outputs/`、ground truth 或任何评估目标数据来指导 agent。
3. **Prompt 必须 General**: 例如你不能写 "use closure phases for gain-invariance"（天文术语），但可以写 "use pre-computed features from data files"（通用建议）。
4. **框架必须泛化**: 所有改动必须能在不同任务（天文、医学、工业成像等）上同样适用。
5. **审计要求**: 每次更改必须更新 `docs/PROGRESS_REPORT.md`，记录：改了什么、为什么、是否 task-specific。

---

## 1. 当前状态

### 1.1 项目结构

```
imaging-101/
├── evaluation_harness/          # 评估框架（你要改进的主体）
│   ├── multi_agent.py           # Multi-Agent 流水线编排器
│   ├── agent.py                 # ReAct Agent（对比基线）
│   ├── scorer.py                # 指标计算 (NRMSE, NCC, PSNR, SSIM)
│   ├── config.py                # 配置
│   ├── llm_client.py            # LLM API 客户端
│   ├── local_runner.py          # 本地沙箱
│   └── agents/                  # Multi-Agent 的子 Agent
│       ├── base.py              # BaseAgent（LLM 调用 + 续写）
│       ├── planner_agent.py     # Planner + Critic
│       ├── architect_agent.py   # Architect
│       ├── coder_agent.py       # Coder
│       └── judge_agent.py       # Judge（故障诊断 + 路由）
├── tasks/                       # 各成像任务
│   └── eht_black_hole_original/ # 当前主要评估任务
├── docs/
│   └── PROGRESS_REPORT.md       # 完整变更审计（必读！）
└── results/                     # 运行结果 JSON
```

### 1.2 Multi-Agent Pipeline 流程

```
Planner → Critic(审核循环) → Architect → Coder(逐文件) → Execution(python main.py) → Judge
   ↑                                                                                      |
   └──────────────────── 路由回相应阶段 ←────────────────────────────────────────────────────┘
```

**详细流程**:
1. **Phase 0: Data Exploration** — 自动探索 `data/` 目录，获取数据键名、形状、类型、元数据、requirements。构建 `data_inventory` 供所有 agent 使用。
2. **Planner** — 接收 task README + data_inventory + requirements + 历史失败信息，输出算法计划（数学公式、策略、步骤、超参数、文件结构）。
3. **Critic** — 审核计划，JSON PASS/REJECT 循环（最多 3 轮）。
4. **Architect** — 根据计划输出代码骨架（文件路径 + 函数签名 + docstring + pass）。
5. **Coder** — 逐文件实现（接收计划、骨架、全模块接口摘要、数据清单、feedback）。每个文件 AST 语法检查，失败重试 1 次。
6. **Execution** — 在沙箱中运行 `python main.py`。检查 `output/reconstruction.npy` 是否存在、格式是否有效（2D 数值数组，非零/NaN/常数）、优化器是否收敛。
7. **Judge** — 诊断失败原因（4 步协议：语法→接口→实现→算法），输出 JSON，路由回 Planner/Architect/Coder。

**安全机制**:
- Quick-fix：常见 Python 错误（ImportError, KeyError 等）的自动修补
- 卡死检测：同一 agent 连续分配 3 次 → 升级给 Planner
- 收敛检查：检测 scipy 优化器异常终止

### 1.3 当前最佳结果 (v7 Baseline)

| 指标 | Agent v7 | Reference (Closure-only cal) | Reference (Amp+CP cal) | Reference (Vis RML cal) |
|---|---|---|---|---|
| **NRMSE** ↓ | **0.7355** | 0.8226 | 0.7043 | **0.2648** |
| **NCC** ↑ | **0.6940** | 0.7523 | 0.7630 | **0.9669** |
| Iterations | 1 | — | — | — |
| LLM Calls | 11 | — | — | — |
| Time | 664s | — | — | — |

**目标**:
- **中期目标**: 超过 Amp+CP (cal)：**NRMSE ≤ 0.60, NCC ≥ 0.80**
- **终极目标**: 逼近 Vis RML (cal)：**NRMSE ≤ 0.30, NCC ≥ 0.95**

### 1.4 v7 的已知问题

1. **Runtime 错误未完全修复**: off-by-one 数组索引错误
2. **优化可能不充分**: 只跑了部分 optimization round 就停止
3. **缺乏自我验证**: Agent 不做 sanity check
4. **Regularizer 可能缺失或不当**
5. **1 次迭代就成功返回**: 没有迭代改进的机会

---

## 2. 工作流程（你必须遵循的循环）

### 核心循环: 改代码 → 跑评估 → 记录 → 重复

```
┌─────────────────────────────────────────────────────┐
│ 1. 选择一个改进方向（从 Section 3 或自己分析）           │
│ 2. 修改 evaluation_harness/ 中的代码                   │
│ 3. AST 语法检查所有改动文件                             │
│ 4. 运行评估（至少 1 次，建议 3 次取中位数）               │
│ 5. 记录结果到 PROGRESS_REPORT.md                      │
│ 6. 如果指标提升 → 保留改动，继续下一个方向               │
│    如果指标持平或变差 → 回滚改动，尝试其他方向            │
│ 7. git commit + push backup                          │
│ 8. 回到步骤 1                                         │
└─────────────────────────────────────────────────────┘
```

### 2.1 环境准备

```bash
cd /home/yjh/imaging-101

# API 配置
export BASE_URL="https://ai-gateway-internal.dp.tech/v1"
export API_KEY="sk-Zj3a7RQDVCXr-Axg-0gtkg"
```

### 2.2 运行一次评估

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --model "gemini-2.5-pro" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --max-iterations 10 \
    --timeout 7200 \
    --framework multi_agent \
    --output results \
    -v 2>&1 | tee logs/eval_runs/run_$(date +%Y%m%d_%H%M%S).log
```

### 2.3 查看结果

```bash
# 查看所有 end-to-end 结果
python3 -c "
import json, glob
for f in sorted(glob.glob('results/*end_to_end*.json')):
    d = json.load(open(f))
    qm = d.get('quality_metrics') or {}
    if not qm or qm.get('error'): continue
    print(f'{d.get(\"framework\",\"?\"):12s} | '
          f'NRMSE={qm.get(\"nrmse\",\"—\"):>7} | '
          f'NCC={qm.get(\"ncc\",\"—\"):>7} | '
          f'PSNR={qm.get(\"psnr\",\"—\"):>6} | '
          f'SSIM={qm.get(\"ssim\",\"—\"):>7} | '
          f'calls={d.get(\"llm_calls\",\"?\"):>3} | '
          f'{f.split(\"/\")[-1][:50]}')
"
```

### 2.4 AST 语法检查

```bash
for f in evaluation_harness/multi_agent.py evaluation_harness/agents/*.py; do
  python3 -c "import ast; ast.parse(open('$f').read()); print('OK: $f')"
done
```

### 2.5 Git 备份

```bash
git add -A && git commit -m "v<N>: <简要描述改动>" && git push backup main
```

### 2.6 评估结果解读

由于 LLM 随机性，单次运行不可靠。建议：
- **每次改动后至少跑 1 次**，记录指标
- **重要改动跑 3 次**，取中位数
- v7 baseline: NRMSE ≈ 0.74, NCC ≈ 0.69（但单次运行可能在 0.7~1.0 之间波动）

---

## 3. 改进方向（General-Purpose Only）

以下改进按优先级排序。每个方向都不依赖任何特定任务知识。

### 3.1 [高优先级] 增强 Coder 的数值严谨性

**问题**: v7 中 "index 421 is out of bounds" 是经典 off-by-one 错误。科学计算中数组索引/数值稳定性 bug 极为常见。

**方案**: 在 `coder_agent.py` 的 system prompt 中添加通用数值计算规则：

- Array Indexing Safety: 循环中确认 `i < array.shape[axis]`
- Gradient Verification: 用 `scipy.optimize.approx_fprime` 做有限差分验证
- Numerical Stability: 除法加 `epsilon=1e-10`，`np.log` 加 `np.maximum(x, epsilon)`
- Shape Assertion: 关键计算前后加 `assert result.shape == expected_shape`

**改动范围**: 仅 `coder_agent.py`，最小改动。

---

### 3.2 [高优先级] 自动代码验证（Validator 阶段）

**问题**: Coder 写完代码后直接运行 `main.py`，bug 只在 Execution 才发现，浪费整个 iteration。

**方案**: 在 Execution 前插入 Validation 阶段：

```
Planner → Critic → Architect → Coder → [Validator] → Execution → Judge
```

Validator 工作（全部通用）：
- Import 检查：`import src.preprocessing` 等
- 函数调用链：用 mock 数据 `np.zeros((4,4))` 走一遍 pipeline，确认形状兼容
- 梯度有限性：调一次 objective/gradient，确认返回有限值
- 输出格式：确认能生成 2D 数值数组

**改动范围**: `multi_agent.py`（Coder 后插入 validation）。

---

### 3.3 [高优先级] 迭代式质量提升（Self-Refine Loop）

**问题**: 当前 pipeline 只在失败时循环。v7 第一次成功就终止（exit code 0），但 NRMSE=0.7355 远未达到最优。"成功运行" ≠ "高质量重建"。

**方案**: Execution 成功后，不直接返回，而是：
1. 运行通用质量评估脚本（不用 ground truth）：
   - 数据拟合残差：`||A(x_hat) - y||² / ||y||²`
   - 优化器最终 cost 值和梯度范数
   - 重建图像的基本统计：动态范围、负值比例
2. 将信号反馈给 Judge，判断是否值得继续优化
3. Judge 可选择：(a) 接受 → 终止；(b) 路由回 Coder → 调参继续

**改动范围**: `multi_agent.py`（成功分支逻辑），`judge_agent.py`（支持质量评估）。

---

### 3.4 [高优先级] 多轮优化策略

**问题**: 许多逆问题需要 coarse-to-fine / regularization annealing 才能收敛到好解。Planner 可能只规划了单轮优化。

**方案**: 在 Planner 的 system prompt 中添加通用建议：
- 从强正则化开始，逐步减小
- 使用多轮优化 + warm-start
- 至少 300-1000 总迭代次数

**改动范围**: 仅 `planner_agent.py`，prompt 修改。

---

### 3.5 [中优先级] 更聪明的 Critic

**问题**: Critic 只做 PASS/REJECT，审核标准粗糙（4 条 checklist）。

**方案**:
1. 扩展 checklist（全部通用）：正则化存在性、超参数完整性、优化策略稳健性、数值稳定性、降维策略
2. Critic 输出评分 + PASS/REVISE/REJECT 三级决策

**改动范围**: `planner_agent.py`（Critic prompt）。

---

### 3.6 [中优先级] 结构化执行输出解析

**问题**: Judge 收到的是原始 stdout/stderr（截断到 2000 字符），需要自行解析，容易出错。

**方案**: 在 `multi_agent.py` 中实现 `_parse_execution_output()`，将原始文本结构化为：
- Error Summary（异常类型+消息）
- Error Location（文件名+行号）
- Optimizer Status（迭代次数+最终 cost）
- Full Log

**改动范围**: `multi_agent.py`。

---

### 3.7 [中优先级] Agent 自省 & 自检

**问题**: Agent 不验证自己的 forward model 是否和数据一致。

**方案**: Execution 成功后运行通用 self-check：
- Forward-model 一致性：`A(x_hat)` 是否大致匹配观测？
- 图像基本统计：min/max/mean/负值比例/动态范围

**改动范围**: `multi_agent.py`。

---

### 3.8 [低优先级] 并行文件实现

**问题**: Coder 串行实现 5 个文件，约 2.5 分钟。

**方案**: 同时生成互不依赖的文件（需分析 import 依赖图）。

**改动范围**: `multi_agent.py`（Coder 阶段），复杂度高。

---

## 4. 建议实施顺序

```
Phase 1: 确认 baseline
  → 运行 1-2 次评估，确认框架工作正常
  → 预期: NRMSE ≈ 0.74, NCC ≈ 0.69（单次可能波动到 0.7~1.0）

Phase 2: 高优先级（3.1 → 3.4 → 3.2 → 3.3）
  → 3.1 数值严谨性（最小改动，只改 prompt）
  → 3.4 多轮优化策略（只改 prompt）
  → 3.2 自动代码验证（中等改动）
  → 3.3 Self-Refine Loop（较大改动）
  → 每步后运行评估，记录指标，不好就回滚

Phase 3: 中优先级（3.5 → 3.6 → 3.7）
  → 3.5 增强 Critic
  → 3.6 结构化输出解析
  → 3.7 Agent 自省

Phase 4: 低优先级 + 泛化验证
  → 3.8 并行实现（可选）
  → 在其他任务上验证泛化
```

---

## 5. 关键文件清单

| 文件 | 作用 | 改动方向 |
|---|---|---|
| `evaluation_harness/multi_agent.py` | 流水线编排 | Validation、Self-Refine、结构化输出 |
| `evaluation_harness/agents/coder_agent.py` | Coder prompt | 数值计算规则 |
| `evaluation_harness/agents/planner_agent.py` | Planner + Critic | 多轮优化建议、增强 Critic |
| `evaluation_harness/agents/judge_agent.py` | Judge | 质量评估（不只是故障诊断）|
| `evaluation_harness/agents/architect_agent.py` | Architect | 通常不需要改 |
| `evaluation_harness/agents/base.py` | BaseAgent | 通常不需要改 |
| `evaluation_harness/scorer.py` | 指标计算 | 通常不需要改 |
| `docs/PROGRESS_REPORT.md` | 审计文档 | **每次改动后必须更新** |

---

## 6. 成功指标

### 单任务 (eht_black_hole_original)

| 指标 | 当前 v7 | 中期目标 | 终极目标 | Reference Best |
|---|---|---|---|---|
| **NRMSE** ↓ | 0.7355 | ≤ 0.60 | ≤ 0.30 | 0.2648 (Vis RML cal) |
| **NCC** ↑ | 0.6940 | ≥ 0.80 | ≥ 0.95 | 0.9669 (Vis RML cal) |
| **PSNR** ↑ | 20.50 | ≥ 22 | ≥ 28 | — |
| **SSIM** ↑ | 0.5617 | ≥ 0.70 | ≥ 0.90 | — |

### 跨任务泛化

- 在至少 2 个不同任务上运行，确认改进仍有效
- 可用任务：`eht_black_hole_original`, `light_field_microscope`, `hessian_sim`, `reflection_ODT`

---

## 7. 历史教训（避免踩坑）

### 7.1 已修复的 Bug（不要回退）

| Bug | 影响 | 修复位置 |
|---|---|---|
| `_format_failure_history()` 空函数体 | Agent 看不到历史错误 | `multi_agent.py` |
| Judge feedback 丢失 | Coder 不知道之前出了什么错 | `multi_agent.py` |
| 代码截断 | Judge 做出错误判断 | `judge_agent.py` |
| 卡死检测缺失 | 同一 agent 被反复分配 | `multi_agent.py` |
| `np.load` 缺少 `allow_pickle=True` | scorer 无法加载结果 | `scorer.py` |

### 7.2 成功的改进

1. **数据探索 (`_explore_data`)**: 自动给 agent 数据键名/形状，显著减少 KeyError
2. **Requirements 传递**: Agent 知道可用的包（numpy/scipy/matplotlib），不会尝试 import torch
3. **Critic 审核**: Critic 拒绝有问题的计划 → 产出更好的第二版
4. **全模块接口摘要**: Coder 能看到所有文件签名，减少跨模块 import 错误

### 7.3 失败的教训

1. **v3-v5 退步**: 修 bug 引入新 bug（failure history 空函数体），必须逐步验证
2. **过度路由**: Judge 把 Coder 的 bug 路由给 Architect → 不必要的代码重写
3. **单次运行不可靠**: LLM 随机性大，必须多次运行

---

## 8. Checklist（每次改动后检查）

- [ ] 改动是否 general-purpose？（不含任何 task-specific 术语/参数/算法名）
- [ ] 是否更新了 `docs/PROGRESS_REPORT.md`？
- [ ] 是否运行了至少 1 次评估并记录了 4 指标？
- [ ] 指标是否优于或持平 v7 baseline？（变差需回滚或解释）
- [ ] 代码是否通过 AST 语法检查？
- [ ] 是否查看了 `logs/interactions/` 中的最新日志？
- [ ] 是否 git commit + push 到 backup？

---

## 9. 运行命令快速参考

```bash
# === 工作目录 ===
cd /home/yjh/imaging-101

# === API 配置 ===
export BASE_URL="https://ai-gateway-internal.dp.tech/v1"
export API_KEY="sk-Zj3a7RQDVCXr-Axg-0gtkg"

# === End-to-End Multi-Agent 评估 ===
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --model "gemini-2.5-pro" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --max-iterations 10 \
    --timeout 7200 \
    --framework multi_agent \
    --output results -v

# === 查看最新结果 ===
ls -lt results/*end_to_end*.json | head -5

# === 查看交互日志 ===
ls -lt logs/interactions/*.md | head -3

# === AST 语法检查 ===
for f in evaluation_harness/multi_agent.py evaluation_harness/agents/*.py; do
  python3 -c "import ast; ast.parse(open('$f').read()); print('OK: $f')"
done

# === Git 备份 ===
git add -A && git commit -m "v<N>: <描述>" && git push backup main

# === 在其他任务上测试泛化 ===
python -m evaluation_harness run \
    --task light_field_microscope \
    --mode end_to_end \
    --model "gemini-2.5-pro" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --max-iterations 10 \
    --timeout 7200 \
    --framework multi_agent \
    --output results -v
```

---

## Appendix A: 指标定义

| 指标 | 公式 | 方向 |
|---|---|---|
| **NRMSE** | `‖x̂ - x‖₂ / ‖x‖₂` | ↓ 越小越好 |
| **NCC** | `⟨x̂, x⟩ / (‖x̂‖₂ · ‖x‖₂)` | ↑ 越大越好 |
| **PSNR** | `20 · log₁₀(max(x) / √MSE)` dB | ↑ 越大越好 |
| **SSIM** | 结构相似性（亮度+对比度+结构） | ↑ 越大越好 |

其中 `x` = ground truth (flux-normalized), `x̂` = reconstruction (flux-normalized)。

## Appendix B: 框架文件完整路径

```
evaluation_harness/
├── __init__.py
├── __main__.py              # CLI 入口
├── agent.py                 # ReAct Agent
├── config.py                # 配置 dataclass
├── docker_runner.py         # Docker 沙箱
├── llm_client.py            # LLM API 客户端
├── local_runner.py          # 本地沙箱
├── multi_agent.py           # ★ Multi-Agent 编排器
├── plan_scorer.py           # Plan 评分
├── prompts.py               # Prompt 模板
├── runner.py                # BenchmarkRunner
├── scorer.py                # 指标计算
├── visualizer.py            # 可视化
└── agents/
    ├── __init__.py
    ├── base.py              # BaseAgent
    ├── planner_agent.py     # Planner + Critic
    ├── architect_agent.py   # Architect
    ├── coder_agent.py       # Coder
    └── judge_agent.py       # Judge
```
