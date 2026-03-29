# Multi-Agent Framework Improvement — Handoff Prompt

> **交付对象**: 后续 Agent  
> **目标**: 复现当前最佳实验结果，然后系统性提升 multi-agent framework 的 end-to-end 效果  
> **红线**: 所有改动必须是 general-purpose 的，禁止任何 task-specific tuning  
> **最后更新**: 2026-03-29

---

## 0. 你必须理解的约束（读完再动手）

1. **禁止 Cheating**: 不得在 prompt 中注入 task-specific 的算法选择、超参数、数据格式、数学公式。所有 prompt 规则必须对任意 computational imaging 任务通用。
2. **禁止信息泄漏**: 不得读取 `evaluation/reference_outputs/`、ground truth 或任何评估目标数据来指导 agent。
3. **Prompt 必须 General**: 例如你不能写 "use closure phases for gain-invariance"（天文术语），但可以写 "use pre-computed features from data files"（通用建议）。
4. **框架必须泛化**: 所有改动必须能在不同任务（天文、医学、工业成像等）上同样适用。
5. **审计要求**: 每次更改必须更新 `docs/PROGRESS_REPORT.md`，记录：改了什么、为什么、是否 task-specific。

---

## 1. 当前状态摘要

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

### 1.3 当前最佳结果 (v7)

| 指标 | Agent v7 | Reference (Closure-only cal) | Reference (Amp+CP cal) | Reference (Vis RML cal) |
|---|---|---|---|---|
| **NRMSE** ↓ | **0.7355** | 0.8226 | 0.7043 | **0.2648** |
| **NCC** ↑ | **0.6940** | 0.7523 | 0.7630 | **0.9669** |
| Iterations | 1 | — | — | — |
| LLM Calls | 11 | — | — | — |
| Time | 664s | — | — | — |

**解读**:
- Agent 当前水平：略好于 Closure-only（cal），略差于 Amp+CP（cal），远差于 Vis RML（cal）
- **终极目标**: 逼近 Vis RML (cal) 的水平：**NRMSE ≤ 0.30, NCC ≥ 0.95**
- **中期目标**: 超过 Amp+CP (cal) 的水平：**NRMSE ≤ 0.70, NCC ≥ 0.77**
- 注意：这些 reference 数据中的 `(cal)` 表示"calibrated data"，`(corrupt)` 表示"corrupted data"。Agent 的表现取决于它选择了什么算法和数据处理方式。

### 1.4 v7 的已知问题

从交互日志分析：
1. **Runtime 错误未完全修复**: "index 421 is out of bounds for axis 0 with size 421" — 经典 off-by-one，说明 Coder 的数组操作不够严谨
2. **优化可能不充分**: 只跑了部分 optimization round 就停止
3. **缺乏自我验证**: Agent 没有对自己的输出做 sanity check（如检查重建图像是否呈现预期结构）
4. **Regularizer 可能缺失或不当**: 日志未明确记录使用了什么正则化，可能导致 over/under-regularization
5. **1 次迭代就成功返回**: 效率高但也意味着没有迭代改进的机会

---

## 2. 复现步骤

### 2.1 环境准备

```bash
cd /home/yjh/imaging-101

# API 配置
export BASE_URL="https://ai-gateway-internal.dp.tech/v1"
export API_KEY="sk-Zj3a7RQDVCXr-Axg-0gtkg"
```

### 2.2 复现 v7 最佳结果

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
    -v 2>&1 | tee logs/eval_runs/reproduce_$(date +%Y%m%d_%H%M%S).log
```

### 2.3 验证结果

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

### 2.4 预期

由于 LLM 随机性，单次运行结果可能浮动。建议运行 3 次取中位数。如果 3 次中位数的 NRMSE < 0.80 且 NCC > 0.60，说明复现成功。

---

## 3. 改进方向（General-Purpose Only）

以下是经过分析的、**完全通用的**改进方向。每个方向都不依赖于任何特定任务的知识。

### 3.1 [高优先级] 自动代码验证 & 渐进式测试

**问题**: Coder 写完代码后直接运行 `main.py`。如果有 bug，只能在最后的 Execution 阶段才发现，然后整体回到 Judge。这意味着一个微小的 off-by-one 错误会浪费整个 iteration。

**方案**: 在 Execution 前插入一个 **Validation 阶段**，让 Coder 或一个新的 Validator Agent 对每个模块做基本 sanity check：

```
Planner → Critic → Architect → Coder → [Validator] → Execution → Judge
```

Validator 的工作（全部通用，不涉及任何 task-specific 知识）：
- **Import 检查**: 尝试 `import src.preprocessing` 等，确认所有模块可正确 import
- **函数调用链检查**: 用极小的 mock 数据（如 `np.zeros((4,4))`）从 preprocessing → physics_model → solvers 走一遍，确认形状兼容
- **梯度有限性检查**: 对 solvers.py 的 objective/gradient 做一次调用，确认返回有限值（不是 NaN/Inf）
- **输出格式检查**: 确认 `main.py` 运行后 `output/reconstruction.npy` 是 2D 数值数组

**实现要点**:
- 在 `multi_agent.py` 的 Coder 阶段之后、Execution 之前，生成一个 `_validate_modules.py` 脚本
- 该脚本自动从 architecture summary 中读取函数签名，用 mock 数据测试
- 如果 validation 失败，直接回到 Coder 修复（不需要 Judge），节省一次 LLM 调用

**为什么这是 general 的**: 所有 computational imaging 任务都有 preprocessing → forward_model → solver 的管线结构。用 mock 数据测试形状兼容性是通用的软件工程实践。

---

### 3.2 [高优先级] 迭代式质量提升（Self-Refine Loop）

**问题**: 当前框架的循环只在失败时触发（Execution → Judge → 修复 → 重试）。如果第一次就"成功"了（`main.py` exit code 0，reconstruction.npy 存在），pipeline 直接终止。但"成功运行"≠"高质量重建"。v7 就是这种情况：exit code 0，但 NRMSE=0.7355，远未达到最优。

**方案**: 添加一个 **Self-Refine 机制**：即使 Execution 成功，如果 Agent 还有剩余迭代，让它分析自己的输出并尝试改进。

具体做法：
1. 在 `_validate_reconstruction` 后，额外计算一些**通用的质量信号**（不使用 ground truth）：
   - 数据拟合残差：`||A(x_hat) - y||² / ||y||²`（Agent 代码已计算 forward model，可以复用）
   - 优化器最终 cost 值和梯度范数
   - 重建图像的基本统计：动态范围、负值比例、频谱集中度
2. 将这些信号反馈给 Judge，让 Judge 判断是否值得继续优化
3. 如果 Judge 认为有改进空间（如 "优化器只跑了 100 步就停了" 或 "最终 cost 值很高"），路由回 Coder 调整超参数

**实现要点**:
- 在 `multi_agent.py` 的 Execution 成功分支中，不直接 return，而是运行一个 `_assess_quality.py` 脚本
- 这个脚本 import agent 自己写的模块，计算 data-fit residual 和 basic stats
- 将结果传给 Judge，并附上一条通用提示："Execution succeeded. Assess whether reconstruction quality is good enough or whether hyperparameters/optimization should be improved."
- Judge 可以选择：(a) 接受当前结果 → 终止；(b) 路由回 Coder → 调整超参

**为什么这是 general 的**: data-fit residual 是所有逆问题的通用质量指标。不使用 ground truth，只使用 agent 自己的 forward model 和观测数据。

**注意**: 需要设置一个迭代上限（如最多 self-refine 2 次），避免无限循环。

---

### 3.3 [高优先级] 增强 Coder 的数值严谨性

**问题**: v7 中的 "index 421 is out of bounds" 是经典的 off-by-one 错误。Coder 在处理数组索引、循环边界、切片操作时不够严谨，这类 bug 在科学计算中极为常见且致命。

**方案**: 在 Coder 的 system prompt 中添加**通用的数值计算最佳实践**规则：

候选规则（全部通用）：
- "Array Indexing Safety: 使用 `array[i:i+n]` 而非 `array[i:i+n+1]`。任何循环 `for i in range(N)` 中对数组的访问都应确认 `i < array.shape[axis]`。"
- "Gradient Verification: 实现 objective function 后，用 `scipy.optimize.approx_fprime` 做一次有限差分验证。如果 analytic gradient 和 finite-diff gradient 的相对误差 > 1%，说明 gradient 有 bug。"
- "Numerical Stability: 所有除法 `a/b` 都应使用 `a/(b + epsilon)` 其中 `epsilon=1e-10`。所有 `np.log(x)` 都应使用 `np.log(np.maximum(x, epsilon))`。"
- "Shape Assertion: 在关键计算前后加 `assert result.shape == expected_shape, f'Expected {expected_shape}, got {result.shape}'`。"

**实现方式**: 在 `coder_agent.py` 的 `_build_system_prompt` 中添加一个 "### Numerical Computing Best Practices" 部分。

---

### 3.4 [中优先级] 更聪明的 Planner-Critic 交互

**问题**: 当前 Critic 只做 PASS/REJECT 的二元决策，且审核标准较粗糙（4 条 checklist）。它无法评估计划的**精细质量**，例如 regularization weight 的量级是否合理、优化迭代次数是否足够。

**方案**: 增强 Critic 的评估能力：

1. **Critic 的 checklist 扩展**（全部通用）：
   - ✅ 正则化存在性：计划是否包含正则化项？没有正则化的逆问题几乎必定失败
   - ✅ 超参数完整性：所有 hyperparameters 是否有明确的初始值？
   - ✅ 优化策略稳健性：optimizer 迭代次数是否 >= 100？是否有多轮优化策略？
   - ✅ 数值稳定性：是否提到了 log-transform / positivity constraint 等稳定化技巧？
   - ✅ 降维策略：如果问题欠定（N_unknowns >> N_observations），是否有降维或正则化策略？

2. **Critic 可输出评分而非仅 PASS/REJECT**：
   ```json
   {
     "decision": "PASS" | "REVISE" | "REJECT",
     "score": 0.0-1.0,
     "strengths": ["..."],
     "weaknesses": ["..."],
     "suggestion": "..."
   }
   ```
   - PASS (score >= 0.8): 直接进入 Architect
   - REVISE (0.5 <= score < 0.8): 给出具体改进建议，再迭代一轮
   - REJECT (score < 0.5): 完全重写

---

### 3.5 [中优先级] Execution 输出的结构化解析

**问题**: 当前 Judge 收到的执行日志是原始 stdout/stderr 文本（最多 2000 字符截断）。Judge 需要自己从文本中解析错误类型、行号、变量值。这容易出错，且长输出会被截断丢失关键信息。

**方案**: 在 Execution 阶段对输出做结构化预处理：

1. 将 `python main.py` 的输出分为三部分传给 Judge：
   - **Error Summary**: 提取 traceback 最后一行（异常类型 + 消息）
   - **Error Location**: 提取 traceback 中的文件名、行号、函数名
   - **Optimizer Status**: 提取优化器相关输出（iterations, final cost, convergence status）
   - **Full Log**: 完整日志（截断到合理长度）

2. 在 `multi_agent.py` 中实现 `_parse_execution_output(output: str) -> Dict`，将原始文本结构化

**实现示例（通用）**:
```python
def _parse_execution_output(self, output: str) -> Dict:
    result = {"error_type": None, "error_msg": None, "error_file": None, 
              "error_line": None, "optimizer_iters": None, "final_cost": None}
    # Extract last exception
    tb_match = re.search(r'(\w+Error): (.+)', output)
    if tb_match:
        result["error_type"] = tb_match.group(1)
        result["error_msg"] = tb_match.group(2)
    # Extract file/line from traceback
    file_matches = re.findall(r'File "([^"]+)", line (\d+)', output)
    if file_matches:
        result["error_file"] = file_matches[-1][0]
        result["error_line"] = int(file_matches[-1][1])
    # Extract optimizer info
    iter_match = re.search(r'(?:iterations?|nit)\s*[:=]\s*(\d+)', output, re.I)
    if iter_match:
        result["optimizer_iters"] = int(iter_match.group(1))
    return result
```

---

### 3.6 [中优先级] 多轮优化策略的通用支持

**问题**: 许多逆问题需要多轮优化（coarse-to-fine, warm-start, regularization annealing）才能收敛到好解。但当前 Planner 可能只规划了简单的单轮优化。

**方案**: 在 Planner 的 system prompt 中添加通用建议：

```
9. **Multi-Round Optimization**: For inverse problems, a single optimization run
   often converges to a local minimum. Consider:
   - Start with strong regularization (large λ), then gradually decrease
   - Use multiple optimization rounds with decreasing tolerance
   - Warm-start each round from the previous solution
   - Run at least 300-1000 total optimizer iterations across all rounds
```

**为什么这是 general 的**: 这是逆问题优化的标准工程实践，不依赖任何特定 task。

---

### 3.7 [低优先级] Agent 的自省能力（Reflection）

**问题**: Agent 在 Execution 成功后不会反思自己代码的质量。它不知道自己的 gradient 计算是否正确、自己的 forward model 是否和数据一致。

**方案**: 在 Execution 成功后，让 Agent 运行一个通用的 self-check 脚本：

```python
# _self_check.py (auto-generated, general-purpose)
import numpy as np
from src.preprocessing import load_data
from src.physics_model import ForwardModel  # or whatever the architecture defined

data = load_data()
x_hat = np.load('output/reconstruction.npy')

# 1. Forward-model consistency: does A(x_hat) roughly match observations?
model = ForwardModel(...)
y_pred = model.forward(x_hat)
y_obs = data['observations']  # key from data_inventory
residual = np.linalg.norm(y_pred - y_obs) / np.linalg.norm(y_obs)
print(f"Data-fit residual: {residual:.4f}")

# 2. Basic image stats
print(f"Image min={x_hat.min():.6f}, max={x_hat.max():.6f}, mean={x_hat.mean():.6f}")
print(f"Negative pixels: {(x_hat < 0).sum()} / {x_hat.size}")
print(f"Dynamic range: {x_hat.max() / (x_hat.min() + 1e-10):.1f}")
```

这个脚本不使用 ground truth，只用 agent 自己的 code 做一致性检查。

---

### 3.8 [低优先级] 并行文件实现

**问题**: 当前 Coder 逐个文件串行实现，每个文件一次 LLM 调用。对于 5 个文件 × 每次调用 30s = 2.5 分钟。

**方案**: 如果 LLM client 支持并行请求，可以同时生成互不依赖的文件（如 `preprocessing.py` 和 `visualization.py`），只对有依赖关系的文件串行（如 `solvers.py` 依赖 `physics_model.py`）。

**注意**: 这需要分析 architecture 中的 import 依赖图，复杂度较高，优先级最低。

---

## 4. 实施计划

### Phase 1: 复现验证（1-2 次运行）

1. 运行 Section 2.2 的命令，确认框架可正常工作
2. 对比结果与 v7 基线（NRMSE ≈ 0.74, NCC ≈ 0.69）
3. 如果大幅偏离，先排查环境问题

### Phase 2: 高优先级改进（3.1 + 3.2 + 3.3）

**建议实施顺序**:

1. **3.3 数值严谨性规则** — 最小改动，只修改 `coder_agent.py` 的 prompt
2. **3.1 自动代码验证** — 中等改动，修改 `multi_agent.py`，在 Execution 前插入 validation
3. **3.2 Self-Refine Loop** — 较大改动，修改 `multi_agent.py` 的成功分支逻辑

每次改动后：
- 运行 1 次评估，记录 4 指标（NRMSE, NCC, PSNR, SSIM）
- 更新 `docs/PROGRESS_REPORT.md`
- 如果指标变差，回滚该改动

### Phase 3: 中优先级改进（3.4 + 3.5 + 3.6）

1. **3.6 多轮优化策略** — 只改 planner prompt
2. **3.5 结构化输出解析** — 改 `multi_agent.py`
3. **3.4 增强 Critic** — 改 `planner_agent.py`

### Phase 4: 低优先级 + 泛化验证

1. 实现 3.7 和 3.8（可选）
2. **在非 EHT 任务上运行**，验证改进是否泛化

---

## 5. 关键文件清单

你需要关注的文件（按修改频率排序）：

| 文件 | 作用 | 你可能要改的地方 |
|---|---|---|
| `evaluation_harness/multi_agent.py` | 流水线编排 | 添加 Validation 阶段、Self-Refine、结构化输出 |
| `evaluation_harness/agents/coder_agent.py` | Coder prompt | 添加数值计算规则 |
| `evaluation_harness/agents/planner_agent.py` | Planner + Critic prompt | 添加多轮优化建议、增强 Critic |
| `evaluation_harness/agents/judge_agent.py` | Judge prompt | 支持质量评估（而非仅故障诊断）|
| `evaluation_harness/scorer.py` | 指标计算 | 可能不需要改 |
| `evaluation_harness/agents/base.py` | BaseAgent | 可能不需要改 |
| `docs/PROGRESS_REPORT.md` | 审计文档 | **每次改动后必须更新** |

---

## 6. 衡量成功的标准

### 短期（单任务）

| 指标 | 当前 v7 | 中期目标 | 终极目标 | Reference |
|---|---|---|---|---|
| **NRMSE** ↓ | 0.7355 | ≤ 0.60 | ≤ 0.30 | 0.2648 (Vis RML cal) |
| **NCC** ↑ | 0.6940 | ≥ 0.80 | ≥ 0.95 | 0.9669 (Vis RML cal) |
| **PSNR** ↑ | 20.50 | ≥ 22 | ≥ 28 | — |
| **SSIM** ↑ | 0.5617 | ≥ 0.70 | ≥ 0.90 | — |

### 长期（跨任务泛化）

- 在至少 2 个不同任务上运行 multi_agent，确认改进仍有效
- 可用任务：`eht_black_hole_original`, `light_field_microscope`, `hessian_sim`, `reflection_ODT` 等

---

## 7. 已做的优化及教训（避免踩坑）

以下是前序工作的关键教训，帮助你避免重复犯错：

### 7.1 已修复的 Bug（不要回退）

| Bug | 影响 | 修复位置 |
|---|---|---|
| `_format_failure_history()` 空函数体 | Agent 看不到历史错误 | `multi_agent.py` |
| Judge feedback 丢失 | Coder 不知道之前出了什么错 | `multi_agent.py` (last_judge_feedback) |
| 代码截断（未做智能截断） | Judge 看到截断的代码做出错误判断 | `judge_agent.py` (_identify_error_file) |
| 卡死检测缺失 | 同一 agent 被反复分配 | `multi_agent.py` (stuck detection) |
| L-BFGS-B constraints= 被忽略 | optimizer 直接返回初始值 | Planner/Coder prompt rules |
| `np.load` 缺少 `allow_pickle=True` | scorer 无法加载重建结果 | `scorer.py` |

### 7.2 成功的改进（可以继续沿用）

1. **数据探索 (`_explore_data`)**: 自动给 agent 数据键名/形状信息，显著减少 KeyError
2. **Requirements 传递**: Agent 知道只有 numpy/scipy/matplotlib，不会尝试 import torch
3. **Critic 审核**: v7 中 Critic 拒绝了有问题的计划，导致了更好的第二版计划
4. **全模块接口摘要**: Coder 能看到所有文件的 import/签名，减少跨模块 import 错误

### 7.3 失败的教训

1. **v3-v5 退步**: 早期修复引入了新问题（failure history 空函数体），说明必须逐步验证
2. **过度路由**: Judge 把 Coder 的 bug 路由给 Architect，导致不必要的代码重写
3. **单次运行不可靠**: v7 好但 v6 差，说明 LLM 随机性很大，必须多次运行取统计量

---

## 8. 运行命令快速参考

```bash
# === 工作目录 ===
cd /home/yjh/imaging-101

# === End-to-End Multi-Agent ===
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --model "gemini-2.5-pro" \
    --base-url "https://ai-gateway-internal.dp.tech/v1" \
    --api-key "sk-Zj3a7RQDVCXr-Axg-0gtkg" \
    --max-iterations 10 \
    --timeout 7200 \
    --framework multi_agent \
    --output results -v

# === 查看结果 ===
ls -lt results/*end_to_end*.json | head -5
python3 -c "
import json; d = json.load(open('results/<latest>.json'))
print(d['quality_metrics'])"

# === 查看交互日志 ===
ls -lt logs/interactions/*.md | head -3
cat logs/interactions/<latest>.md

# === 在新任务上测试泛化 ===
python -m evaluation_harness run \
    --task light_field_microscope \
    --mode end_to_end \
    --model "gemini-2.5-pro" \
    --base-url "https://ai-gateway-internal.dp.tech/v1" \
    --api-key "sk-Zj3a7RQDVCXr-Axg-0gtkg" \
    --max-iterations 10 \
    --timeout 7200 \
    --framework multi_agent \
    --output results -v
```

---

## 9. Checklist（每次改动后检查）

- [ ] 改动是否 general-purpose？（不含任何 task-specific 术语/参数/算法名）
- [ ] 是否更新了 `docs/PROGRESS_REPORT.md`？
- [ ] 是否运行了至少 1 次评估并记录了 4 指标？
- [ ] 指标是否优于或持平 v7 baseline？（如果变差，需要回滚或解释）
- [ ] 代码是否通过了 `python -c "import ast; ast.parse(open('file').read())"` 语法检查？
- [ ] 是否检查了 `logs/interactions/` 中的最新日志，确认 agent 行为符合预期？

---

## Appendix A: 指标定义

| 指标 | 公式 | 含义 | 方向 |
|---|---|---|---|
| **NRMSE** | `‖x̂ - x‖₂ / ‖x‖₂` | 归一化均方根误差 | ↓ 越小越好 |
| **NCC** | `⟨x̂, x⟩ / (‖x̂‖₂ · ‖x‖₂)` | 归一化互相关 | ↑ 越大越好 |
| **PSNR** | `20 · log₁₀(max(x) / √MSE)` | 峰值信噪比 (dB) | ↑ 越大越好 |
| **SSIM** | 结构相似性（亮度+对比度+结构） | 感知质量 | ↑ 越大越好 |

其中 `x` = ground truth, `x̂` = reconstruction (flux-normalized)。

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
├── plan_scorer.py           # Plan 评分（LLM-as-judge）
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
