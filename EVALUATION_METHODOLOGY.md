# imaging-101 评估方法论

**日期**: 2026-03-27
**版本**: v3 — 基于源码深度审计更新

---

## 目录

1. [项目概述与架构](#一项目概述与架构)
2. [三种评估模式详解](#二三种评估模式详解)
3. [评分体系](#三评分体系)
4. [Agent 交互协议](#四agent-交互协议)
5. [沙箱执行环境](#五沙箱执行环境)
6. [附录：三种模式对比总结](#附录三种模式对比总结)

---

## 一、项目概述与架构

### 1.1 项目定位

imaging-101 是一个面向**计算成像领域 LLM 编码智能体 (Coding Agent)** 的 Benchmark 评估框架。它通过三个递增难度的模式（Plan → Function → End-to-End）测试模型在理解物理问题、编写科学计算代码、端到端实现重建管线方面的能力。

### 1.2 代码架构

框架核心由 `evaluation_harness` 包实现，主要组件如下：

```
CLI 入口 (__main__.py)
 │  解析参数，构建 RunConfig
 │
 ├─→ BenchmarkRunner (runner.py) — 核心编排器
 │    │  根据 --mode 选择执行流程 (_run_plan_mode / _run_function_mode / _run_end_to_end_mode)
 │    │  控制 Agent 的文件可见性 (VISIBLE_PATHS)
 │    │  管理沙箱环境 (DockerRunner 或 LocalRunner)
 │    │
 │    ├─→ Agent (agent.py) — ReAct 循环
 │    │    │  驱动 LLM 进行 Thought → Action → Observation 交互
 │    │    │  支持 WRITE_FILE / RUN / READ_FILE / DONE 动作
 │    │    │  处理上下文窗口滑动和 Hallucination 过滤
 │    │    └─→ LLMClient (llm_client.py) — OpenAI 兼容接口
 │    │
 │    └─→ Scorer (scorer.py) — 评分系统
 │         ├─ Function 模式：解析 pytest 输出，计算通过率
 │         ├─ End-to-End 模式：计算图像质量指标 (NRMSE/NCC)
 │         └─ Plan 模式：调用 PlanScorer 进行 LLM-as-a-Judge 评估
 │
 └─→ Results JSON — 输出到 results/ 目录
```

### 1.3 文件可见性控制

为了模拟真实的开发场景并防止作弊，`BenchmarkRunner` 严格限制了 Agent 在不同模式下能看到的文件：

*   **Plan 模式**: 仅可见 `README.md` 和 `data/`。
*   **Function 模式**: 可见 `README.md`, `plan/`, `data/` 以及 **`evaluation/`** (包含测试用例)。
*   **End-to-End 模式**: 仅可见 `README.md` 和 `data/`。**不可见**参考代码或测试用例。

---

## 二、三种评估模式详解

### 2.1 Plan Mode (计划模式)

**目标**: 评估 Agent 理解科学问题并设计解决方案的能力。

**流程**:
1.  **Phase 1 (Approach)**: Agent 阅读 `README.md` 和 `data/meta_data`，生成 `plan/approach.md` (包含物理模型公式、算法步骤)。
2.  **Phase 2 (Design)**: Agent 读取自己生成的 `approach.md`，生成 `plan/design.md` (包含代码架构、函数签名)。

**特点**:
*   分阶段生成，模拟先思考后设计的流程。
*   不涉及代码实现。

### 2.2 Function Mode (函数模式)

**目标**: 评估 Agent 实现特定功能模块（如正向模型、求解器）的能力，并能通过单元测试。

**流程**:
1.  **依赖注入 (Dependency Seeding)**: Harness 会自动将 `src/` 下**除了目标模块以外**的所有参考实现复制到沙箱中。
    *   *例子*: 如果目标是 `physics_model.py`，Harness 会预先写入 `preprocessing.py`, `solvers.py` 等的参考代码。
    *   *目的*: 确保 Agent 可以 `import` 其他模块的正确实现，专注于当前目标函数的逻辑，避免因依赖缺失导致测试失败。
2.  **实现 (ReAct 循环)**: Agent 基于 ReAct (Thought-Action-Observation) 模式运行。它阅读 `README.md`, `plan/` 和对应的测试文件 (如 `evaluation/tests/test_physics_model.py`)，逐步编写代码并运行测试。
3.  **自测**: Agent 被鼓励运行 `pytest` 来验证自己的代码，并根据测试输出进行自我修正。

**特点**:
*   采用 ReAct Agent 架构，具备自我修正能力。
*   提供测试用例 (`evaluation/fixtures` 和 `evaluation/tests`) 给 Agent。
*   Agent 只需实现一个模块，环境已准备好上下文依赖。

**命令示例 (单个模块)**:
```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode function \
    --target-function preprocessing \
    --model cds/Claude-4.6-opus \
    --base-url https://ai-gateway-internal.dp.tech/v1 \
    --api-key $OPENAI_API_KEY
```

**命令示例 (批量测试所有模块)**:
创建一个 Shell 脚本 (如 `run_function_evals.sh`) 来遍历所有核心模块：

```bash
#!/bin/bash
TASK="eht_black_hole_original"
MODEL="cds/Claude-4.6-opus"
BASE_URL="https://ai-gateway-internal.dp.tech/v1"
MODULES=("preprocessing" "physics_model" "solvers" "visualization")

for module in "${MODULES[@]}"; do
    echo "Running function eval for $module..."
    python -m evaluation_harness run \
        --task "$TASK" \
        --mode function \
        --target-function "$module" \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --api-key "$OPENAI_API_KEY" \
        --output results
done
```

### 2.3 End-to-End Mode (端到端模式)

**目标**: 评估 Agent 从零构建完整成像管线的能力。

**流程**:
1.  **Planning Phase**: Agent 首先生成计划 (同 Plan 模式)。
2.  **Implementation Phase**: Agent 基于计划，实现整个 `src/` 目录下的所有代码以及 `main.py`。
3.  **Execution**: Agent 运行 `python main.py` 生成重建结果。

**特点**:
*   **黑盒测试**: Agent **无法看到**任何测试用例或参考代码。
*   必须独立处理所有模块的实现和整合。
*   最终通过运行 `main.py` 产出的图像质量来评分。

---

## 三、评分体系

### 3.1 Function 模式评分
基于 `pytest` 的测试结果：
*   **Pass Rate**: 通过的测试用例数 / 总测试用例数。
*   **详细报告**: 记录每个测试用例的通过/失败状态。

### 3.2 End-to-End 模式评分
基于重建结果的图像质量指标 (Ground Truth vs Reconstruction)：
*   **NRMSE** (Normalized Root Mean Square Error): 归一化均方根误差，越低越好。
*   **NCC** (Normalized Cross Correlation): 归一化互相关，越高越好 (接近 1.0)。
*   评分脚本会自动加载 `evaluation/reference_outputs/` 中的 Ground Truth 进行对比。

### 3.3 Plan 模式评分 (LLM-as-a-Judge)
使用 `PlanScorer` 进行多维评估：
1.  **Pairwise ELO**: 让 LLM (Judge) 对比 Agent 生成的计划与参考计划 (Reference Plan)，判断胜负或平局。
2.  **Rubric Scoring**: 基于评分标准 (Problem Understanding, Math Formulation, etc.) 进行 1-5 分打分。
3.  **综合得分**: 结合 Pairwise 胜率和 Rubric 分数。

---

## 四、Agent 交互协议

Agent 运行在 ReAct (Reasoning + Acting) 循环中。

**系统提示词 (System Prompts)**:
*   `prompts.py` 定义了不同模式的 Prompt。
*   Function 模式明确提示 "Available files: ... evaluation/" 并鼓励运行测试。
*   其他模式则不提及测试文件。

**动作空间 (Action Space)**:
1.  `WRITE_FILE`: 创建或覆盖文件 (需提供完整内容)。
2.  `READ_FILE`: 读取文件内容。
3.  `RUN`: 执行 Shell 命令。
4.  `DONE`: 任务完成信号。

**特殊机制**:
*   **Hallucination 过滤**: 代码会检测并移除模型输出中伪造的 `Observation:` 块，防止模型自问自答。
*   **滑动窗口**: 保持上下文在 LLM 的 Token 限制内。

---

## 五、沙箱执行环境

`runner.py` 支持两种运行时的沙箱：

1.  **DockerRunner (推荐)**:
    *   使用 `imaging101-sandbox` 镜像。
    *   通过 `docker exec` 执行命令，实现完全隔离。
    *   文件系统隔离在容器内。

2.  **LocalRunner (备用)**:
    *   当系统无 Docker 时自动回退。
    *   使用 `tempfile.mkdtemp()` 创建临时工作目录。
    *   **警告**: 直接在主机执行代码，安全性较低。

---

## 附录：三种模式对比总结

| 特性 | Plan Mode | Function Mode | End-to-End Mode |
| :--- | :--- | :--- | :--- |
| **主要目标** | 方案设计 | 模块实现 &通过测试 | 全流程重建 & 图像质量 |
| **Agent 可见文件** | README, Data | README, Plan, Data, **Tests** | README, Data |
| **依赖注入** | 无 | **有** (预置其他参考模块) | 无 (需全量实现) |
| **核心产出** | approach.md, design.md | 单个 .py 模块 | 完整 src/ 目录, main.py |
| **评分依据** | LLM 评分 (vs Reference) | 单元测试通过率 | 图像质量 (NRMSE/NCC) |

│   ├── solvers.py         # 逆问题求解器
│   ├── visualization.py   # 绘图与指标计算
│   └── generate_data.py   # 可选：合成数据生成
├── evaluation/
│   ├── reference_outputs/ # 参考输出：ground_truth.npy, metrics.json 等
│   ├── fixtures/          # 测试固件：per-function 输入/输出数据
│   └── tests/             # pytest 测试文件
│       ├── test_preprocessing.py
│       ├── test_physics_model.py
│       ├── test_solvers.py
│       ├── test_visualization.py
│       ├── test_end_to_end.py
│       └── test_parity.py  # 与原始库的数值一致性测试
└── notebooks/
    └── <task_name>.ipynb   # 教程 notebook
```

## 二、三种评估模式详解

### 2.1 模式间的文件可见性控制

`runner.py` 中的 `VISIBLE_PATHS` 字典精确控制每种模式下哪些文件被复制到沙箱中供 Agent 访问：

```python
VISIBLE_PATHS = {
    "plan":       ["README.md", "data"],
    "function":   ["README.md", "plan", "evaluation", "data"],
    "end_to_end": ["README.md", "data"],
}
```

**关键区别**：End-to-End 模式的 Agent **看不到** `evaluation/` 目录（包括所有测试文件、fixture 文件和 reference_outputs），也看不到 `plan/` 和 `src/`。Agent 只能看到 `README.md`（问题描述）和 `data/`（观测数据）。

---

### 2.2 Plan Mode（规划模式）— ★☆☆☆☆

**目标**：测试 LLM 能否从问题描述中**理解物理问题并制定数学方案**，不要求写代码。

**Agent 可见文件**：`README.md`, `data/`（仅元数据和观测数据）

**Agent 不可见**：`plan/`（参考方案）、`src/`（参考实现）、`evaluation/`（测试和参考输出）

**要求输出**：`plan/approach.md`（算法方案）+ `plan/design.md`（代码设计）

**执行流程**（对应 `runner.py::_run_plan_mode()`）：

```
Phase 1 — Approach:
  1. runner 从宿主机读取 README.md 和 data/meta_data
  2. 构造 plan_approach_prompt(readme, meta_data)
     提示 Agent 写出: 问题定义、数学公式、求解策略、预期结果
  3. Agent 通过 ReAct 循环 WRITE_FILE plan/approach.md → DONE

Phase 2 — Design:
  1. runner 从沙箱读取 Agent 刚生成的 plan/approach.md
  2. 构造 plan_design_prompt(readme, approach)
     提示 Agent 写出: 文件结构、函数签名、类定义、数据流
  3. Agent 通过 ReAct 循环 WRITE_FILE plan/design.md → DONE
```

**系统提示**：使用 `SYSTEM_PROMPT`（通用版），不提及 evaluation/tests。

**评分**：不运行 pytest。使用 **LLM-as-Judge** 双维度评分（详见第三节）。

---

### 2.3 Function Mode（函数级实现模式）— ★★★☆☆

**目标**：给定参考方案 + 单元测试，测试 LLM 能否**正确实现指定 Python 模块中的全部函数**。

**Agent 可见文件**：`README.md`, `data/`, `plan/`（approach.md + design.md）, `evaluation/`（tests + fixtures + reference_outputs）

**Agent 不可见**：`src/`（参考实现，Agent 必须自己从零实现目标模块）

**调用参数**：`--target-function <module_name>`，如 `preprocessing`、`physics_model`、`solvers`、`visualization`

**要求输出**：`src/<target_module>.py`

**执行流程**（对应 `runner.py::_run_function_mode()`）：

```
1. runner 从宿主机读取 README.md, plan/approach.md, plan/design.md
2. runner 读取 evaluation/tests/test_<module>.py 的完整源码
3. runner 调用 _seed_dependency_modules(module):
   遍历 src/ 目录，将所有非目标模块的参考实现复制到沙箱
   例如目标是 physics_model → 复制 preprocessing.py, solvers.py, visualization.py
   确保 src/__init__.py 存在
   目标模块本身 不复制 — Agent 必须自己实现
4. 构造 function_prompt(readme, approach, design, target, test_content)
   Prompt 包含: 问题描述 + 方案 + 代码设计 + 目标函数名 + 完整测试代码
5. Agent 通过 ReAct 循环:
   READ_FILE 查看 fixture → WRITE_FILE src/<module>.py →
   RUN pytest evaluation/tests/test_<module>.py -v →
   修复 → ... → DONE
```

**系统提示**：使用 `SYSTEM_PROMPT_FUNCTION`（函数模式专用），提及 `evaluation/` 目录的存在。

**DONE 门控**：Agent 必须至少运行过一次 pytest 才允许 DONE。否则 Agent 收到提示强制继续。

**评分**：pytest 通过率 = `passed / total`。

**`_seed_dependency_modules()` 的作用**：这是一个关键设计。由于测试文件存在跨模块 import（如 `test_physics_model.py` 中 `from src.preprocessing import load_observation`），如果不将非目标模块的参考实现预置到沙箱，Agent 的测试会因 `ModuleNotFoundError` 全部失败。此机制确保测试失败只能归因于 Agent 实现的目标模块本身的问题。

---

### 2.4 End-to-End Mode（端到端实现模式）— ★★★★★

**目标**：**最高难度**。Agent 仅获得问题描述和观测数据，必须**从零开始自主完成全部工作**：理解问题 → 制定方案 → 实现全部代码 → 运行管线 → 产出重建结果。**不提供任何测试用例、参考方案或参考代码**。

**Agent 可见文件**：仅 `README.md` 和 `data/`

**Agent 不可见**：`plan/`、`src/`、`evaluation/`（全部不可见）

**要求输出**：`plan/approach.md` + `plan/design.md` + `src/*.py` + `main.py` + `output/reconstruction.npy`

**执行流程**（对应 `runner.py::_run_end_to_end_mode()`）：

```
Phase 1 — Planning (使用部分 max_iterations 配额):
  1. runner 从宿主机读取 README.md 和 data/meta_data
  2. 构造 end_to_end_plan_prompt(readme, meta_data):
     "You will implement a full computational imaging pipeline from scratch.
      First, create the solution plan:
      1. Write plan/approach.md
      2. Write plan/design.md"
  3. Agent 通过 ReAct 循环:
     READ_FILE 查看数据 → WRITE_FILE plan/approach.md →
     WRITE_FILE plan/design.md → DONE

Phase 2 — Implementation (使用剩余 max_iterations 配额):
  1. runner 从沙箱读取 Agent 在 Phase 1 生成的 approach.md 和 design.md
  2. 构造 end_to_end_impl_prompt(approach, design):
     "Implement the full reconstruction pipeline following the plan below.
      CRITICAL INSTRUCTIONS:
      1. Explore data/ directory
      2. Write src/__init__.py
      3. Implement source modules (自由组织代码结构)
      4. Write main.py
      5. Run: python main.py
      Pipeline MUST produce output/reconstruction.npy"
  3. Agent 通过 ReAct 循环:
     WRITE_FILE src/*.py → WRITE_FILE main.py →
     RUN python main.py → 验证输出 → DONE
```

**系统提示**：使用 `SYSTEM_PROMPT`（通用版），**不提及** evaluation/tests 目录，因为 Agent 看不到任何测试。

**DONE 门控**：Agent 必须至少运行过一次包含 `main.py` 的命令才允许 DONE。否则强制继续，提示 "You must run main.py to produce output/reconstruction.npy before signaling DONE."

**⚠️ 评分方式 — 这是与 Function Mode 的核心区别**：

End-to-End 模式**不运行 pytest，不计算测试通过率**。评分完全基于**重建质量指标**：

1. Scorer 从**宿主机任务目录**的 `evaluation/reference_outputs/ground_truth.npy` 复制到沙箱
2. 在沙箱中执行 Python 脚本，对比 Agent 产出的 `output/reconstruction.npy` 与真实图像
3. 计算两个指标：
   - **NRMSE** = ‖output − gt‖₂ / ‖gt‖₂ （越小越好，0 = 完美匹配）
   - **NCC** = (output · gt) / (‖output‖₂ · ‖gt‖₂) （越大越好，1.0 = 完美匹配）
4. 对比前先做 **flux 归一化**：`output = output × (gt.sum() / output.sum())`

具体计算代码（`scorer.py::_compute_quality_metrics()` 注入沙箱执行的 snippet）：

```python
import numpy as np, json, sys, os
out = np.load("output/reconstruction.npy")
gt = np.load("evaluation/reference_outputs/ground_truth.npy")
# Flux-normalize
out = out * (gt.sum() / (out.sum() + 1e-30))
nrmse = float(np.linalg.norm(out - gt) / (np.linalg.norm(gt) + 1e-30))
ncc = float(np.sum(out * gt) / (np.linalg.norm(out) * np.linalg.norm(gt) + 1e-30))
print(json.dumps({"nrmse": round(nrmse, 4), "ncc": round(ncc, 4)}))
```

**特别说明**：虽然 `evaluation/tests/test_end_to_end.py` 文件存在于任务目录中，但它**不参与 End-to-End 模式的评分**。这些 test_end_to_end.py 文件的设计目的是验证**参考实现**的正确性（例如验证 metrics.json 中各方法的 NCC/NRMSE 是否符合预期物理行为），而非评估 Agent 的输出。在 End-to-End 模式中，`scorer.py` 中的 `score()` 方法只调用 `_compute_quality_metrics()`，完全不调用 `_run_tests()`。

## 三、评分体系

### 3.1 Function Mode 评分：pytest 通过率

`scorer.py::_run_tests()` 在沙箱中执行 pytest 并解析输出：

```python
# 执行命令
test_cmd = f"python -m pytest evaluation/tests/test_{module}.py -v --tb=short --no-header"
output, _ = self.runner.exec(test_cmd)
```

解析逻辑：
1. 用正则 `(\S+::\S+)\s+(PASSED|FAILED|ERROR)` 从 pytest `-v` 输出中提取每个测试的结果
2. 用正则 `(\d+)\s+passed` 和 `(\d+)\s+failed` 从摘要行提取计数
3. 如果摘要行被截断（`total == 0 and details`），fallback 到逐条计数

输出结构：
```json
{
  "tests_total": 11,
  "tests_passed": 11,
  "tests_failed": 0,
  "test_pass_rate": 1.0,
  "test_details": [
    {"test": "test_physics_model.py::TestForward::test_output_shape", "status": "PASSED"},
    ...
  ]
}
```

### 3.2 End-to-End Mode 评分：重建质量指标

`scorer.py::_compute_quality_metrics()` 的完整流程：

1. **注入 ground truth**：从宿主机 `evaluation/reference_outputs/ground_truth.npy` 复制到沙箱的 `evaluation/reference_outputs/ground_truth.npy`
2. **注入评分脚本**：通过 `runner.exec()` 执行内联 Python 代码
3. **Flux 归一化**：`out = out × (gt.sum() / (out.sum() + 1e-30))`
   - 这消除了绝对亮度标度差异，只评估空间结构的相似性
4. **NRMSE 计算**：`‖out - gt‖₂ / (‖gt‖₂ + 1e-30)`
5. **NCC 计算**：`Σ(out × gt) / (‖out‖₂ × ‖gt‖₂ + 1e-30)`
6. **JSON 返回**：`{"nrmse": float, "ncc": float}`

如果 `output/reconstruction.npy` 不存在，返回 `{"error": "output/reconstruction.npy not found"}`。

输出结构：
```json
{
  "quality_metrics": {"nrmse": 0.8226, "ncc": 0.7523}
}
```

**注意**：End-to-End 的结果 JSON 中 `tests_total`, `tests_passed`, `test_pass_rate` 均为 0，`plan_scores` 为 null。

### 3.3 Plan Mode 评分：LLM-as-Judge

`plan_scorer.py::evaluate_plan()` 实现两个维度的评分，由同一个 LLM 充当评判者：

#### 3.3.1 Pairwise Comparison（配对对比，ELO-inspired）

将 Agent 生成的计划与任务中的**参考计划**（`plan/approach.md` + `plan/design.md`）对比：

- **3 轮评判**，交替交换 A/B 位置以消除位置偏差：
  - Round 1: A=生成计划, B=参考计划
  - Round 2: A=参考计划, B=生成计划（swap=True，结果取反）
  - Round 3: A=生成计划, B=参考计划
- 每轮输出 `[[A is Better]]` / `[[B is Better]]` / `[[Tie]]`
- 解析为 A 得分: 1.0 / 0.0 / 0.5，swap 轮取反后变为生成计划的得分
- `pairwise_win_rate` = 三轮得分的算术平均

评判标准（写在 prompt 中）：
1. **Correctness**：物理/数学/算法是否正确
2. **Completeness**：是否覆盖全管线阶段
3. **Mathematical Precision**：方程是否显式且正确
4. **Code Architecture**：函数签名/模块结构/数据流是否清晰
5. **Implementability**：能否仅凭此文档实现完整管线

#### 3.3.2 Rubric Scoring（维度评分）

6 个维度，每个 1-5 分 + 文字反馈：

| 维度 | 权重 | 考察内容 |
|:-----|:-----|:---------|
| `problem_understanding` | 15% | 对物理问题的理解深度 |
| `mathematical_formulation` | 25% | 数学公式推导的正确性和完整性 |
| `algorithm_design` | 20% | 算法方案的合理性 |
| `code_architecture` | 20% | 代码结构设计 |
| `completeness` | 10% | 方案的完整性 |
| `scientific_correctness` | 10% | 科学准确性 |

加权平均：`rubric_weighted_avg = Σ(score_i × weight_i) / Σ(weight_i)`

#### 3.3.3 综合评分

```
rubric_normalized = (rubric_weighted_avg - 1) / 4    # 映射 1→0, 5→1
overall_score = 0.5 × pairwise_win_rate + 0.5 × rubric_normalized
```

范围: [0, 1]。如果没有参考计划则仅用 rubric_normalized。

### 3.4 各模式评分汇总

| 模式 | 评分指标 | 评分方法 | 结果字段 |
|:-----|:---------|:---------|:---------|
| Plan | overall_score ∈ [0,1] | LLM-as-Judge | `plan_scores` |
| Function | test_pass_rate ∈ [0,1] | pytest 通过率 | `tests_*` |
| End-to-End | nrmse, ncc | 重建 vs ground truth | `quality_metrics` |

## 四、Agent 交互协议

### 4.1 ReAct 循环

Agent 类（`agent.py`）实现 **Thought → Action → Observation** 循环：

```
初始化:
  messages = [system_prompt, user_prompt]

循环 (最多 max_iterations 次):
  1. 对 messages 应用滑动窗口 → windowed
  2. 调用 LLM: response_text = client.chat(windowed)
  3. 解析 response_text 中的所有 Action
  4. 检查是否有伪造的 Observation + DONE（模拟多轮）
  5. 逐个执行 Action，收集 Observation
  6. 如果遇到 DONE → 检查门控条件 → 如通过则退出
  7. 将 Observation 追加到 messages，继续下一轮
```

### 4.2 可用操作

| 操作 | 格式 | 说明 |
|:-----|:-----|:-----|
| `WRITE_FILE` | `Action: WRITE_FILE\nPath: <path>\nContent:\n<内容>\nEND_CONTENT` | 创建/覆盖文件 |
| `RUN` | `Action: RUN\nCommand: <命令>` | 在沙箱中执行 shell 命令 |
| `READ_FILE` | `Action: READ_FILE\nPath: <path>` | 读取文件内容 |
| `DONE` | `Action: DONE` | 标记任务完成 |

### 4.3 多动作解析器

`_parse_all_actions()` 从 LLM 回复中提取**所有** Action（模型有时在一条回复中模拟多轮对话）。用正则 `^Action:\s*(\S+)` 定位每个动作，按顺序提取参数。遇到 DONE 立即终止列表。

### 4.4 伪造 DONE 检测

如果 LLM 回复中同时包含 `Observation:` 块和 `DONE`，说明模型在模拟多轮对话。Agent 会剥离 DONE，只执行真实 Action。

### 4.5 DONE 门控

| 模式 | 门控条件 | 不满足时的行为 |
|:-----|:---------|:---------------|
| Plan | 无门控 | 立即接受 DONE |
| Function | `commands_run` 中至少有一条包含 `pytest` | 强制继续，提示运行 pytest |
| End-to-End | `commands_run` 中至少有一条包含 `main.py` | 强制继续，提示运行 main.py |

### 4.6 上下文管理

`_apply_sliding_window()` 实现三层保护防止上下文溢出：

1. **滑动窗口**（`max_context_messages=10`）：保留 messages[0]（system）+ messages[1]（initial user）+ 最近 N-2 条。被丢弃的中间消息替换为一条摘要占位符。
2. **总字符预算**（`MAX_TOTAL_CHARS=90,000`）：超出时找最长的非 system/initial 消息，截断到 3,000 字符（保留首 1,500 + 尾 1,200）。
3. **单消息截断**：Observation 在追加前截断到 12,000 字符；RUN 命令输出截断到 8,000 字符。

### 4.7 系统提示差异

- **`SYSTEM_PROMPT`**（Plan/End-to-End 模式）：提及 `data/` 和 `README.md`，不提及 `evaluation/`
- **`SYSTEM_PROMPT_FUNCTION`**（Function 模式）：额外提及 `evaluation/ (fixtures and tests)`

## 五、沙箱执行环境

框架支持两种沙箱后端，通过 `_docker_available()` 自动选择：

### 5.1 DockerRunner（`docker_runner.py`）

- **创建**：`docker run -d --name imaging101-<uuid> -v <task_dir>:/workspace_src:ro -w /workspace --memory=4g --cpus=2 <image> sleep infinity`
- **文件隔离**：任务目录以只读方式挂载到 `/workspace_src`，Agent 操作的 `/workspace` 是容器内的可写目录
- **可见文件**：通过 `cp -a /workspace_src/<path> /workspace/<path>` 选择性复制
- **命令执行**：`docker exec <container> bash -c "timeout <T> <cmd>"`
- **文件写入**：`docker exec -i <container> bash -c "cat > <path>"` + stdin 管道
- **清理**：`docker rm -f <container>`，所有 Agent 文件随容器销毁

### 5.2 LocalRunner（`local_runner.py`）— Docker 不可用时的替代

- **创建**：`tempfile.mkdtemp(prefix="imaging101-local-")` 创建临时目录
- **可见文件**：`shutil.copytree / shutil.copy2` 复制到临时目录
- **命令执行**：`subprocess.run(["bash", "-c", cmd], cwd=workspace, env=safe_env, timeout=T)`
  - 环境变量白名单：仅 PATH, LANG, PYTHONPATH=workspace, HOME=workspace
- **路径映射**：自动将 LLM 输出中的 `/workspace/` 替换为实际临时目录路径，`/workspace_src/` 替换为宿主机任务目录
- **路径安全**：`_resolve_path()` 验证所有路径不逃逸出 workspace
- **二进制文件保护**：READ_FILE 遇到 `.npz/.npy/.pkl` 等二进制扩展名时返回提示信息
- **代码归档**：stop() 时将 Agent 生成的 `src/`、`plan/`、`output/` 复制到 `/data/yjh/function_eval_code_archive/<timestamp>_<task>/`，然后删除临时目录
- **依赖安装**：如果任务有 `requirements.txt`，在临时目录中创建 venv 并安装

### 5.3 两种 Runner 的统一接口

| 方法 | 签名 | 说明 |
|:-----|:-----|:-----|
| `start(visible_paths)` | `list[str] → None` | 创建沙箱并复制可见文件 |
| `exec(command)` | `str → (str, int)` | 执行命令，返回 (output, exit_code) |
| `write_file(path, content)` | `str, str → None` | 写文件到沙箱 |
| `read_file(path)` | `str → str` | 从沙箱读文件 |
| `stop()` | `→ None` | 销毁沙箱 |

## 六、当前任务详解

### 6.1 任务总览

目前框架包含两个完整实现的任务，均属于 EHT（Event Horizon Telescope）黑洞成像领域：

| 属性 | eht_black_hole_original | eht_black_hole_UQ |
|:-----|:------------------------|:-------------------|
| 难度 | Hard | Hard |
| 核心论文 | Chael 2018 (ApJ 857 23) | Sun 2020 (arXiv 2010.14462) |
| 成像方法 | 闭合量 RML（正则化最大似然） | DPI（深度概率成像，Normalizing Flow） |
| 图像大小 | 64×64 | 32×32 |
| 数据格式 | raw_data.npz (NumPy) | obs.uvfits + gt.fits (FITS) |
| 观测基线 | 421 条（7站 EHT） | 938 条可见度（230 GHz） |
| 测试总数 | 63 | 86（62 功能 + 24 parity） |
| 参考 NCC | 0.6044（closure-only corrupt） | 0.9165（posterior mean） |

---

### 6.2 eht_black_hole_original：闭合量 VLBI 成像

#### 6.2.1 物理问题

射电干涉测量中，各天线的增益误差会污染可见度（visibility）数据。闭合量（closure quantities）是一组对增益误差免疫的观测量：
- **闭合相位**（Closure Phase）：三条基线组成的三角形上可见度相位之和，消除了站点相位误差
- **对数闭合振幅**（Log Closure Amplitude）：四条基线组成的四边形上可见度振幅的特定比值取对数，消除了站点增益误差

任务目标：仅使用闭合量重建 M87* 黑洞图像，证明即使在严重增益腐蚀下仍能恢复可靠的图像结构。

#### 6.2.2 源代码模块（参考实现）

**`src/preprocessing.py`** — 数据加载与闭合量计算
| 函数 | 功能 |
|:-----|:-----|
| `load_observation(data_dir)` | 加载 raw_data.npz：可见度、UV坐标、噪声σ、站点ID |
| `load_metadata(data_dir)` | 加载 meta_data JSON：npix, fov, 正则化权重等 |
| `find_triangles(ant1, ant2)` | 从基线列表中枚举所有三站三角形及对应基线索引 |
| `find_quadrangles(ant1, ant2)` | 枚举所有四站四边形及对应基线索引 |
| `_find_baseline(ant1, ant2, a1, a2)` | 查找两站间的基线索引（内部辅助函数） |
| `compute_closure_phases(vis, triangles)` | 从可见度计算闭合相位 |
| `compute_log_closure_amplitudes(vis, quadrangles)` | 从可见度计算对数闭合振幅 |
| `closure_phase_sigma(sigma, triangles)` | 闭合相位的噪声传播 |
| `closure_amplitude_sigma(sigma, quadrangles)` | 闭合振幅的噪声传播 |
| `prepare_data(data_dir)` | 整合以上步骤，返回完整数据包 |

**`src/physics_model.py`** — DFT 正向模型与 χ² 数据项
| 类/函数 | 功能 |
|:--------|:-----|
| `_triangle_pulse_F(u, v, pdim)` | 三角脉冲像素响应的傅里叶变换 |
| `_ftmatrix(uv, npix, fov)` | 构建 DFT 矩阵（UV→图像空间的正向映射） |
| `ClosureForwardModel` | 封装正向/伴随/脏图/PSF 计算 |
| `.forward(image)` | 图像 → 可见度（DFT） |
| `.adjoint(vis)` | 可见度 → 脏图（逆 DFT） |
| `.dirty_image()` / `.psf()` | 脏图和点扩散函数 |
| `.visibility_chisq` / `_grad` | 可见度 χ² 及其梯度 |
| `.chisq_cphase_from_uv` (static) | 从 UV 可见度计算闭合相位 χ² |
| `.chisqgrad_cphase_from_uv` (static) | 闭合相位 χ² 的梯度 |
| `.chisq_logcamp_from_uv` (static) | 对数闭合振幅 χ² |
| `.chisqgrad_logcamp_from_uv` (static) | 对数闭合振幅 χ² 的梯度 |

**`src/solvers.py`** — 正则化器与 L-BFGS-B 求解器
| 类 | 功能 |
|:---|:-----|
| `GullSkillingRegularizer` | Gull-Skilling 最大熵正则化 |
| `SimpleEntropyRegularizer` | 简单熵正则化 |
| `TVRegularizer` | 全变分（TV）正则化，使用周期边界（`np.roll`） |
| `ClosureRMLSolver` | 闭合量 RML 求解器（L-BFGS-B，bounds=[0,∞)） |
| `VisibilityRMLSolver` | 可见度 RML 求解器（对比基准） |

**`src/visualization.py`** — 指标计算与绘图
| 函数 | 功能 |
|:-----|:-----|
| `compute_metrics(recon, gt)` | 计算 NRMSE + NCC（flux 归一化后） |
| `print_metrics_table(results)` | 打印对比表格 |
| `plot_reconstruction(image, ...)` | 绘制单张重建图 |
| `plot_comparison(results, gt)` | 绘制多方法对比图 |

#### 6.2.3 管线流程（main.py）

```
1. load_observation + load_metadata → 观测数据
2. ClosureForwardModel(obs, meta) → 正向模型
3. 创建高斯先验图像（用于正则化器的参考图像）
4. 四次重建:
   ├─ Vis RML (calibrated)     — 使用校准后可见度
   ├─ Vis RML (corrupt)        — 使用增益腐蚀后可见度（对比：增益腐蚀导致失败）
   ├─ Closure-only (calibrated) — 仅使用闭合量
   └─ Closure-only (corrupt)   — 仅使用闭合量（核心：证明闭合量对增益免疫）
5. compute_metrics → 计算 NRMSE/NCC → output/metrics.json
6. 保存最佳重建 → output/reconstruction.npy（closure-only corrupt 方法）
```

#### 6.2.4 参考指标（metrics.json）

```json
{
  "vis_rml_cal":         {"nrmse": 0.3609, "ncc": 0.9669},
  "vis_rml_corrupt":     {"nrmse": 1.4145, "ncc": 0.0001},
  "amp_cp_cal":          {"nrmse": 0.4649, "ncc": 0.9096},
  "amp_cp_corrupt":      {"nrmse": 0.4830, "ncc": 0.9007},
  "closure_only_cal":    {"nrmse": 0.7124, "ncc": 0.7455},
  "closure_only_corrupt":{"nrmse": 0.7959, "ncc": 0.6044}
}
```

**物理解读**：
- `vis_rml_cal`（NCC=0.97）：校准数据 + 可见度拟合 → 最佳重建
- `vis_rml_corrupt`（NCC≈0）：增益腐蚀完全摧毁了可见度拟合 → 重建失败
- `closure_only_corrupt`（NCC=0.60）：闭合量方法在同样腐蚀数据下仍能恢复结构 → **证明了闭合量方法的核心价值**

#### 6.2.5 测试分布（63 tests）

| 测试文件 | 测试数 | 考察内容 |
|:---------|:-------|:---------|
| `test_preprocessing.py` | 16 | 数据加载、三角形/四边形枚举、闭合量计算、噪声传播 |
| `test_physics_model.py` | 11 | DFT 正向/伴随、脏图/PSF、χ² 值及梯度正确性 |
| `test_solvers.py` | 7 | 正则化器值/梯度、L-BFGS-B 收敛性、重建质量 |
| `test_visualization.py` | 4 | 指标计算（NRMSE/NCC）、绘图函数不报错 |
| `test_end_to_end.py` | 7 | 验证 metrics.json 各方法结果、ground truth 属性 |
| `test_parity.py` | 18 | 与 ehtim 库的数值一致性：DFT矩阵、χ²、梯度、重建结果 |

**Parity 测试说明**：`test_parity.py` 使用预生成的 fixture 数据（存放在 `evaluation/fixtures/` 中），将参考实现的输出与 ehtim 库的输出进行逐值比对，确保 DFT 约定（+2πi）、像素网格、闭合量公式等与原始论文/库完全一致。

---

### 6.3 eht_black_hole_UQ：DPI 不确定性量化

#### 6.3.1 物理问题

传统成像方法（如 RML）只给出单点估计，无法量化重建的不确定性。DPI（Deep Probabilistic Imaging）使用 **Real-NVP Normalizing Flow** 学习图像的后验分布 p(x|d)，从而可以：
- 通过采样获得多个可能的重建图像
- 计算后验均值（posterior mean）和后验标准差（posterior std）
- 识别哪些区域的重建是可靠的、哪些区域不确定

训练不需要 ground truth 图像，只使用观测数据通过 KL 散度最小化。

#### 6.3.2 源代码模块（参考实现）

**`src/preprocessing.py`** — 数据加载（依赖 ehtim 库）
| 函数 | 功能 |
|:-----|:-----|
| `load_observation(data_dir)` | 通过 ehtim 加载 obs.uvfits → 提取 UV坐标、可见度、σ |
| `load_ground_truth(data_dir)` | 通过 ehtim 加载 gt.fits → 32×32 图像 |
| `extract_closure_indices(obs)` | 从 ehtim 观测对象提取三角形/四边形基线索引 |
| `compute_nufft_params(uv, npix, fov)` | 计算 NUFFT 的归一化 UV 坐标 |
| `build_prior_image(npix, fov, total_flux)` | 构建高斯先验图像 |
| `prepare_data(data_dir)` | 整合所有预处理步骤 |

**`src/physics_model.py`** — NUFFT 正向模型与损失函数
| 类 | 功能 |
|:---|:-----|
| `NUFFTOperator` | 使用 PyTorch 实现的 NUFFT（非均匀快速傅里叶变换） |
| `.forward(image)` | 图像 → 可见度 |
| `DPILoss` | DPI 训练损失函数 |
| `.closure_phase_chisq(vis)` | 闭合相位 χ² |
| `.log_closure_amp_chisq(vis)` | 对数闭合振幅 χ² |
| `.regularizers(image)` | 图像先验：MEM + TSV + L1 + flux + centering |
| `.forward(image, log_det)` | 总损失 = 数据项 + 正则化 - 熵（log_det） |

**`src/solvers.py`** — Real-NVP Normalizing Flow
| 类 | 功能 |
|:---|:-----|
| `ActNorm` | 激活归一化层（可学习缩放+平移） |
| `AffineCoupling` | 仿射耦合层（Real-NVP 核心组件） |
| `RealNVPBlock` | 单个 Real-NVP 块 = ActNorm + AffineCoupling |
| `RealNVP` | 完整流模型：多个 Block + 交替 mask |
| `.forward(z)` | 隐变量 z → 图像 x + log_det |
| `.inverse(x)` | 图像 x → 隐变量 z + log_det |
| `DPISolver` | 训练封装：Adam 优化、epoch 循环、学习率调度 |
| `.train(n_epochs)` | 训练流模型，返回 loss_history |
| `.sample(n_samples)` | 从学习的后验中采样图像 |

**`src/visualization.py`** — 后验可视化
| 函数 | 功能 |
|:-----|:-----|
| `compute_metrics(mean, gt)` | NRMSE + NCC |
| `plot_posterior_summary(mean, std, gt)` | 后验均值/标准差/真值对比图 |
| `plot_posterior_samples(samples)` | 后验采样展示 |
| `plot_loss_curve(loss_history)` | 训练损失曲线 |
| `plot_uncertainty_map(std)` | 不确定性热图 |

#### 6.3.3 管线流程（main.py）

```
1. prepare_data("data") → 观测数据、闭合索引、NUFFT参数、先验图像
2. DPISolver(flow_model, loss_fn, ...) → 构建求解器
3. solver.train(n_epochs=30000) → 训练 Normalizing Flow（~数小时）
4. solver.sample(1024) → 从后验采样 1024 张图像
5. posterior_mean = samples.mean(axis=0) → 后验均值
6. posterior_std = samples.std(axis=0) → 后验标准差
7. compute_metrics(posterior_mean, gt) → NRMSE/NCC
8. 保存: output/reconstruction.npy (= posterior_mean)
         output/posterior_samples_1024.npy
         output/posterior_std.npy
         output/metrics.json
```

#### 6.3.4 参考指标（metrics.json）

```json
{
  "posterior_mean": {"nrmse": 0.3991, "ncc": 0.9165}
}
```

参考实现 NCC=0.9165，NRMSE=0.3991。训练 30000 epochs 后后验均值与 ground truth 高度一致。

#### 6.3.5 测试分布（86 tests）

| 测试文件 | 测试数 | 考察内容 |
|:---------|:-------|:---------|
| `test_preprocessing.py` | 23 | ehtim 数据加载、闭合索引提取、NUFFT参数、先验图像 |
| `test_physics_model.py` | 13 | NUFFT 正向模型、DPILoss 各项（χ²、正则化、总损失） |
| `test_solvers.py` | 13 | ActNorm/AffineCoupling/RealNVP 架构、可逆性、训练收敛 |
| `test_visualization.py` | 5 | 指标计算、各类绘图函数 |
| `test_end_to_end.py` | 8 | 缩减训练（200 epochs）验证：损失下降、后验形状、正性约束 |
| `test_parity.py` | 24 | 与原始 DPI 代码的数值一致性：NUFFT、χ²、流模型、采样 |

**Parity 测试说明**：`test_parity.py` 使用 `evaluation/fixtures/` 中的预生成数据，将参考实现与原始 DPI 论文的代码（`evaluation/reference_code/`）进行逐值比对，确保 NUFFT 约定、闭合量计算、流模型架构等与原始实现数值一致。

## 七、框架审计与修复

在对框架进行全面代码审读和实际运行过程中，发现并修复了以下问题：

### 7.1 代码归档缺失（已修复）

**问题**：`LocalRunner.stop()` 在评估结束后直接删除临时目录 `/tmp/imaging101-local-*`，导致 Agent 生成的代码永久丢失，无法事后分析模型的实现错误。

**修复**：在 `local_runner.py` 的 `stop()` 方法中增加归档逻辑，在删除临时目录前将 `src/`、`plan/`、`output/` 复制到持久化目录 `/data/yjh/function_eval_code_archive/<timestamp>_<task_name>/`。

### 7.2 API 提示过长导致 400 错误（已识别）

**问题**：`physics_model` 模块的 Function Mode 评估中，系统提示包含完整的 README（含论文摘要）、所有 fixture 描述、依赖模块代码，总 token 数可能超过 API 网关的限制（~100K tokens），导致 HTTP 400 Bad Request。

**状态**：已识别但未在框架层面修复。建议在 `runner.py` 中增加 prompt 大小检测和分阶段发送逻辑。

### 7.3 pytest 输出截断影响评分解析（已识别）

**问题**：`agent.py` 对 RUN 命令输出有 8,000 字符截断限制。当测试数量较多时（如 UQ 任务的 24 个 parity 测试），pytest `-v` 输出可能被截断，导致 `scorer.py` 的正则解析遗漏部分测试结果。`scorer.py` 已有 fallback 逻辑（从逐条明细中统计），但在极端截断情况下仍可能不准确。

**影响**：评分准确性。当前 fallback 机制在大多数情况下可以正确恢复计数。

### 7.4 滑动窗口丢弃关键上下文（已识别）

**问题**：`_apply_sliding_window()` 在 `max_context_messages=10` 时会丢弃中间消息，仅保留首条 system + initial user + 最近 8 条。对于需要大量迭代的模块（如 preprocessing 经历 10 次迭代才通过全部测试），早期的错误分析和修复经验会被丢弃，可能导致 Agent 重复犯错。

**影响**：Agent 效率。对评分本身无影响。

### 7.5 End-to-End 模式的 ground truth 注入路径硬编码（已识别）

**问题**：`scorer.py::_compute_quality_metrics()` 硬编码了 ground truth 路径为 `evaluation/reference_outputs/ground_truth.npy`。如果未来有任务使用不同的文件名或路径，需要修改评分代码。

**状态**：当前两个任务均符合此约定，暂不需修改。

### 7.6 Docker 与 Local Runner 行为差异（已识别）

**问题**：两种 Runner 在以下方面存在细微差异：
- **环境变量**：DockerRunner 继承容器默认环境，LocalRunner 使用严格白名单
- **路径映射**：LocalRunner 需要将 `/workspace/` 替换为实际临时目录路径，DockerRunner 不需要
- **超时机制**：DockerRunner 使用 `timeout` 命令，LocalRunner 使用 `subprocess.run(timeout=)`

**影响**：可能导致同一 Agent 代码在两种环境下行为不同。建议统一环境变量策略。

### 7.7 二进制文件保护仅限 LocalRunner（已识别）

**问题**：`LocalRunner.read_file()` 对 `.npz/.npy/.pkl/.fits` 等二进制扩展名返回提示信息而非原始字节。但 `DockerRunner.read_file()` 没有对应的保护逻辑，可能向 Agent 返回乱码数据。

**状态**：当前默认使用 LocalRunner，影响有限。

### 7.8 伪造 DONE 检测的边界情况（已识别）

**问题**：`_parse_all_actions()` 通过检查 `Observation:` 关键字来识别伪造的多轮对话。如果 Agent 在正常文本中恰好写了 "Observation:" 字样（如在注释或文档中），可能触发误判。

**影响**：低概率事件。当前检测逻辑已足够稳健

## 八、评估基准线验证

参考实现（`src/` 目录中的代码）必须通过全部测试，以确认测试本身的正确性和参考实现的可靠性。

### 8.1 eht_black_hole_original：63/63 通过

```
test_preprocessing.py    16/16  ✅
test_physics_model.py    11/11  ✅
test_solvers.py           7/7   ✅
test_visualization.py     4/4   ✅
test_end_to_end.py        7/7   ✅
test_parity.py           18/18  ✅
────────────────────────────────
Total                    63/63  ✅  (100%)
```

**Parity 基准**：18 个 parity 测试验证参考实现与 ehtim 库的数值一致性，最大相对误差阈值通常为 `rtol=1e-4`（DFT 矩阵）到 `rtol=1e-2`（重建结果，因迭代优化的浮点累积误差）。

### 8.2 eht_black_hole_UQ：86/86 通过

```
test_preprocessing.py    23/23  ✅
test_physics_model.py    13/13  ✅
test_solvers.py          13/13  ✅
test_visualization.py     5/5   ✅
test_end_to_end.py        8/8   ✅
test_parity.py           24/24  ✅
────────────────────────────────
Total                    86/86  ✅  (100%)
```

**Parity 基准**：24 个 parity 测试与原始 DPI 代码（`evaluation/reference_code/`）对比。由于 PyTorch 浮点运算的非确定性，某些测试使用较宽松的阈值（`atol=1e-3`）。

### 8.3 Agent 评估基准线（Claude-4.6-opus, Function Mode）

以下是使用 Claude-4.6-opus 模型在 Function Mode 下的评估结果（来自 `results/` 目录和 `CODE_COMPARISON_REPORT.md`）：

**eht_black_hole_original 任务**：

| 目标模块 | 通过率 | 迭代次数 | 关键发现 |
|:---------|:-------|:---------|:---------|
| `preprocessing` | 16/16 (100%) | 10 | 模型经 10 次迭代自纠正，用 dict 查找替代线性搜索 |
| `physics_model` | 11/11 (100%) | — | 正确实现 +2πi DFT 约定和三角脉冲响应 |
| `solvers` | 4/7 (57%) | — | 梯度形状错误（1D vs 2D）、TV 边界条件错误（zero-pad vs np.roll） |
| `visualization` | 4/4 (100%) | — | 正确使用 raw NCC（非 centered NCC） |

**关键失败分析**（solvers 模块 3 个失败）：
1. **梯度形状不匹配**：正则化器的 `value_and_grad()` 返回 1D 扁平梯度，但测试期望与输入图像同形状（2D）。公式本身正确，仅输出格式错误。
2. **TV 正则化器边界条件**：模型使用零填充（zero-padding），参考实现使用周期边界（`np.roll`）。这是一个物理约定差异，不影响数值量级但影响边界像素梯度的精确值。

### 8.4 批量评估脚本

`run_function_evals.sh` 自动化执行全部模块的 Function Mode 评估：

```bash
# 对 eht_black_hole_original 的 4 个模块依次运行
for module in preprocessing physics_model solvers visualization; do
    python -m evaluation_harness run \
        --task eht_black_hole_original \
        --mode function \
        --target-function $module \
        --model cds/Claude-4.6-opus \
        --max-iterations 20 --timeout 600
done
```

`overnight/run_overnight.sh` 用于多任务、多 prompt 的大规模过夜评估

## 附录：三种模式对比总结

| 维度 | Plan Mode | Function Mode | End-to-End Mode |
|:-----|:----------|:--------------|:----------------|
| **难度** | ★☆☆☆☆ | ★★★☆☆ | ★★★★★ |
| **Agent 可见** | README, data/ | README, data/, plan/, evaluation/ | README, data/ |
| **Agent 不可见** | plan/, src/, evaluation/ | src/ (目标模块) | plan/, src/, evaluation/ |
| **要求输出** | plan/approach.md, plan/design.md | src/\<module\>.py | plan/, src/, main.py, output/reconstruction.npy |
| **系统提示** | SYSTEM_PROMPT | SYSTEM_PROMPT_FUNCTION | SYSTEM_PROMPT |
| **DONE 门控** | 无 | 必须运行过 pytest | 必须运行过 main.py |
| **评分方法** | LLM-as-Judge (Pairwise + Rubric) | pytest 通过率 | NRMSE / NCC 质量指标 |
| **评分代码** | plan_scorer.py | scorer.py::_run_tests() | scorer.py::_compute_quality_metrics() |
| **评分范围** | overall_score ∈ [0, 1] | test_pass_rate ∈ [0, 1] | nrmse ∈ [0, ∞), ncc ∈ [-1, 1] |
| **是否运行 pytest** | ❌ | ✅ | ❌ |
| **是否需要 ground truth** | ❌ | ❌ (测试自带 fixture) | ✅ (scorer 自动注入) |
| **两阶段执行** | ✅ (approach → design) | ❌ (单阶段) | ✅ (planning → implementation) |
| **依赖模块预植** | N/A | ✅ (_seed_dependency_modules) | ❌ (Agent 实现全部) |

---

*文档结束。本文档基于 `evaluation_harness/` 全部源码、两个任务的完整实现代码和测试套件、以及实际运行结果编写，力求每个描述均可追溯到具体代码行。*
