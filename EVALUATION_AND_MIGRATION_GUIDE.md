# Imaging-101 测评与任务迁移指南

本文档将指导您如何使用 `evaluation_harness` 进行 Plan、Function 和 End-to-End 测评，并说明如何添加新的成像任务。

## 1. 环境准备

在开始测评之前，请确保已安装依赖并构建了 Docker 沙箱环境（测评在沙箱中运行以确保安全和隔离）。

```bash
# 1. 安装 harness 依赖
pip install -r evaluation_harness/requirements.txt

# 2. 构建 Docker 沙箱镜像
docker build -t imaging101-sandbox -f evaluation_harness/Dockerfile .
```

## 2. 如何进行测评 (Evaluation)

测评脚本通过 `python -m evaluation_harness run` 启动。主要有三种模式：

### 2.1 Plan 模式 (计划生成)

**目标**：评估 Agent 理解问题并制定解决方案的能力。
**输入**：`README.md` + `data/meta_data`
**输出**：`plan/approach.md` (方法论) 和 `plan/design.md` (代码设计)

**命令示例**：
```bash
python -m evaluation_harness run \
    --task eht_black_hole \
    --mode plan \
    --model gpt-4o \
    --api-key $OPENAI_API_KEY
```

### 2.2 Function 模式 (单函数实现)

**目标**：评估 Agent 实现特定功能函数的能力，通过单元测试验证。
**输入**：`README.md` + `plan/`
**输出**：单个函数的实现代码

**命令示例 (单个模块)**：
需要指定 `--target-function`，格式为 `模块名` (无需 .py)。

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode function \
    --target-function preprocessing \
    --model cds/Claude-4.6-opus \
    --base-url https://ai-gateway-internal.dp.tech/v1 \
    --api-key $OPENAI_API_KEY
```

**命令示例 (批量测试所有模块)**：
如果您想一次性测试任务下的所有核心模块，可以使用如下 Shell 脚本：

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

### 2.3 End-to-End 模式 (全流程)

**目标**：评估 Agent 从零构建完整成像管线的能力。
**输入**：`README.md` + `data/`
**输出**：完整的 `src/` 目录和 `main.py`
**评估标准**：生成的输出与 `evaluation/reference_outputs/` 中的参考结果进行对比。

**命令示例**：
```bash
python -m evaluation_harness run \
    --task eht_black_hole \
    --mode end_to_end \
    --model gpt-4o \
    --api-key $OPENAI_API_KEY
```

---

## 3. 如何迁移到新任务 (Task Migration)

如果您想要添加一个新的成像任务（例如 MRI 重建、地震成像等），需要按照标准化的目录结构准备以下内容。

### 3.1 目录结构准备

在 `tasks/` 目录下创建一个新文件夹（例如 `tasks/my_new_task/`），并包含以下结构：

| 文件/目录 | 描述 |
|-----------|------|
| **`README.md`** | **核心文档**。详细描述物理问题、数据格式、输入输出要求和方法提示。 |
| **`requirements.txt`** | 任务所需的 Python 依赖库（沙箱环境中安装）。 |
| **`main.py`** | 标准化的入口脚本，用于运行整个重建管线。 |
| **`data/`** | 包含 `raw_data.npz` (观测数据) 和 `meta_data` (参数 JSON)。 |
| **`plan/`** | (可选/参考) 包含 `approach.md` 和 `design.md` 作为参考答案。 |
| **`src/`** | 参考实现代码，需拆分为标准模块：<br>- `preprocessing.py` (数据预处理)<br>- `physics_model.py` (正向模型)<br>- `solvers.py` (反演求解器)<br>- `visualization.py` (可视化) |
| **`evaluation/`** | **测评核心**。包含参考输出和测试用例。 |

### 3.2 准备测评内容 (Evaluation Assets)

为了让 Harness 能测评您的新任务，必须准备 `evaluation/` 目录下的内容：

1.  **参考输出 (`evaluation/reference_outputs/`)**
    *   运行您的参考代码（Reference Code），保存关键结果。
    *   需要包含：模型权重 (Checkpoints)、重建图像/结果、Loss 历史、以及 `metrics.json` (包含定量指标如 PSNR, SSIM 等)。
    *   这些将作为 End-to-End 测评的 "标准答案" (Ground Truth)。

2.  **测试固件 (`evaluation/fixtures/`)**
    *   为每个核心函数准备输入输出数据对。
    *   命名规范：`input_*.pkl`, `output_*.pkl`, `config_*.pkl`。
    *   用途：Function 模式下，Harness 会加载这些数据来测试 Agent 生成的函数。

3.  **测试脚本 (`evaluation/tests/`)**
    *   编写 `pytest` 测试脚本。
    *   **单元测试**：加载 fixtures，验证函数输出是否正确。
    *   **Parity Tests (等价性测试)**：验证 Agent 生成的代码是否与原始参考代码在数值上一致（对于确定性算法，通常要求 `rtol=1e-10`）。

### 3.3 任务清理与验证流程 (Task Cleaning Workflow)

迁移现有研究代码时的推荐流程：

1.  **运行原始代码**：确保能跑通并生成结果。
2.  **定义 Parity Tests**：编写测试脚本，对比原始代码和您即将重构的代码。
3.  **代码重构 (Cleaning)**：将代码拆分到 `src/` 的标准结构中 (`preprocessing`, `physics_model` 等)。每改一步都运行 Parity Tests 确保功能未变。
4.  **生成参考输出**：使用重构后的代码生成最终的 `evaluation/reference_outputs/`。
5.  **验证**：使用 `python -m evaluation_harness run --task my_new_task ...` 尝试对自己构建的任务进行测评，确保 Harness 能正确运行。

---

### 总结

*   **测评**：使用 `python -m evaluation_harness run` 配合不同的 `--mode`。
*   **迁移**：遵循 `tasks/` 目录下的标准模板，核心是准备好 `README.md` (题目) 和 `evaluation/` (答案与测试标准)。

您可以参考 `tasks/eht_black_hole/` 作为一个完美的样板任务来构建您的新任务。
