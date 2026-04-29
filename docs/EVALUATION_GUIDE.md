# Evaluation Guide

How to evaluate an LLM agent on imaging-101 benchmark tasks.

---

## Prerequisites

```bash
git clone https://github.com/HeSunPU/imaging-101.git
cd imaging-101
pip install -r evaluation_harness/requirements.txt
```

You need an **OpenAI-compatible API endpoint**. Set your credentials:

```bash
export API_KEY="your-api-key"
export BASE_URL="https://api.openai.com/v1"   # or your gateway
export MODEL="gpt-4o"                          # your model identifier
```

Or configure `config_llm.yaml`:

```yaml
"your-model-name":
    api_type: "openai"
    base_url: "https://your-api-gateway/v1"
    api_key: "your-api-key"
    temperature: 0.2
```

---

## 1. Function-Mode Evaluation

Tests whether an agent can implement **individual modules** (e.g., `physics_model.py`, `preprocessing.py`, `solvers.py`) given the task description and plan. Each module is evaluated against unit tests.

### Evaluate a single module

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode function \
    --target-function physics_model \
    --model $MODEL \
    --base-url $BASE_URL \
    --api-key $API_KEY \
    --framework react \
    --max-iterations 20 \
    --timeout 600 \
    --output results/function_mode \
    -v
```

**What happens:**
1. The agent reads the task README, plan docs, and the target module's test file
2. It iteratively writes code, runs tests, and fixes errors
3. Results are saved to `results/function_mode/<task>/<model>_<date>/<module>/`

### Evaluate all modules of a task

Run each testable module separately:

```bash
for module in physics_model preprocessing solvers; do
    python -m evaluation_harness run \
        --task ct_sparse_view \
        --mode function \
        --target-function $module \
        --model $MODEL \
        --base-url $BASE_URL \
        --api-key $API_KEY \
        --framework react \
        --max-iterations 20 \
        --timeout 600 \
        --output results/function_mode \
        -v
done
```

### Generate summary

```bash
python -m evaluation_harness summarize \
    --dir results/function_mode/ct_sparse_view/your-model_20260401
```

This creates `summary.json` with aggregate test pass rates.

### Re-score without LLM calls

If you want to re-run pytest on previously generated code (e.g., after fixing environment issues):

```bash
python scripts/rescore_existing.py
```

### Output structure

```
results/function_mode/<task>/<model>_<date>/
├── physics_model/
│   ├── result.json              # {"tests_passed": 11, "tests_total": 11, ...}
│   └── src/physics_model.py     # Generated code
├── preprocessing/
│   ├── result.json
│   └── src/preprocessing.py
├── solvers/
│   ��── result.json
│   └── src/solvers.py
└── summary.json                 # Aggregated metrics
```

### Metrics

- **Test pass rate** per module: `tests_passed / tests_total`
- **Aggregate pass rate** across all modules

---

## 2. End-to-End Evaluation

Tests whether an agent can **implement the full imaging pipeline** from scratch, producing a reconstruction from raw data.

### Run end-to-end with the built-in agent

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --model $MODEL \
    --base-url $BASE_URL \
    --api-key $API_KEY \
    --framework react \
    --level L1 \
    --max-iterations 50 \
    --timeout 1200 \
    --output results/end_to_end \
    -v
```

**Difficulty levels:**

| Level | Agent receives | Difficulty |
|-------|---------------|------------|
| `L1`  | Task README only | Hardest — agent plans from scratch |
| `L2`  | README + `approach.md` | Medium — approach is given |
| `L3`  | README + `approach.md` + `design.md` | Easiest — full design is given |

### Run with multi-agent pipeline

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --framework multi_agent \
    --level L1 \
    --model $MODEL \
    --base-url $BASE_URL \
    --api-key $API_KEY \
    --max-iterations 50 \
    --output results/end_to_end/multi_agent \
    -v
```

### Evaluate with a third-party agent (Claude Code, Cursor, etc.)

For agents that run outside this harness:

```bash
# Step 1: Prepare sandbox workspace
python -m evaluation_harness prepare \
    --task eht_black_hole_original \
    --level L1

# Step 2: Run your agent in the prepared workspace
# The workspace contains a .prompt.md file — paste it into your agent.
# The agent should produce output/reconstruction.npy

# Step 3: Collect and score results
python -m evaluation_harness collect \
    --task eht_black_hole_original \
    --workspace-dir ~/copilot_workspaces/eht_black_hole_original_L1/ \
    --agent-name claude_code \
    --output results/end_to_end/copilot
```

### Compute reconstruction quality metrics

After `output/reconstruction.npy` is produced:

```bash
# From a workspace directory
python scripts/compute_metrics.py \
    --workspace /path/to/workspace \
    --task eht_black_hole_original

# From explicit files
python scripts/compute_metrics.py \
    --reconstruction output/reconstruction.npy \
    --ground-truth tasks/eht_black_hole_original/evaluation/reference_outputs/ground_truth.npy

# Compare against reference methods
python scripts/compute_metrics.py \
    --workspace /path/to/workspace \
    --task eht_black_hole_original \
    --compare-reference --json
```

### Metrics

- **NCC** (Normalized Cross-Correlation): higher is better, 1.0 = perfect match
- **NRMSE** (Normalized Root Mean Square Error): lower is better, 0.0 = perfect match

---

## 3. Plan-Mode Evaluation

Tests an agent's **planning ability** without code implementation.

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode plan \
    --model $MODEL \
    --base-url $BASE_URL \
    --api-key $API_KEY \
    --framework react \
    --output results/plan \
    -v
```

The agent's plan is evaluated against the reference `approach.md` and `design.md` using LLM-as-judge pairwise comparison and rubric scoring.

---

## 4. Batch Evaluation

### All tasks for one model (function mode)

```bash
# Edit API_KEY/BASE_URL/MODEL in the script, then run:
bash scripts/run_gemini_remaining.sh    # Gemini
bash scripts/run_server_claude.sh       # Claude
bash scripts/run_server_gpt.sh          # GPT
```

### Custom batch script

```bash
TASKS=$(ls tasks/)
for task in $TASKS; do
    # Skip tasks without tests
    if [ ! -d "tasks/$task/evaluation/tests" ]; then
        echo "Skipping $task (no tests)"
        continue
    fi
    
    for module in physics_model preprocessing solvers; do
        test_file="tasks/$task/evaluation/tests/test_${module}.py"
        if [ -f "$test_file" ]; then
            python -m evaluation_harness run \
                --task $task \
                --mode function \
                --target-function $module \
                --model $MODEL \
                --base-url $BASE_URL \
                --api-key $API_KEY \
                --framework react \
                --max-iterations 20 \
                --timeout 600 \
                --output results/function_mode \
                -v
        fi
    done
done
```

---

## 5. Interpreting Results

### Function-mode result.json

```json
{
    "task_name": "eht_black_hole_original",
    "mode": "function",
    "model": "gpt-4o",
    "framework": "react",
    "target_function": "physics_model",
    "tests_passed": 11,
    "tests_total": 11,
    "test_pass_rate": 1.0,
    "iterations": 5,
    "wall_time_seconds": 45.2,
    "stopped_reason": "done"
}
```

### End-to-end result.json

```json
{
    "task_name": "eht_black_hole_original",
    "mode": "end_to_end",
    "level": "L1",
    "model": "gpt-4o",
    "ncc": 0.869,
    "nrmse": 0.584,
    "iterations": 32,
    "files_created": ["src/physics_model.py", "src/preprocessing.py", "src/solvers.py", "main.py"]
}
```

### Quality thresholds

Each task defines baseline metrics in `evaluation/metrics.json` (or `evaluation/reference_outputs/metrics.json`). A reconstruction is considered successful when:

- **NCC** exceeds the baseline method's NCC
- **NRMSE** is below the baseline method's NRMSE

---

## Quick Reference

| What you want | Command |
|--------------|---------|
| Evaluate one module | `python -m evaluation_harness run --task TASK --mode function --target-function MODULE ...` |
| Evaluate full pipeline | `python -m evaluation_harness run --task TASK --mode end_to_end --level L1 ...` |
| Prepare for external agent | `python -m evaluation_harness prepare --task TASK --level L1` |
| Score external agent | `python -m evaluation_harness collect --task TASK --workspace-dir DIR --agent-name NAME` |
| Compute NCC/NRMSE | `python scripts/compute_metrics.py --workspace DIR --task TASK` |
| Summarize function results | `python -m evaluation_harness summarize --dir results/function_mode/TASK/MODEL_DATE` |
| Re-score existing code | `python scripts/rescore_existing.py` |
