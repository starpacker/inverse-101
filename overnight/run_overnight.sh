#!/bin/bash
# =============================================================================
# Overnight Claude Code Runner for imaging-101 Task Cleaning
# =============================================================================
# Usage:
#   nohup bash overnight/run_overnight.sh > overnight/overnight.log 2>&1 &
#
# To monitor:
#   tail -f overnight/overnight.log
#
# To stop:
#   kill $(cat overnight/overnight.pid)
# =============================================================================

set -uo pipefail
# NOTE: no -e so that one task failure doesn't stop the rest

REPO_DIR="/home/groot/Documents/PKUlab/imaging-101"
PROMPT_DIR="$REPO_DIR/overnight/prompts"
LOG_DIR="$REPO_DIR/overnight/logs"
BRANCH="main"

# Save PID for easy kill
echo $$ > "$REPO_DIR/overnight/overnight.pid"

# Create log directory
mkdir -p "$LOG_DIR"

# Timestamp helper
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "============================================="
echo "$(ts) Overnight run started"
echo "============================================="

# -----------------------------------------------------------------------------
# Common preamble injected into every prompt
# -----------------------------------------------------------------------------
read -r -d '' PREAMBLE << 'PREAMBLE_EOF' || true
You are working in the imaging-101 benchmark repository at /home/groot/Documents/PKUlab/imaging-101.

IMPORTANT CONTEXT:
- There is a completed pilot task at tasks/eht_black_hole/ — use it as the FORMAT REFERENCE for directory structure, code style, test structure, and fixture format.
- Read the pilot task's files (README.md, plan/, src/, evaluation/) to understand the expected format.
- Read the CLAUDE.md at the repo root for project overview.

IMPORTANT RULES:
1. Each task must be SELF-CONTAINED — no cross-task imports, no dependency on ehtim/DPI/bhnerf packages.
2. Follow the EXACT directory structure of the pilot task.
3. For each src/*.py function, create a corresponding test fixture in evaluation/fixtures/ and test in evaluation/tests/.
4. Test fixtures: use param_*, input_*, config_*, output_* naming convention in NPZ files.
5. Tests for stochastic functions: test statistical properties, NOT exact values from fixed seeds.
6. Generate synthetic data if real data is not available — results should be qualitatively similar to the paper.
7. Keep existing files (PDFs, reference_website_github.md) in the task folder — do not delete them.
8. Run `cd <task_dir> && python -m pytest evaluation/tests/ -v` at the end and fix failures.
9. Run `cd <task_dir> && python main.py` to verify the pipeline works end-to-end.
10. When cloning reference repos, clone to /tmp/ and read code there. Do NOT leave cloned repos in the task directory.

PREAMBLE_EOF

# -----------------------------------------------------------------------------
# Task runner function
# -----------------------------------------------------------------------------
run_task() {
    local task_num="$1"
    local task_name="$2"
    local prompt_file="$3"
    local max_budget="${4:-50.00}"

    echo ""
    echo "============================================="
    echo "$(ts) TASK $task_num: $task_name"
    echo "============================================="

    local task_prompt
    task_prompt=$(cat "$prompt_file")

    local full_prompt="$PREAMBLE

--- TASK INSTRUCTIONS ---

$task_prompt"

    # Run Claude Code in headless mode
    cd "$REPO_DIR"

    echo "$(ts) Starting claude -p for $task_name (budget=\$$max_budget)..."

    if claude -p "$full_prompt" \
        --allowedTools "Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch,Agent" \
        --max-budget-usd "$max_budget" \
        --verbose \
        < /dev/null \
        > "$LOG_DIR/${task_num}_${task_name}.log" 2>&1; then
        echo "$(ts) Claude completed $task_name successfully"
    else
        echo "$(ts) WARNING: Claude exited with error for $task_name (exit code: $?)"
    fi

    # Auto-commit and push changes
    cd "$REPO_DIR"
    if [ -n "$(git status --porcelain)" ]; then
        echo "$(ts) Committing changes for $task_name..."
        git add -A
        git commit -m "Clean task: $task_name

Automated overnight task cleaning by Claude Code.
Restructured from reference paper and code into standardized benchmark format.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>" || true

        echo "$(ts) Pushing to origin/$BRANCH..."
        git push origin "$BRANCH" || echo "$(ts) WARNING: Push failed for $task_name"
    else
        echo "$(ts) No changes to commit for $task_name"
    fi

    echo "$(ts) TASK $task_num: $task_name DONE"
    echo ""
}

# -----------------------------------------------------------------------------
# Run all 5 tasks sequentially
# -----------------------------------------------------------------------------

# Task 1: Static reconstruction with closure quantities (CPU-only)
run_task "01" "eht_black_hole_original" \
    "$PROMPT_DIR/01_eht_black_hole_original.md" 50.00

# Task 2: Dynamic reconstruction StarWarps (CPU-only) — SKIPPED
# run_task "02" "eht_black_hole_dynamic" \
#     "$PROMPT_DIR/02_eht_black_hole_dynamic.md" 50.00

# Task 3: Deep Probabilistic Imaging (GPU)
run_task "03" "eht_black_hole_UQ" \
    "$PROMPT_DIR/03_eht_black_hole_UQ.md" 50.00

# Task 4: α-DPI Feature Extraction (GPU) — SKIPPED
# run_task "04" "eht_black_hole_feature_extraction" \
#     "$PROMPT_DIR/04_eht_black_hole_feature_extraction.md" 50.00

# Task 5: BH-NeRF Tomography (GPU, most complex) — SKIPPED
# run_task "05" "eht_black_hole_tomography" \
#     "$PROMPT_DIR/05_eht_black_hole_tomography.md" 50.00

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================="
echo "$(ts) ALL TASKS COMPLETED"
echo "============================================="
echo ""
echo "Task logs:"
ls -la "$LOG_DIR/"
echo ""
echo "Git log:"
git -C "$REPO_DIR" log --oneline -10

# Clean up PID file
rm -f "$REPO_DIR/overnight/overnight.pid"
