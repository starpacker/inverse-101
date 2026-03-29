#!/bin/bash
# Run function-level evaluation specifically for the preprocessing module
# with logging enabled.

set -uo pipefail

# Configuration
API_KEY="sk-Zj3a7RQDVCXr-Axg-0gtkg"
BASE_URL="https://ai-gateway-internal.dp.tech/v1"
MODEL="gemini-3-pro-preview"
TASK="eht_black_hole_original"
MODE="function"
TARGET_FUNCTION="preprocessing"
MAX_ITER=20
TIMEOUT=600
LOG_FILE="logs_interaction_preprocessing_gemini.md"

# Ensure we are in the repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "============================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting function evaluation for: $TARGET_FUNCTION"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Log File: $LOG_FILE"
echo "============================================="

python -m evaluation_harness run \
    --task "$TASK" \
    --mode "$MODE" \
    --target-function "$TARGET_FUNCTION" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --max-iterations "$MAX_ITER" \
    --timeout "$TIMEOUT" \
    --output results \
    --log-file "$LOG_FILE" \
    -v

echo ""
echo "============================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') Evaluation complete."
echo "Interaction log saved to: $LOG_FILE"
