# Evaluation Harness

A minimal ReAct coding agent for benchmarking LLMs on computational imaging tasks.

## Quick Start

```bash
# 1. Install harness dependencies
pip install -r evaluation_harness/requirements.txt

# 2. Build the Docker sandbox image
docker build -t imaging101-sandbox -f evaluation_harness/Dockerfile .

# 3. Run an evaluation
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --model gpt-4o \
    --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY
```

## Evaluation Modes

| Mode | Command | What the agent does |
|------|---------|---------------------|
| **plan** | `--mode plan` | Generates `plan/approach.md` + `plan/design.md` from README + data |
| **function** | `--mode function --target-function preprocessing.load_observation` | Implements a single function, tested by unit tests |
| **end_to_end** | `--mode end_to_end` | Plans + implements the full `src/` pipeline + `main.py` |

## Architecture

```
__main__.py     CLI entry point (argparse)
config.py       LLMConfig / TaskConfig / RunConfig dataclasses
llm_client.py   OpenAI-compatible API calls (raw requests)
docker_runner.py  Persistent Docker container sandbox
prompts.py      Prompt templates per evaluation mode
agent.py        ReAct loop: Thought → Action → Observation
scorer.py       pytest runner + quality metrics + JSON output
runner.py       Top-level orchestrator
Dockerfile      Sandbox image (Python 3.11 + numpy/scipy/matplotlib/pytest)
```

## Output

Results are saved as JSON in `results/`:
```json
{
  "task_name": "eht_black_hole_original",
  "mode": "end_to_end",
  "model": "gpt-4o",
  "tests_passed": 22,
  "tests_total": 25,
  "test_pass_rate": 0.88,
  "quality_metrics": {"nrmse": 0.61, "ncc": 0.82},
  "total_tokens": 57000,
  "wall_time_seconds": 320.5,
  "iterations": 15
}
```
