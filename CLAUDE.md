# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**imaging-101** is a benchmark suite for evaluating coding agents on computational imaging tasks. It covers 8 domains (physics, chemistry, biology, medicine, astronomy, earth science, mechanics, industrial/commercial) with standardized task structure, reference implementations, and automated evaluation.

## Repository Structure

```
imaging-101/
├── skills/                    # Shared skill library (available to agents)
├── evaluation_harness/        # Benchmark evaluation framework
└── tasks/<task_name>/         # Individual imaging tasks
```

### Task Template

Every task follows this standardized layout:

```
tasks/<task_name>/
├── README.md                  # Problem definition: physics, data description, method hints
├── requirements.txt           # Sandbox environment dependencies
├── main.py                    # Pipeline entry point
├── data/
│   ├── raw_data.npz           # Observation data (format varies per task)
│   └── meta_data              # JSON: imaging parameters for forward modeling
├── plan/
│   ├── approach.md            # Solution methodology and algorithm description
│   └── design.md              # Code architecture with function signatures
├── src/
│   ├── preprocessing.py       # Raw data → processed observations (y)
│   ├── physics_model.py       # Forward model: x → y
│   ├── solvers.py             # Inverse solvers: y → x_hat
│   ├── visualization.py       # Plotting utilities and metrics
│   └── generate_data.py       # Optional: synthetic data generation
├── notebooks/
│   └── <task_name>.ipynb      # **Required** end-to-end tutorial notebook (user review interface)
└── evaluation/
    ├── reference_outputs/     # Pretrained checkpoints, samples, ground truth, metrics.json
    ├── fixtures/              # Per-function test fixtures
    └── tests/                 # Unit tests, parity tests, integration tests
```

## Evaluation Modes

| Mode              | Agent sees                         | Agent produces      | Evaluated against                |
|-------------------|------------------------------------|---------------------|----------------------------------|
| Plan generation   | README.md + data/meta_data         | plan/               | plan/approach.md + plan/design.md|
| Function-level    | README.md + plan/                  | Single function     | Unit tests + numerical precision |
| End-to-end        | README.md + data/                  | src/ + main.py      | evaluation/reference_outputs/    |
| + Skills          | Same + skills/                     | Same                | Same                             |

## Commands

```bash
cd tasks/<task_name>
pip install -r requirements.txt          # Install dependencies
python main.py                           # Run reconstruction pipeline
python -m pytest evaluation/tests/ -v    # Run all tests
```

## Task Cleaning Principles

Tasks are created by **cleaning** existing research code from published papers and their reference implementations. The goal is to restructure real computational imaging code into the standardized benchmark format.

### Code cleaning workflow

Cleaning follows a **test-driven** process. Never rewrite code first and hope it works — define success criteria from the original code, then clean.

1. **Run the original code** — execute the reference implementation on the task's dataset end-to-end. Save all intermediate and final outputs (preprocessed data, model checkpoints, loss history, reconstructed images/samples).
2. **Define parity tests** — write `test_parity.py` that compares original vs cleaned code at every pipeline stage: preprocessing, forward model, loss functions, solver/architecture, and full-pipeline outputs. These tests guarantee the cleaned code is functionally identical to the original.
3. **Clean the code** — restructure into our `src/` layout. Run parity tests after each change; fix any discrepancies immediately. Do not proceed to the next stage until all parity tests pass.
4. **Generate reference outputs** — from the trained original model, produce reference checkpoints (model weights, loss history), posterior samples/statistics, ground truth, and `metrics.json` for `evaluation/reference_outputs/`.
5. **Write unit tests and fixtures** — per-function tests using `evaluation/fixtures/`.
6. **Write the notebook** (`notebooks/<task_name>.ipynb`) — this is the **primary user-facing deliverable** and must always be generated as the final step of cleaning. The notebook serves as the interface for users to review, validate, and critique the implementation. Requirements:
   - Loads precomputed results from `evaluation/reference_outputs/` so it **runs in seconds** without heavy computation.
   - Contains all key visualizations: ground truth, reconstructions, comparison figures, metrics tables/charts.
   - Includes **commented-out code** for running the full pipeline from scratch (so users can reproduce).
   - Uses the cleaned `src/` modules (not the original library) to demonstrate the implementation works.
   - Should be self-contained and tell a clear story: problem → data → method → results → conclusion.
   - **Verify the notebook executes without errors** by running `jupyter nbconvert --execute` before finishing.
   - **Never call `matplotlib.use('Agg')` in `src/` modules** — it locks the backend globally and prevents `%matplotlib inline` from working in notebooks. Only set the backend in `main.py` (which runs headless). Library code in `src/` should just `import matplotlib.pyplot as plt` without forcing a backend.

### Completion checklist

Before considering a task "done", verify every item:

- [ ] `python main.py` runs to completion
- [ ] `python -m pytest evaluation/tests/ -v` — all tests pass
- [ ] `notebooks/<task_name>.ipynb` exists and executes without errors
- [ ] `evaluation/reference_outputs/metrics.json` exists with expected metrics
- [ ] `README.md`, `plan/approach.md`, `plan/design.md` exist
- [ ] `requirements.txt` lists all dependencies

### Code extraction
- **Copy and adapt from reference repos** — clone the reference repo, extract relevant functions, and fit them to our directory structure. Preserve original algorithm logic faithfully; do not rewrite algorithms from scratch.
- **Domain-standard libraries are allowed** — widely-used packages the original code depends on (e.g., `ehtim`, `astropy`, `torchkbnufft`, `scikit-image`) may be imported directly. Only task-specific research packages (e.g., `import dpi`, `import bhnerf`) should have their functions extracted into `src/`. List all dependencies in `requirements.txt`.
- **No cross-task imports** — each task must run independently.
- **No information leakage** — do not depend on packages published after the paper, since an agent solving the task would not have access to them.

### Data
- If real observation data is unavailable, **generate synthetic data** via `src/generate_data.py`. Results should be qualitatively similar to the paper figures, not necessarily identical.
- **Coordinate convention consistency** — `generate_data.py` must use the exact same pixel coordinate convention (e.g., half-pixel offset, axis direction) as `physics_model.py`. A mismatch causes correct amplitudes but wrong phases, yielding χ²/DOF orders of magnitude too high. Always verify by round-tripping: forward-model the ground truth with `physics_model.py`, compare to generated visibilities, and check χ²/DOF ≈ 1.
- Keep all existing reference materials (PDFs, `reference_website_github.md`) in each task folder.

### Testing
- Every function in `src/` must have a corresponding test fixture (`evaluation/fixtures/`) and unit test (`evaluation/tests/`).
- Fixture naming: `param_*` (constructor args), `input_*` (function inputs), `config_*` (settings), `output_*` (expected outputs).
- **Deterministic outputs** (e.g., forward model, gradient computation): test with exact numerical comparison (`rtol=1e-10`).
- **Stochastic outputs** (e.g., noise injection, MCMC samples): test statistical properties only (mean, variance, shape, dtype). Never rely on random seed matching for exact value comparison.

### Pilot reference
The completed tasks `tasks/eht_black_hole/` and `tasks/eht_black_hole_UQ/` serve as format references. When in doubt, follow their structure, code style, and test patterns.
