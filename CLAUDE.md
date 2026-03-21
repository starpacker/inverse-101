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
│   └── <task_name>.ipynb      # End-to-end tutorial
└── evaluation/
    ├── reference_outputs/     # Ground truth, reference reconstructions, metrics.json
    └── tests/                 # Unit tests (per-function) and integration tests (end-to-end)
```

## Evaluation Modes

| Mode              | Agent sees                         | Agent produces      | Evaluated against                |
|-------------------|------------------------------------|---------------------|----------------------------------|
| Plan generation   | README.md + data/meta_data         | plan/               | plan/approach.md + plan/design.md|
| Function-level    | README.md + plan/                  | Single function     | Unit tests + numerical precision |
| End-to-end        | README.md + data/                  | src/ + main.py      | evaluation/reference_outputs/    |
| + Skills          | Same + skills/                     | Same                | Same                             |

## Commands

### Install dependencies (per-task)
```bash
pip install -r tasks/eht_black_hole/requirements.txt
```

### Run reconstruction pipeline
```bash
cd tasks/eht_black_hole && python main.py
```

### Run evaluation tests
```bash
cd tasks/eht_black_hole && python -m pytest evaluation/tests/ -v
```

### Generate synthetic data (if needed)
```bash
cd tasks/eht_black_hole && python -c "from src.generate_data import generate_dataset; generate_dataset()"
```

## Architecture

- **preprocessing.py**: Loads raw_data (NPZ) and meta_data (JSON), returns arrays ready for the physics model.
- **physics_model.py**: Implements the forward operator (x → y).
- **solvers.py**: All solvers share the interface `reconstruct(model, vis, noise_std) → image`.
- **visualization.py**: Plotting and quality metrics (NRMSE, NCC, dynamic range).

## Key Dependencies

NumPy, SciPy (L-BFGS-B optimizer), Matplotlib. Code prioritizes clarity over performance.
GPU tasks (e.g., DPI, α-DPI, BH-NeRF) additionally require PyTorch or JAX.

## Task Cleaning Principles

Tasks are created by **cleaning** existing research code from published papers and their reference implementations. The goal is to restructure real computational imaging code into the standardized benchmark format.

### Code extraction
- **Copy from reference repos is encouraged** — clone the reference repo, extract relevant functions, and adapt them to fit our directory structure. Do not rewrite algorithms from scratch.
- **Preserve original algorithm logic** — the cleaned code should faithfully follow the paper and original implementation, not reinvent the approach.
- **Self-contained** — each task must run independently. No cross-task imports, and no runtime dependency on the original package (e.g., do not `import ehtim`, `import dpi`, or `import bhnerf`). Instead, extract the needed functions directly into `src/`.
- **No information leakage** — tasks simulate real-world development. Do not depend on packages published after the paper, since an agent solving the task would not have access to them.

### Data
- If real observation data is unavailable, **generate synthetic data** via `src/generate_data.py`. Results should be qualitatively similar to the paper figures, not necessarily identical.
- Keep all existing reference materials (PDFs, `reference_website_github.md`) in each task folder.

### Testing
- Every function in `src/` must have a corresponding test fixture (`evaluation/fixtures/`) and unit test (`evaluation/tests/`).
- Fixture naming: `param_*` (constructor args), `input_*` (function inputs), `config_*` (settings), `output_*` (expected outputs).
- **Deterministic outputs** (e.g., forward model, gradient computation): test with exact numerical comparison (`rtol=1e-10`).
- **Stochastic outputs** (e.g., noise injection, MCMC samples): test statistical properties only (mean, variance, shape, dtype). Never rely on random seed matching for exact value comparison.

### Pilot reference
The completed task `tasks/eht_black_hole/` serves as the format reference for all other tasks. When in doubt, follow its structure, code style, and test patterns.
