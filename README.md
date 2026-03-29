# imaging-101: Scientific Imaging Benchmark

A benchmark suite for evaluating **coding agents** on **computational and scientific imaging** tasks.
It covers 8 domains (physics, chemistry, biology, medicine, astronomy, earth science, mechanics, industrial/commercial)
with standardized task structure, reference implementations, and automated evaluation via an integrated
**ReAct / Multi-Agent evaluation harness**.

---

## Repository Structure

```
imaging-101/
├── CLAUDE.md                     # Agent guidance (coding conventions, evaluation rules)
├── config_llm.yaml               # LLM provider/model configurations
├── evaluation_harness/           # Benchmark evaluation framework
│   ├── __main__.py               # CLI entry point: python -m evaluation_harness run ...
│   ├── config.py                 # LLMConfig / TaskConfig / RunConfig dataclasses
│   ├── runner.py                 # Top-level orchestrator (BenchmarkRunner)
│   ├── agent.py                  # ReAct agent loop (Thought → Action → Observation)
│   ├── multi_agent.py            # Multi-agent pipeline (Planner/Critic/Architect/Coder/Judge)
│   ├── scorer.py                 # pytest runner + quality metrics + visualization generation
│   ├── visualizer.py             # Auto-generates comparison/residual/metrics figures
│   ├── llm_client.py             # OpenAI-compatible API calls
│   ├── prompts.py                # Prompt templates per evaluation mode
│   ├── local_runner.py           # Local sandbox backend
│   ├── docker_runner.py          # Docker container sandbox backend
│   ├── Dockerfile                # Sandbox image (Python 3.11 + numpy/scipy/matplotlib/pytest)
│   └── agents/                   # Multi-agent sub-agent modules
├── tasks/                        # Individual imaging tasks (see Task Template below)
│   ├── eht_black_hole/
│   ├── eht_black_hole_original/  # ← Primary benchmark task (cleaned from ehtim)
│   ├── eht_black_hole_dynamic/
│   ├── eht_black_hole_tomography/
│   ├── eht_black_hole_UQ/
│   ├── light_field_microscope/
│   ├── reflection_ODT/
│   ├── single_molecule_light_field/
│   ├── SSNP_ODT/
│   └── ...
├── skills/                       # Shared skill library (available to agents)
├── scripts/                      # Convenience scripts for running evaluations
│   ├── run_end2end_gemini.sh     # End-to-end eval with Gemini (react + multi_agent)
│   ├── run_function_evals.sh     # Function-level eval for all modules
│   ├── run_preprocessing_eval_gemini.sh
│   ├── run_comparison.sh
│   └── compare_frameworks.py     # Python script: runs both frameworks & writes comparison JSON
├── results/                      # Evaluation result JSONs + figure artifacts
│   └── figures/                  # Auto-generated visualization outputs
├── logs/
│   ├── eval_runs/                # Run logs
│   └── interactions/             # Agent interaction transcripts
└── README.md                     # ← You are here
```

---

## Tasks

| Task | Domain | Measurement | Status |
|------|--------|-------------|--------|
| [EHT Black Hole Original](tasks/eht_black_hole_original/) | Radio astronomy | Sparse Fourier (VLBI) | ✅ Cleaned & evaluated |
| [EHT Black Hole](tasks/eht_black_hole/) | Radio astronomy | Sparse Fourier (VLBI) | ✅ Reference |
| [EHT Black Hole UQ](tasks/eht_black_hole_UQ/) | Radio astronomy | Uncertainty quantification | ✅ Reference |
| [EHT Black Hole Dynamic](tasks/eht_black_hole_dynamic/) | Radio astronomy | Dynamic imaging | 🔧 In progress |
| [Light Field Microscope](tasks/light_field_microscope/) | Biology | Light field | 🔧 In progress |
| [Reflection ODT](tasks/reflection_ODT/) | Biology | Diffraction tomography | 🔧 In progress |
| [SSNP ODT](tasks/SSNP_ODT/) | Biology | Scattering tomography | 🔧 In progress |
| *More coming soon* | | | |

---

## Task Template

Every task follows a standardized layout:

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
│   └── <task_name>.ipynb      # End-to-end tutorial notebook (user review interface)
└── evaluation/
    ├── reference_outputs/     # Ground truth, metrics.json, pretrained checkpoints
    ├── fixtures/              # Per-function test fixtures
    └── tests/                 # Unit tests, parity tests, integration tests
```

**Design Principles:**
- Forward models grounded in physics, not black boxes
- Solvers span the spectrum: analytical → iterative → regularized → learned
- All tasks include synthetic data generation (no external downloads required)
- Code written for clarity and education, not just performance

---

## Getting Started

### 1. Clone & install

```bash
git clone https://github.com/HeSunPU/imaging-101.git
cd imaging-101

# Install harness dependencies
pip install -r evaluation_harness/requirements.txt

# Install task-specific dependencies
pip install -r tasks/eht_black_hole_original/requirements.txt
```

### 2. Configure LLM provider

Edit `config_llm.yaml` or pass via CLI flags:

```yaml
# config_llm.yaml example
"gemini-2.5-pro":
  api_type: "openai"
  base_url: "https://your-api-gateway/v1"
  api_key: "your-api-key"
  temperature: 0.2
```

Or use environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. (Optional) Build Docker sandbox

```bash
docker build -t imaging101-sandbox -f evaluation_harness/Dockerfile .
```

> If Docker is unavailable, the harness falls back to `LocalRunner` automatically.

---

## Evaluation Modes

The harness supports three evaluation modes, each testing different agent capabilities:

### Plan Generation

Agent sees `README.md` + `data/` and generates solution plan files.

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode plan \
    --model gemini-2.5-pro \
    --base-url https://your-api/v1 \
    --api-key $API_KEY
```

### Function-Level

Agent implements a single function (e.g., `preprocessing.load_observation`), tested by unit tests.

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode function \
    --target-function preprocessing.load_observation \
    --model gemini-2.5-pro \
    --base-url https://your-api/v1 \
    --api-key $API_KEY
```

### End-to-End

Agent plans + implements the full `src/` pipeline + `main.py`, producing `output/reconstruction.npy`.
Quality metrics (NRMSE, NCC, PSNR, SSIM) and **comparison visualizations** are generated automatically.

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --framework react \
    --model gemini-2.5-pro \
    --max-iterations 50 \
    --timeout 600 \
    --base-url https://your-api/v1 \
    --api-key $API_KEY \
    --output results \
    -v
```

---

## Agent Frameworks

| Framework | CLI Flag | Description |
|-----------|----------|-------------|
| **ReAct** | `--framework react` (default) | Single-agent loop: Thought → Action → Observation. Actions: `DO`, `RUN`, `WRITE_FILE`, `READ_FILE` |
| **Multi-Agent** | `--framework multi_agent` | Pipeline of specialized agents: Planner → Critic → Architect → Coder → Judge |

### Compare frameworks side-by-side

```bash
# Using the convenience script
bash scripts/run_end2end_gemini.sh

# Or using the Python comparator
python scripts/compare_frameworks.py \
    --task eht_black_hole_original \
    --model gemini-2.5-pro \
    --base-url https://your-api/v1 \
    --api-key $API_KEY
```

---

## CLI Reference

```
python -m evaluation_harness run [OPTIONS]

Required:
  --task TEXT              Task name (e.g., eht_black_hole_original)
  --mode TEXT              Evaluation mode: plan | function | end_to_end
  --model TEXT             LLM model identifier

Mode-specific:
  --target-function TEXT   Function target for function mode (e.g., preprocessing.load_observation)
  --framework TEXT         Agent framework: react | multi_agent (default: react)

LLM configuration:
  --base-url TEXT          API base URL (default: https://api.openai.com/v1)
  --api-key TEXT           API key (or set OPENAI_API_KEY env var)

Execution:
  --max-iterations INT     Maximum agent iterations (default: 20)
  --timeout INT            Execution timeout in seconds (default: 600)
  --docker-image TEXT      Docker sandbox image (default: imaging101-sandbox)

Output:
  --output TEXT            Output directory (default: results)
  --log-file TEXT          Log file path (default: logs/interactions/<task>_<mode>_<ts>.md)
  -v, --verbose            Verbose output
```

---

## Output Structure

After an evaluation run, artifacts are organized as follows:

### Result JSON

```
results/<task>_<mode>_<framework>_<model>_<timestamp>.json
```

Example fields:
```json
{
  "task_name": "eht_black_hole_original",
  "mode": "end_to_end",
  "model": "gemini-2.5-pro",
  "framework": "multi_agent",
  "quality_metrics": {
    "nrmse": 0.738,
    "ncc": 0.701,
    "psnr": 19.6,
    "ssim": 0.345
  },
  "total_tokens": 1291011,
  "wall_time_seconds": 5302,
  "iterations": 8,
  "visualization_paths": { ... }
}
```

### Auto-Generated Visualizations (end_to_end mode)

```
results/figures/<run_id>/
├── comparison.png        # Side-by-side: reconstruction vs ground truth
├── residual.png          # Residual map (reconstruction - GT)
├── metrics_card.png      # Quality metrics summary card
├── cross_section.png     # Cross-section comparison plot
└── reconstruction.npy    # Copy of agent's output
```

### Agent Interaction Logs

```
logs/interactions/<task>_<mode>[_<target>]_<timestamp>.md
```

---

## Evaluation Workflow

The `end_to_end` evaluation flow:

```
1. Agent receives: README.md + data/ (raw_data.npz, meta_data)
2. Agent generates: plan/approach.md + plan/design.md
3. Agent implements: src/*.py + main.py
4. Harness runs: main.py → output/reconstruction.npy
5. Scorer computes: quality metrics (NRMSE, NCC, PSNR, SSIM) vs ground_truth.npy
6. Visualizer generates: comparison, residual, metrics card, cross-section figures
7. Results saved: results/<run_id>.json + results/figures/<run_id>/
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/run_end2end_gemini.sh` | Run e2e eval with Gemini-2.5-pro (both react & multi_agent) |
| `scripts/run_function_evals.sh` | Run function-level evals for all src modules |
| `scripts/run_preprocessing_eval_gemini.sh` | Single function eval: preprocessing module |
| `scripts/run_comparison.sh` | Run framework comparison |
| `scripts/compare_frameworks.py` | Python: run both frameworks, write comparison JSON |

---

## Contributing

To add a new imaging task, follow the task template and submit a pull request.
See `CLAUDE.md` for detailed task cleaning guidelines and coding conventions.

Tasks of interest: X-ray CT, MRI, phase retrieval, lensless imaging,
seismic imaging, optical coherence tomography, and more.
