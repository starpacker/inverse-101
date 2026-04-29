# Imaging-101

**Benchmarking LLM Agents for Scientific Computational Imaging Problems**

[Website](https://starpacker.github.io/agent-imaging-website/) | Paper | [Benchmark Tasks](tasks/) | [Evaluation Guide](docs/EVALUATION_GUIDE.md)

---

## Overview

**Imaging-101** is a benchmark of **57 expert-verified computational imaging tasks** spanning **six scientific domains** (astronomy, biology, chemistry & material science, earth science, medicine, and physics). Each task is grounded in a peer-reviewed publication and canonicalized into a standardized four-stage pipeline: *preprocessing*, *forward physics modeling*, *inverse solver*, and *visualization*.

Three complementary evaluation tracks probe distinct agent capabilities:

| Track | What it tests | Metric |
|-------|---------------|--------|
| **Planning** | Algorithm selection and pipeline design | Expert-reviewed pass rate |
| **Function-level** | Per-module correctness against unit tests | Test pass rate |
| **End-to-end** | Full pipeline reconstruction quality | NCC / NRMSE |

Evaluating seven frontier LLMs reveals systematic challenges — algorithm selection errors, numerical convention drift, and pipeline integration failures — that go beyond those exposed by general coding benchmarks.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Setup](#setup)
- [Evaluation](#evaluation)
  - [Function-Level](#1-function-level-evaluation)
  - [End-to-End](#2-end-to-end-evaluation)
  - [Third-Party Agents](#3-third-party-agents-claude-code-cursor-etc)
  - [Plan-Mode](#4-plan-mode-evaluation)
- [Add Your Own Task](#add-your-own-task)
- [Task List](#task-list)
- [Task Structure](#task-structure)
- [Repository Layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

---

## Key Results

Results from evaluating 7 frontier LLMs on all 57 tasks (multi-agent pipeline). **Bold** = best; <ins>underline</ins> = second.

| Model | Planning (Overall) | Function (Module Pass) | End-to-End (Success) |
|-------|--------------------|------------------------|----------------------|
| **Claude-4.6-Opus** | **78.9%** | 24.6% | **31.6%** |
| Kimi-k2.5 | <ins>75.4%</ins> | 17.5% | 10.5% |
| GPT-5.4 | 68.4% | 26.3% | <ins>14.0%</ins> |
| DeepSeek-V3 | 54.4% | 12.3% | 7.0% |
| Qwen3.6-Plus | 52.6% | 17.5% | 5.3% |
| GLM-5 | 47.4% | 21.1% | 8.8% |
| Gemini-3.1-Pro | 43.9% | **40.4%** | 19.3% |

> **Reference:** Claude Code (black-box agent with shell access) achieves **56.1%** end-to-end success rate, reported separately due to its richer execution environment.

See the full breakdown (per-module, per-domain) in the [paper](#citation).

---

## Setup

**Requirements:** Python 3.9+ and an OpenAI-compatible API endpoint.

```bash
git clone https://github.com/HeSunPU/imaging-101.git
cd imaging-101
pip install -r evaluation_harness/requirements.txt   # only `requests>=2.28`
```

Configure your LLM credentials — either via environment variables:

```bash
export API_KEY="your-api-key"
export BASE_URL="https://api.openai.com/v1"
export MODEL="gpt-4o"
```

Or in `config_llm.yaml`:

```yaml
"your-model-name":
    api_type: "openai"
    base_url: "https://your-api-gateway/v1"
    api_key: "your-api-key"
    temperature: 0.2
```

---

## Evaluation

### 1. Function-Level Evaluation

Tests whether an agent can implement **individual modules** (`physics_model.py`, `preprocessing.py`, `solvers.py`) given the task description and plan. Each module is graded by pytest against reference fixtures.

```bash
# Evaluate a single module
python -m evaluation_harness run \
    --task ct_sparse_view \
    --mode function \
    --target-function physics_model \
    --model $MODEL \
    --base-url $BASE_URL \
    --api-key $API_KEY \
    --framework react \
    --output results/function_mode \
    -v

# Evaluate all modules of a task
for module in physics_model preprocessing solvers; do
    python -m evaluation_harness run \
        --task ct_sparse_view \
        --mode function \
        --target-function $module \
        --model $MODEL --base-url $BASE_URL --api-key $API_KEY \
        --framework react --output results/function_mode -v
done

# Aggregate results
python -m evaluation_harness summarize \
    --dir results/function_mode/ct_sparse_view/your-model_20260401
```

### 2. End-to-End Evaluation

Tests whether an agent can **build the full imaging pipeline from scratch**, producing a reconstruction from raw observation data. Three difficulty levels control how much guidance the agent receives:

| Level | Agent receives | Difficulty |
|-------|---------------|------------|
| `L1` | Task README only | Hardest — agent plans from scratch |
| `L2` | README + `approach.md` | Medium — solution approach given |
| `L3` | README + `approach.md` + `design.md` | Easiest — full design given |

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --level L1 \
    --model $MODEL \
    --base-url $BASE_URL \
    --api-key $API_KEY \
    --framework react \
    --output results/end_to_end \
    -v
```

Four agent frameworks are supported:

| Framework | `--framework` | Description |
|-----------|---------------|-------------|
| ReAct | `react` | Single-agent Thought → Action → Observation loop |
| Multi-Agent | `multi_agent` | Planner → Critic → Architect → Coder → Judge pipeline |
| Copilot | `copilot` | Sandbox preparation for third-party agents |
| DeepCode | `deepcode` | HKUDS DeepCode autonomous multi-agent |

### 3. Third-Party Agents (Claude Code, Cursor, etc.)

For agents that run outside the harness:

```bash
# Step 1: Prepare a sandbox workspace
python -m evaluation_harness prepare --task ct_sparse_view --level L1

# Step 2: Run your agent in the workspace
#   The workspace contains .prompt.md — paste it into your agent.
#   The agent should produce output/reconstruction.npy

# Step 3: Score the result
python -m evaluation_harness collect \
    --task ct_sparse_view \
    --workspace-dir workspaces/ct_sparse_view_L1/ \
    --agent-name claude_code

# Optional: compute NCC/NRMSE directly
python scripts/compute_metrics.py \
    --workspace workspaces/ct_sparse_view_L1/ \
    --task ct_sparse_view
```

### 4. Plan-Mode Evaluation

Evaluates **planning ability only** (no code execution) using LLM-as-judge pairwise comparison against reference plans.

```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode plan \
    --model $MODEL \
    --base-url $BASE_URL \
    --api-key $API_KEY \
    --framework react \
    --output results/plan -v
```

For the complete evaluation guide, see [`docs/EVALUATION_GUIDE.md`](docs/EVALUATION_GUIDE.md).

---

## Add Your Own Task

Each task represents a computational imaging inverse problem: given observations **y**, recover the signal **x** via a known forward model **A** where **y = A(x) + noise**.

```bash
# 1. Create the directory structure
mkdir -p tasks/my_task/{data,plan,src,evaluation/{tests,fixtures,reference_outputs},notebooks}

# 2. Populate the required files:
#    README.md            — physics background, data description, method hints
#    main.py              — pipeline entry point
#    data/raw_data.npz    — observation data
#    data/meta_data.json  — imaging parameters (no solver hyperparameters)
#    src/physics_model.py — forward model A: x → y
#    src/preprocessing.py — raw data loading and preparation
#    src/solvers.py       — inverse solver: y → x̂
#    evaluation/tests/    — pytest suites with fixtures

# 3. Verify
cd tasks/my_task
python main.py                          # pipeline completes
python -m pytest evaluation/tests/ -v   # all tests pass

# 4. Benchmark an agent on your task
cd ../..
python -m evaluation_harness run \
    --task my_task --mode function \
    --target-function physics_model \
    --model $MODEL --base-url $BASE_URL --api-key $API_KEY \
    --framework react --output results/function_mode -v
```

Full instructions: [`docs/NEW_TASK_GUIDE.md`](docs/NEW_TASK_GUIDE.md)

---

## Task List

57 tasks across 6 scientific domains. Status: ✅ verified | 🔍 pending verification | ⚠️ known issues

<details>
<summary><b>Astronomy</b> (9 tasks)</summary>

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [EHT Black Hole (static)](tasks/eht_black_hole_original/) | radio interferometry, compressed sensing | Hard | ✅ |
| [EHT Black Hole (dynamic)](tasks/eht_black_hole_dynamic/) | radio interferometry, dynamic imaging | Hard | ✅ |
| [EHT Black Hole UQ](tasks/eht_black_hole_UQ/) | radio interferometry, variational inference | Hard | ✅ |
| [EHT Black Hole Feature Extraction](tasks/eht_black_hole_feature_extraction_dynamic/) | variational inference, feature extraction | Hard | 🔍 |
| [EHT Black Hole Tomography (NeRF)](tasks/eht_black_hole_tomography/) | tomography, neural rendering | Hard | 🔍 |
| [Lucky Imaging](tasks/lucky_imaging/) | imaging through turbulence, image sharpening | Medium | 🔍 |
| [Exoplanet Direct Imaging](tasks/exoplanet_imaging/) | high-contrast imaging, PSF subtraction | Medium | 🔍 |
| [Shack-Hartmann Wavefront Sensing](tasks/shack-hartmann/) | adaptive optics, wavefront sensing | Medium | 🔍 |
| [Shapelet Source Reconstruction](tasks/shapelet_source_reconstruction/) | gravitational lensing, shapelet decomposition | Medium | 🔍 |

</details>

<details>
<summary><b>Biology</b> (9 tasks)</summary>

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [SSNP-ODT](tasks/SSNP_ODT/) | tomography, wave imaging, inverse scattering | Hard | ✅ |
| [Reflection-ODT](tasks/reflection_ODT/) | tomography, wave imaging, phase retrieval | Hard | ✅ |
| [Fourier Ptychography](tasks/fourier_ptychography/) | phase retrieval, ptychography, super-resolution | Medium | ✅ |
| [Microscope Denoising](tasks/microscope_denoising/) | denoising, deconvolution, self-supervised learning | Medium | 🔍 |
| [Hessian SIM](tasks/hessian_sim/) | structured illumination, super-resolution | Medium | 🔍 |
| [Light Field Microscope](tasks/light_field_microscope/) | light field imaging, deconvolution | Hard | 🔍 |
| [Single-Molecule Light Field](tasks/single_molecule_light_field/) | localization microscopy, light field | Medium | 🔍 |
| [FPM INR Reconstruction](tasks/fpm_inr_reconstruction/) | phase retrieval, implicit neural representation | Medium | 🔍 |
| [S2ISM](tasks/s2ism/) | image scanning microscopy, deconvolution | Medium | 🔍 |

</details>

<details>
<summary><b>Physics</b> (6 tasks)</summary>

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [Conventional Ptychography](tasks/conventional_ptychography/) | phase retrieval, ptychography | Medium | ✅ |
| [PnP-CASSI](tasks/spectral_snapshot_compressive_imaging/) | compressed sensing, hyperspectral | Medium | 🔍 |
| [Electron Ptychography](tasks/electron_ptychography/) | electron microscopy, phase retrieval | Medium | 🔍 |
| [Confocal NLOS](tasks/confocal-nlos-fk/) | non-line-of-sight, wave imaging | Medium | 🔍 |
| [Lensless Imaging](tasks/lensless_imaging/) | lensless imaging, deconvolution | Medium | 🔍 |
| [Differentiable Deflectometry](tasks/differentiable_deflectometry/) | deflectometry, optical metrology | Hard | 🔍 |

</details>

<details>
<summary><b>Chemistry & Material Science</b> (5 tasks)</summary>

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [MCR Hyperspectral Unmixing](tasks/mcr_hyperspectral/) | hyperspectral, spectral unmixing | Medium | 🔍 |
| [Raman Cell Phenotyping](tasks/raman_cell_phenotyping/) | Raman spectroscopy, unmixing | Medium | 🔍 |
| [CARS Spectroscopy](tasks/cars_spectroscopy/) | Raman spectroscopy, temperature inversion | Medium | 🔍 |
| [X-ray Ptychography](tasks/xray_ptychography_tike/) | wave imaging, phase retrieval | Medium | 🔍 |
| [X-ray Laminography](tasks/xray_laminography_tike/) | tomographic imaging, iterative reconstruction | Easy | 🔍 |

</details>

<details>
<summary><b>Earth Science</b> (6 tasks)</summary>

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [Seismic FWI](tasks/seismic_FWI_original/) | wave imaging, inverse scattering | Medium | ✅ |
| [Seismic Traveltime Tomography](tasks/seismic_traveltime_tomography/) | traveltime tomography, eikonal equation | Medium | ✅ |
| [Seismic LSRTM](tasks/seismic_lsrtm_original/) | wave imaging, seismic migration | Medium | 🔍 |
| [InSAR Phase Unwrapping](tasks/insar_phase_unwrapping/) | phase unwrapping, interferometry | Medium | 🔍 |
| [Weather Radar Data Assimilation](tasks/weather_radar_data_assimilation/) | data assimilation, variational inversion | Hard | 🔍 |
| [ERA5 Tensor-Var](tasks/era5_tensorvar/) | data assimilation, deep kernel features | Hard | 🔍 |

</details>

<details>
<summary><b>Medicine</b> (22 tasks)</summary>

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [Fan-Beam CT](tasks/ct_fan_beam/) | CT, fan-beam, filtered back-projection | Medium | 🔍 |
| [Sparse-View CT](tasks/ct_sparse_view/) | CT, sparse-view, regularized reconstruction | Medium | 🔍 |
| [Low-Dose CT (Poisson)](tasks/ct_poisson_lowdose/) | CT, statistical reconstruction | Medium | 🔍 |
| [Dual-Energy CT](tasks/ct_dual_energy/) | dual-energy CT, material decomposition | Medium | 🔍 |
| [X-ray CT Tooth](tasks/xray_tooth_gridrec/) | CT, filtered back-projection | Easy | 🔍 |
| [MRI L1-Wavelet](tasks/mri_l1_wavelet/) | compressed sensing MRI | Easy | 🔍 |
| [MRI Total Variation](tasks/mri_tv/) | compressed sensing MRI | Easy | 🔍 |
| [MRI T2 Mapping](tasks/mri_t2_mapping/) | quantitative MRI, parameter estimation | Easy | 🔍 |
| [MRI CG-SENSE](tasks/mri_sense/) | parallel imaging | Medium | 🔍 |
| [MRI GRAPPA](tasks/mri_grappa/) | parallel imaging, k-space interpolation | Medium | 🔍 |
| [MRI Non-Cartesian CS](tasks/mri_noncartesian_cs/) | compressed sensing, non-Cartesian | Medium | 🔍 |
| [Diffusion MRI DTI](tasks/diffusion_mri_dti/) | quantitative MRI, diffusion imaging | Medium | 🔍 |
| [Dynamic DCE-MRI](tasks/mri_dynamic_dce/) | dynamic MRI, temporal regularization | Medium | 🔍 |
| [MRI PnP-ADMM](tasks/mri_pnp_admm/) | plug-and-play, learned prior | Medium | 🔍 |
| [MRI VarNet](tasks/mri_varnet/) | deep learning, unrolled network | Hard | 🔍 |
| [PnP-MSSN MRI](tasks/pnp_mri_reconstruction/) | plug-and-play, self-supervised | Medium | ⚠️ |
| [Plane Wave Ultrasound](tasks/plane_wave_ultrasound/) | wave imaging, ultrafast imaging | Medium | 🔍 |
| [Ultrasound SoS Tomography](tasks/ultrasound_sos_tomography/) | ultrasound tomography | Medium | 🔍 |
| [USCT FWI](tasks/usct_FWI/) | wave imaging, full-waveform inversion | Hard | 🔍 |
| [PET MLEM](tasks/pet_mlem/) | emission tomography, expectation maximization | Medium | 🔍 |
| [Photoacoustic Tomography](tasks/photoacoustic_tomography/) | photoacoustic, wave imaging | Medium | 🔍 |
| [EIT Conductivity](tasks/eit_conductivity_reconstruction/) | impedance imaging, finite element method | Medium | 🔍 |

</details>

---

## Task Structure

Every task follows a standardized layout:

```
tasks/<task_name>/
├── README.md               # Physics background, data description, method hints
├── requirements.txt        # Python dependencies
├── main.py                 # Pipeline entry point
├── data/
│   ├── raw_data.npz        # Observations (keys documented in README)
│   ├── ground_truth.npz    # Reference signal for scoring
│   └── meta_data.json      # Imaging parameters (no solver settings)
├── plan/
│   ├── approach.md         # Algorithm description (used at L2 difficulty)
│   └── design.md           # Module architecture (used at L3 difficulty)
├── src/
│   ├── physics_model.py    # Forward model A: x → y
│   ├── preprocessing.py    # Raw data → processed observations
│   ├── solvers.py          # Inverse solver: y → x̂
│   └── visualization.py    # Plotting utilities
├── evaluation/
│   ├── metrics.json        # NCC/NRMSE pass/fail thresholds
│   ├── reference_outputs/  # Precomputed reference results
│   ├── fixtures/           # Captured I/O for unit tests
│   └── tests/              # pytest suites
└── notebooks/
    └── <task_name>.ipynb   # Executed tutorial with precomputed results
```

**Design principles:**
- Forward models grounded in physics, not black boxes
- Solvers span the spectrum: analytical → iterative → regularized → learned
- Standardized data interface across all tasks
- Code optimized for clarity and education

---

## Repository Layout

```
imaging-101/
├── tasks/                   # 57 benchmark tasks
├── evaluation_harness/      # Evaluation framework
│   ├── __main__.py          #   CLI: run | prepare | collect | summarize
│   ├── runner.py            #   Orchestrator
│   ├── scorer.py            #   pytest + NCC/NRMSE
│   ├── config.py            #   Configuration dataclasses
│   ├── llm_client.py        #   OpenAI-compatible API client
│   ├── prompts.py           #   Prompt templates
│   ├── frameworks/          #   react, multi_agent, copilot, deepcode
│   └── agents/              #   Planner, Architect, Coder, Judge
├── scripts/                 # Batch runners and analysis tools
├── docs/
│   ├── EVALUATION_GUIDE.md  #   Full evaluation tutorial
│   └── NEW_TASK_GUIDE.md    #   How to contribute a task
└── config_llm.yaml          # LLM endpoint configurations
```

---

## Citation

```bibtex
@inproceedings{imaging101,
  title     = {Imaging-101: Benchmarking LLM Agents for
               Scientific Computational Imaging Problems},
  author    = {Anonymous},
  booktitle = {International Conference on Computational
               Photography (ICCP)},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

Coming soon.
