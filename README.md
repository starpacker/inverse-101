# imaging-101: Scientific Imaging Benchmark

A benchmark suite for evaluating **coding agents** on **computational and scientific imaging** inverse problems.
Covers **57 tasks** across **6 domains** with standardized task structure, reference implementations, and automated evaluation.

> **Docs:** [Evaluation Guide](docs/EVALUATION_GUIDE.md) | [New Task Guide](docs/NEW_TASK_GUIDE.md) | [Release Checklist](docs/RELEASE_CHECKLIST.md)

---

## Quick Start: Evaluate a Task in 3 Minutes

```bash
# 1. Clone and install (only `requests` is required)
git clone https://github.com/HeSunPU/imaging-101.git
cd imaging-101
pip install -r evaluation_harness/requirements.txt

# 2. Set your LLM API credentials
export API_KEY="your-api-key"
export BASE_URL="https://api.openai.com/v1"   # or any OpenAI-compatible endpoint
export MODEL="gpt-4o"                          # your model name

# 3. Run function-mode evaluation on one task
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
```

That's it — the agent will read the task, write code, run pytest, and iterate until tests pass. Results are saved to `results/function_mode/ct_sparse_view/`.

**Want to run end-to-end instead?** The agent builds the full pipeline from scratch:

```bash
python -m evaluation_harness run \
    --task ct_sparse_view \
    --mode end_to_end \
    --level L1 \
    --model $MODEL \
    --base-url $BASE_URL \
    --api-key $API_KEY \
    --framework react \
    --output results/end_to_end \
    -v
```

**Using a third-party agent (Claude Code, Cursor, etc.)?**

```bash
# Prepare a sandbox workspace with the task prompt
python -m evaluation_harness prepare --task ct_sparse_view --level L1
# → Run your agent in the workspace, let it produce output/reconstruction.npy
# Score the result
python -m evaluation_harness collect \
    --task ct_sparse_view \
    --workspace-dir workspaces/ct_sparse_view_L1/ \
    --agent-name claude_code
```

---

## Add Your Own Task and Evaluate It

You can add any computational imaging task to the benchmark. Here's the minimal workflow:

```bash
# 1. Create the standard directory structure
mkdir -p tasks/my_task/{data,plan,src,evaluation/{tests,fixtures,reference_outputs},notebooks}

# 2. Add your files following the template:
#    - README.md          → problem description (physics, data format, method hints)
#    - data/raw_data.npz  → observation data
#    - data/meta_data.json → imaging parameters
#    - src/physics_model.py, src/preprocessing.py, src/solvers.py → reference implementation
#    - main.py            → pipeline entry point
#    - evaluation/tests/test_*.py → unit tests with fixtures

# 3. Verify everything works
cd tasks/my_task
python main.py                          # pipeline runs end-to-end
python -m pytest evaluation/tests/ -v   # all unit tests pass

# 4. Evaluate an agent on your new task
cd ../..
python -m evaluation_harness run \
    --task my_task \
    --mode function \
    --target-function physics_model \
    --model $MODEL --base-url $BASE_URL --api-key $API_KEY \
    --framework react --output results/function_mode -v
```

See the full [New Task Guide](docs/NEW_TASK_GUIDE.md) for detailed instructions on each file, test fixtures, plan documents, and common pitfalls.

---

## Tasks

Status: ✅ Completed & verified — 🔍 Completed, pending further verification — ⚠️ Known issues

### Astronomy (9 tasks)

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [EHT Black Hole (static)](tasks/eht_black_hole_original/) | radio interferometry, compressed sensing | Hard | ✅ |
| [EHT Black Hole (dynamic)](tasks/eht_black_hole_dynamic/) | radio interferometry, dynamic imaging | Hard | ✅ |
| [EHT Black Hole UQ](tasks/eht_black_hole_UQ/) | radio interferometry, variational inference | Hard | ✅ |
| [EHT Black Hole Feature Extraction](tasks/eht_black_hole_feature_extraction_dynamic/) | radio interferometry, variational inference, feature extraction, dynamic imaging | Hard | 🔍 |
| [EHT Black Hole Tomography (NeRF)](tasks/eht_black_hole_tomography/) | radio interferometry, tomography, neural rendering | Hard | 🔍 |
| [Lucky Imaging](tasks/lucky_imaging/) | imaging through turbulence, lucky imaging, image sharpening | Medium | 🔍 |
| [Exoplanet Direct Imaging](tasks/exoplanet_imaging/) | high-contrast imaging, PSF subtraction | Medium | 🔍 |
| [Shack-Hartmann Wavefront Sensing](tasks/shack-hartmann/) | adaptive optics, wavefront sensing | Medium | 🔍 |
| [Shapelet Source Reconstruction](tasks/shapelet_source_reconstruction/) | gravitational lensing, shapelet decomposition, image deconvolution | Medium | 🔍 |

### Biology (9 tasks)

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [SSNP-ODT](tasks/SSNP_ODT/) | tomography, wave imaging, inverse scattering | Hard | ✅ |
| [Reflection-ODT](tasks/reflection_ODT/) | tomography, wave imaging, phase retrieval | Hard | ✅ |
| [Fourier Ptychography](tasks/fourier_ptychography/) | phase retrieval, ptychography, super-resolution | Medium | ✅ |
| [Microscope Denoising (ZS-DeconvNet)](tasks/microscope_denoising/) | denoising, deconvolution, self-supervised learning | Medium | 🔍 |
| [Hessian SIM](tasks/hessian_sim/) | structured illumination microscopy, super-resolution, regularized reconstruction | Medium | 🔍 |
| [Light Field Microscope](tasks/light_field_microscope/) | wave imaging, light field imaging, deconvolution | Hard | 🔍 |
| [Single-Molecule Light Field (FLFM)](tasks/single_molecule_light_field/) | super-resolution, localization microscopy, light field imaging | Medium | 🔍 |
| [FPM INR Reconstruction](tasks/fpm_inr_reconstruction/) | computational microscopy, phase retrieval, implicit neural representation | Medium | 🔍 |
| [Scalable Structured Image Scanning Microscopy](tasks/s2ism/) | fluorescence microscopy, image scanning microscopy, deconvolution | Medium | 🔍 |

### Physics (6 tasks)

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [Conventional Ptychography](tasks/conventional_ptychography/) | phase retrieval, ptychography | Medium | ✅ |
| [Spectral Snapshot Compressive Imaging (PnP-CASSI)](tasks/spectral_snapshot_compressive_imaging/) | compressed sensing, hyperspectral imaging | Medium | 🔍 |
| [Electron Ptychography](tasks/electron_ptychography/) | electron microscopy, phase retrieval, ptychography | Medium | 🔍 |
| [Confocal NLOS — f-k Migration](tasks/confocal-nlos-fk/) | non-line-of-sight, wave imaging | Medium | 🔍 |
| [Lensless Imaging (DiffuserCam)](tasks/lensless_imaging/) | lensless imaging, deconvolution | Medium | 🔍 |
| [Differentiable Deflectometry](tasks/differentiable_deflectometry/) | deflectometry, differentiable rendering, optical metrology | Hard | 🔍 |

### Chemistry & Material Science (5 tasks)

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [MCR Hyperspectral Unmixing](tasks/mcr_hyperspectral/) | hyperspectral imaging, spectral unmixing | Medium | 🔍 |
| [Raman Cell Phenotyping](tasks/raman_cell_phenotyping/) | Raman spectroscopy, hyperspectral unmixing | Medium | 🔍 |
| [CARS Spectroscopy Temperature Inversion](tasks/cars_spectroscopy/) | Raman spectroscopy, hyperspectral unmixing | Medium | 🔍 |
| [X-ray Ptychography](tasks/xray_ptychography_tike/) | wave imaging, phase retrieval | Medium | 🔍 |
| [X-ray Laminography/Tomography](tasks/xray_laminography_tike/) | tomographic imaging, iterative reconstruction | Easy | 🔍 |

### Earth Science (6 tasks)

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [Seismic Full Waveform Inversion](tasks/seismic_FWI_original/) | wave imaging, inverse scattering | Medium | ✅ |
| [Seismic Traveltime Tomography](tasks/seismic_traveltime_tomography/) | traveltime tomography, eikonal equation, adjoint-state method | Medium | ✅ |
| [Seismic LSRTM](tasks/seismic_lsrtm_original/) | wave imaging, inverse scattering, seismic migration | Medium | 🔍 |
| [InSAR Phase Unwrapping](tasks/insar_phase_unwrapping/) | phase unwrapping, interferometry | Medium | 🔍 |
| [Weather Radar Data Assimilation](tasks/weather_radar_data_assimilation/) | data assimilation, variational inversion, radar | Hard | 🔍 |
| [ERA5 Tensor-Var](tasks/era5_tensorvar/) | data assimilation, weather, deep kernel features, variational inference | Hard | 🔍 |

### Medicine (22 tasks)

| Task | Keywords | Difficulty | Status |
|------|----------|------------|--------|
| [Fan-Beam CT](tasks/ct_fan_beam/) | computed tomography, fan-beam geometry, filtered back-projection | Medium | 🔍 |
| [Sparse-View CT](tasks/ct_sparse_view/) | computed tomography, sparse-view imaging, regularized reconstruction | Medium | 🔍 |
| [Low-Dose CT (Poisson)](tasks/ct_poisson_lowdose/) | computed tomography, statistical reconstruction, Poisson noise | Medium | 🔍 |
| [Dual-Energy CT](tasks/ct_dual_energy/) | dual-energy CT, material decomposition, spectral imaging | Medium | 🔍 |
| [X-ray CT Tooth (Gridrec)](tasks/xray_tooth_gridrec/) | computed tomography, filtered back-projection | Easy | 🔍 |
| [MRI L1-Wavelet](tasks/mri_l1_wavelet/) | compressed sensing, MRI reconstruction | Easy | 🔍 |
| [MRI Total Variation](tasks/mri_tv/) | compressed sensing, MRI reconstruction | Easy | 🔍 |
| [MRI T2 Mapping](tasks/mri_t2_mapping/) | quantitative MRI, parameter estimation | Easy | 🔍 |
| [MRI CG-SENSE](tasks/mri_sense/) | parallel imaging, image-domain reconstruction | Medium | 🔍 |
| [MRI GRAPPA](tasks/mri_grappa/) | parallel imaging, k-space interpolation | Medium | 🔍 |
| [MRI Non-Cartesian CS](tasks/mri_noncartesian_cs/) | compressed sensing, non-Cartesian MRI | Medium | 🔍 |
| [Diffusion MRI DTI](tasks/diffusion_mri_dti/) | quantitative MRI, parameter estimation, diffusion imaging | Medium | 🔍 |
| [Dynamic DCE-MRI](tasks/mri_dynamic_dce/) | dynamic MRI, temporal regularization | Medium | 🔍 |
| [MRI PnP-ADMM](tasks/mri_pnp_admm/) | compressed sensing MRI, plug-and-play, learned prior | Medium | 🔍 |
| [MRI VarNet](tasks/mri_varnet/) | deep learning reconstruction, unrolled network | Hard | 🔍 |
| [PnP-MSSN MRI](tasks/pnp_mri_reconstruction/) | plug-and-play, self-supervised learning | Medium | ⚠️ |
| [Plane Wave Ultrasound (Stolt f-k)](tasks/plane_wave_ultrasound/) | wave imaging, ultrafast imaging | Medium | 🔍 |
| [Ultrasound Speed-of-Sound Tomography](tasks/ultrasound_sos_tomography/) | ultrasound tomography, transmission imaging, iterative reconstruction | Medium | 🔍 |
| [USCT Full Waveform Inversion](tasks/usct_FWI/) | wave imaging, full-waveform inversion | Hard | 🔍 |
| [PET MLEM](tasks/pet_mlem/) | emission tomography, Poisson reconstruction, expectation maximization | Medium | 🔍 |
| [Photoacoustic Tomography](tasks/photoacoustic_tomography/) | photoacoustic imaging, wave imaging, back-projection | Medium | 🔍 |
| [EIT Conductivity Reconstruction](tasks/eit_conductivity_reconstruction/) | impedance imaging, regularized inversion, finite element method | Medium | 🔍 |

---

## Design Philosophy

Each task is structured around a common template:

```
tasks/<task_name>/
├── README.md               # Physics background, problem formulation, references
├── requirements.txt        # Dependencies
├── main.py                 # Reconstruction pipeline entry point
├── data/
│   ├── raw_data.npz        # Observation data (keys documented in README)
│   ├── ground_truth.npz    # Ground truth (keys documented in README)
│   └── meta_data.json      # Imaging parameters (JSON)
├── plan/
│   ├── approach.md         # Solution methodology (L2 hint)
│   └── design.md           # Code architecture with function signatures (L3 hint)
├── src/
│   ├── physics_model.py    # Physics-based forward model
│   ├── preprocessing.py    # Raw data -> processed observations
│   ├── solvers.py          # Inverse problem solvers
│   ├── visualization.py    # Plotting utilities and metrics
│   └── generate_data.py    # Synthetic data generation (optional)
├── evaluation/
│   ├── metrics.json        # Evaluation boundaries (NCC, NRMSE)
│   ├── reference_outputs/  # Ground truth, precomputed results
│   ├── fixtures/           # Per-function test data
│   └── tests/              # Unit tests for function-mode evaluation
└── notebooks/
    └── <task_name>.ipynb   # End-to-end tutorial
```

**Principles:**
- Forward models grounded in physics, not black boxes
- Solvers span the spectrum: analytical → iterative → regularized → learned
- Most tasks include synthetic data generation; a few use real experimental data
- Code written for clarity and education, not just performance
- Standardized data interface: observations in `data/raw_data.npz`, ground truth in `data/ground_truth.npz`

---

## Evaluation

### Modes

| Mode | What it tests | Metrics |
|------|--------------|---------|
| **Function** (`--mode function`) | Individual module implementations (physics_model, preprocessing, solvers) | Per-module test pass rate |
| **End-to-End** (`--mode end_to_end`) | Full pipeline from scratch | NCC, NRMSE reconstruction quality |
| **Plan** (`--mode plan`) | Planning ability only | LLM-as-judge pairwise + rubric scoring |

### Difficulty Levels (end-to-end)

| Level | Agent receives |
|-------|---------------|
| L1 | Task README only (agent plans from scratch) |
| L2 | README + approach.md |
| L3 | README + approach.md + design.md |

### Agent Frameworks

| Framework | CLI Flag | Description |
|-----------|----------|-------------|
| **ReAct** | `--framework react` | Single-agent: Thought → Action → Observation loop |
| **Multi-Agent** | `--framework multi_agent` | Pipeline: Planner → Critic → Architect → Coder → Judge |
| **Copilot** | `--framework copilot` | Third-party agent: prepares sandbox + prompt, agent works externally |
| **DeepCode** | `--framework deepcode` | HKUDS DeepCode autonomous multi-agent |

For detailed evaluation instructions, see [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md).

---

## LLM Configuration

Edit `config_llm.yaml` or pass credentials directly via CLI flags (`--model`, `--base-url`, `--api-key`):

```yaml
"your-model-name":
    api_type: "openai"
    base_url: "https://your-api-gateway/v1"
    api_key: "YOUR_API_KEY"
    temperature: 0.2
```

---

## Repository Structure

```
imaging-101/
├── tasks/                        # 57 imaging tasks (see tables above)
├── evaluation_harness/           # Benchmark evaluation framework
│   ├── __main__.py               #   CLI: run | prepare | collect | summarize
│   ├── runner.py                 #   BenchmarkRunner orchestrator
│   ├── scorer.py                 #   pytest runner + NCC/NRMSE quality metrics
│   ├── config.py                 #   LLMConfig / TaskConfig / RunConfig
│   ├── llm_client.py             #   OpenAI-compatible API client
│   ├── prompts.py                #   Prompt templates per evaluation mode
│   ├── multi_agent.py            #   Multi-agent pipeline orchestrator
│   ├── agents/                   #   Agent role implementations
│   └── frameworks/               #   Framework adapters (react, multi_agent, copilot, deepcode)
├── scripts/                      # Batch evaluation & analysis scripts
├── results/                      # Evaluation outputs
│   ├── function_mode/            #   51 tasks × 7 models (canonical)
│   ├── end_to_end/               #   Per-framework subdirs
│   └── audit/                    #   Summary CSVs
├── docs/                         # Documentation
│   ├── EVALUATION_GUIDE.md       #   Step-by-step evaluation tutorial
│   └── NEW_TASK_GUIDE.md         #   How to add a new task
├── config_llm.yaml               # LLM provider/model configurations
└── CLAUDE.md                     # Agent guidance
```

---

## CLI Reference

```
python -m evaluation_harness <command> [OPTIONS]

Commands:
  run          Run a benchmark evaluation
  prepare      Prepare sandbox for third-party agent
  collect      Collect and score third-party agent results
  summarize    Generate summary.json for function-mode runs

run options:
  --task TEXT              Task name (e.g., eht_black_hole_original)
  --mode TEXT              Evaluation mode: function | end_to_end | plan
  --model TEXT             LLM model identifier
  --target-function TEXT   Function target for function mode
  --framework TEXT         Agent framework: react | multi_agent | copilot | deepcode
  --level TEXT             Difficulty level: L1 | L2 | L3 (default: L1)
  --base-url TEXT          API base URL
  --api-key TEXT           API key (or set OPENAI_API_KEY)
  --max-iterations INT     Maximum agent iterations (default: 20)
  --timeout INT            Execution timeout in seconds (default: 600)
  --output TEXT            Output directory (default: results)
  -v, --verbose            Verbose output
```

---

## Contributing

To add a new imaging task, follow the [New Task Guide](docs/NEW_TASK_GUIDE.md) and submit a pull request.
