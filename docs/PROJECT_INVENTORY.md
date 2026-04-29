# imaging-101 Project Inventory (2026-04-29)

> Full survey of the project state across local codebase, GitHub, HuggingFace, website, and evaluation results.

---

## 1. Repositories & Hosting

| Asset | URL | Status |
|-------|-----|--------|
| **Upstream GitHub** | `github.com/HeSunPU/imaging-101` | Public, 3 stars, 11 open PRs, last push 2026-04-17 |
| **Backup fork** | `github.com/starpacker/inverse-101` | Git remote `backup` in local repo |
| **Website repo** | `github.com/starpacker/agent-imaging-website` | Next.js + TypeScript + Tailwind; marked "[MOVED] to starpacker/inverse_benchmark/website/" |
| **HuggingFace dataset** | `huggingface.co/datasets/AI4Imaging/imaging-101` | Private dataset; stores fixture files from `tasks/**/fixtures/**` |
| **Local repo** | `C:\Users\30670\Desktop\eny\imaging-101` | Main working copy; `origin` = HeSunPU, `backup` = starpacker |

---

## 2. Task Inventory (57 tasks, 10 domains)

### Domain Breakdown

| Domain | # Tasks | Tasks |
|--------|---------|-------|
| **Medicine** | 22 | ct_fan_beam, ct_sparse_view, ct_poisson_lowdose, ct_dual_energy, xray_tooth_gridrec, mri_l1_wavelet, mri_tv, mri_t2_mapping, mri_sense, mri_grappa, mri_noncartesian_cs, diffusion_mri_dti, mri_dynamic_dce, mri_pnp_admm, mri_varnet, pnp_mri_reconstruction, plane_wave_ultrasound, ultrasound_sos_tomography, usct_FWI, pet_mlem, photoacoustic_tomography, eit_conductivity_reconstruction |
| **Astronomy** | 9 | eht_black_hole_original, eht_black_hole_dynamic, eht_black_hole_UQ, eht_black_hole_feature_extraction_dynamic, eht_black_hole_tomography, lucky_imaging, exoplanet_imaging, shack-hartmann, shapelet_source_reconstruction |
| **Biology / Microscopy** | 9 | SSNP_ODT, reflection_ODT, fourier_ptychography, microscope_denoising, hessian_sim, light_field_microscope, single_molecule_light_field, fpm_inr_reconstruction, s2ism |
| **Earth Science** | 6 | seismic_FWI_original, seismic_lsrtm_original, seismic_traveltime_tomography, insar_phase_unwrapping, weather_radar_data_assimilation, era5_tensorvar |
| **Physics / Optics** | 6 | conventional_ptychography, electron_ptychography, confocal-nlos-fk, lensless_imaging, differentiable_deflectometry, spectral_snapshot_compressive_imaging |
| **Chemistry** | 5 | mcr_hyperspectral, raman_cell_phenotyping, cars_spectroscopy, xray_ptychography_tike, xray_laminography_tike |

### Task Completeness

All 57 tasks have: `main.py`, `data/`, `src/`, `evaluation/`, `requirements.txt` (100% execution-ready).

56/57 tasks have: `README.md`, `plan/` (approach.md + design.md), `notebooks/`, `evaluation/tests/`, `evaluation/reference_outputs/`.

**1 incomplete task:** `hessian_sim` -- missing README.md, plan/, notebooks/, reference_outputs/.

### Tasks with unit tests (for function-mode evaluation): **43+**

Tasks **without** unit tests (end-to-end only): eht_black_hole_feature_extraction_dynamic, SSNP_ODT, hessian_sim, reflection_ODT, single_molecule_light_field, conventional_ptychography, exoplanet_imaging, lucky_imaging, lensless_imaging, shack-hartmann (uses metrics.json instead).

---

## 3. Evaluation Harness

### Architecture

```
evaluation_harness/
  __main__.py       CLI: run | prepare | collect | summarize
  runner.py         BenchmarkRunner orchestrator
  scorer.py         pytest runner + NCC/NRMSE quality metrics
  config.py         LLMConfig / TaskConfig / RunConfig
  agent.py          ReAct agent loop (Thought -> Action -> Observation)
  multi_agent.py    Pipeline: Planner -> Critic -> Architect -> Coder -> Judge
  llm_client.py     OpenAI-compatible API client
  prompts.py        Prompt templates per mode/level
  plan_scorer.py    LLM-as-judge plan evaluation
  visualizer.py     Comparison figure generation
  local_runner.py   Local sandbox backend
  docker_runner.py  Docker sandbox backend
  agents/           Specialized agent roles (planner, architect, coder, judge)
  frameworks/       Pluggable frameworks (react, multi_agent, claude_code, deepcode)
```

### Evaluation Modes

| Mode | What it tests | Metrics |
|------|--------------|---------|
| **Function** (`--mode function`) | Individual module implementations (physics_model, preprocessing, solvers, etc.) | Per-module test pass rate |
| **End-to-End** (`--mode end_to_end`) | Full pipeline from scratch | NCC, NRMSE reconstruction quality |
| **Plan** (`--mode plan`) | Planning ability only | LLM-as-judge pairwise + rubric scoring |

### Difficulty Levels (end-to-end)

| Level | Agent receives |
|-------|---------------|
| L1 | Task README only |
| L2 | README + approach.md |
| L3 | README + approach.md + design.md |

### Agent Frameworks

| Framework | Description |
|-----------|-------------|
| **ReAct** | Single-agent Thought -> Action -> Observation loop |
| **Multi-Agent** | Pipeline: Planner -> Critic -> Architect -> Coder -> Judge |
| **Copilot (Claude Code)** | Third-party agent: prepare sandbox + prompt, agent works externally |
| **DeepCode** | HKUDS DeepCode autonomous multi-agent |

### Dependencies

Minimal: only `requests>=2.28` + Python stdlib.

---

## 4. Models Evaluated

7 models configured in `config_llm.yaml` (all via dp.tech gateway):

| Model | Pass Rate (Function) | Fully Passed Tasks |
|-------|---------------------|--------------------|
| **Gemini-3.1-Pro** | **93.3%** (1664/1783) | **24** |
| **Claude-4.6-Opus** | **93.7%** (1410/1505) | **21** |
| **GPT-5.4** | **92.2%** (1440/1562) | **18** |
| **GLM-5** | **89.5%** (1511/1689) | **15** |
| **DeepSeek-V3** | **86.5%** (1345/1555) | **15** |
| **Qwen3.6-Plus** | **84.6%** (1301/1537) | **12** |
| **Kimi-K2.5** | **84.3%** (1390/1649) | **11** |

---

## 5. Results Inventory

### 5.1 Function Mode Results

| Location | Tasks | Models | Notes |
|----------|-------|--------|-------|
| `results/function_mode/` | 41 | Claude, Gemini (early tests) | Original runs |
| `results/function_mode_3/` | 15 | 3+ models each | Subset re-runs |
| `results/function_mode_total/` | **51 tasks, 465 model-task combos** | All 7 models | **Primary function-mode data** |
| `results/function_mode_server/` | 52 | Multiple | Server-based evaluation |

### 5.2 End-to-End Results

| Framework | Location | Tasks | Model |
|-----------|----------|-------|-------|
| **Copilot (Claude Code)** | `results/end_to_end/copilot/` | 21 JSONs | claude_code, L1 |
| **Multi-Agent** | `results/end_to_end/multi_agent/` | 26 JSONs (16 unique tasks) | Claude-4.6-opus, L1+L3 |
| **ReAct** | `results/end_to_end/react/` | 2 JSONs | Claude-4.6-opus |
| **Claude Code (expanded)** | `results/end_to_end/claude_code/` | 50+ task dirs + report + trajectories | claude_code |
| **Server e2e** | `results/audit/e2e_audit.csv` | 57 tasks x 7 models (many MISSING) | All 7 |

### 5.3 Claude Code End-to-End Quality (from `claude_code_audit.csv`)

58 tasks scored. Highlights:

| Verdict | Count | Examples |
|---------|-------|---------|
| **Pass** (NCC > 0.8, NRMSE < 0.5) | ~28 | cars_spectroscopy (NCC 1.0), microscope_denoising (NCC 0.9999), s2ism (NCC 0.9999), seismic_FWI (NCC 0.9992) |
| **Partial** | ~3 | eht_black_hole_UQ (NCC 0.735), lensless_imaging (NCC 0.896), photoacoustic_tomography (NCC 0.809) |
| **Fail** | ~27 | Shape mismatches, no reference found, poor reconstruction quality |

### 5.4 Audit CSVs

| File | Content | Rows |
|------|---------|------|
| `results/audit/function_mode_audit.csv` | 57 tasks x 7 models, per-module test pass rates | 406 |
| `results/audit/e2e_audit.csv` | 57 tasks x 7 models, NCC/NRMSE/verdict | 406 |
| `results/audit/claude_code_audit.csv` | 58 tasks, Claude Code e2e quality | 58 |

### 5.5 Other Result Artifacts

| Artifact | Description |
|----------|-------------|
| `results/code_archive/` | 122 timestamped workspace snapshots (plan + src code) |
| `results/claude_code_batch_summary_20260409.md` | 21-task copilot batch summary |
| `results/claude_code_visualization_20260409.md` | 21-task visual comparison panels |
| `results/end_to_end/claude_code/REPORT.md` | Claude Code comprehensive analysis |
| `results/end_to_end/claude_code/claude_code_trajectories/` | Agent interaction traces |
| `figs/appC/` | 191 comparison figures (baseline/claude/gt/preview per task) |
| `function_mode_success_summary.md` | Per-model detailed pass/fail breakdown |
| `claude_opus_error_analysis.md` | Error taxonomy for Claude Opus failures |

---

## 6. Website (`starpacker/agent-imaging-website`)

**Stack:** Next.js + TypeScript + Tailwind CSS

**Contents:**
- `public/data/notebooks/` -- 57 JSON files (one per task) for interactive notebook display
- `public/data/tasks_db.json` -- Central database of all tasks (49.8 KB)
- `public/images/` -- Input/reconstruction comparison images per task
- `src/components/` -- TaskCard, TaskGrid, NotebookViewer, ComparisonView
- `src/app/` -- Pages: home, task detail, about, notebook viewer
- `.github/workflows/deploy.yml` -- CI/CD pipeline

**Note:** Repo description says "[MOVED] to starpacker/inverse_benchmark/website/" but still active (last push 2026-04-20).

---

## 7. HuggingFace Dataset (`AI4Imaging/imaging-101`)

- **Status:** Private dataset
- **Access token:** (stored locally, not committed)
- **Contents:** Fixture files from `tasks/**/fixtures/**` (test data/reference outputs too large for GitHub)
- **Upload script:** `archive/old_scripts/huggingface_upload.py`

---

## 8. Scripts & Tooling

### Batch Evaluation Scripts (`scripts/`)

| Script Pattern | Description |
|---------------|-------------|
| `run_15hard_<model>.sh` (x7) | 15 GPU/hard tasks per model |
| `run_35rest_<model>.sh` (x7) | 35 remaining tasks per model |
| `run_e2e_57tasks_3models.sh` | All 57 tasks e2e for 3 models |
| `run_missing8_all_models.sh` | Gap-fill for 8 incomplete tasks |
| `run_parallel_all.sh` | Orchestrate parallel model runs |

### Analysis & Consolidation Scripts

| Script | Description |
|--------|-------------|
| `consolidate_results.py` | Aggregate function-mode results to summary CSV |
| `consolidate_e2e_results.py` | Consolidate end-to-end results |
| `audit_all_results.py` | Pre-indexed audit of all results |
| `compare_frameworks.py` | ReAct vs Multi-Agent comparison |
| `compute_metrics.py` | NCC/NRMSE computation |
| `score_claude_code.py` | Score 57 Claude Code e2e results |
| `rescore_existing.py` | Re-score without re-running agents |
| `get_e2e_iters.py` | Iteration statistics for passing tasks |

### Paper/Overleaf Support Scripts (root level)

| Script | Description |
|--------|-------------|
| `_build_appc_figures.py` | Generate Appendix C comparison figures |
| `_gen_appC_overleaf.py` | Generate LaTeX for Appendix C |
| `_fill_appc_metrics.py` | Fill metrics into LaTeX tables |
| `_expand_appc_tex.py` | Expand Appendix C TeX macros |
| `_inventory_tasks.py` | Inventory all tasks for paper |
| `_appc_inventory.json` | Task inventory data (76 KB) |

---

## 9. Disk Usage

| Directory | Size |
|-----------|------|
| `tasks/` | 2.1 GB |
| `results/` | 2.1 GB |
| `archive/` | 338 MB |
| `figs/` | 30 MB |
| `evaluation_harness/` | 978 KB |
| `scripts/` | 264 KB |
| `tasks.zip` | 1.1 GB |
| **Total local footprint** | ~5.7 GB |

---

## 10. Key Differences: Local vs Upstream

**Local has extensive additions not in upstream:**
- Full `results/` directory with all evaluation data (2.1 GB)
- `figs/appC/` with 191 paper figures
- `function_mode_success_summary.md` and `claude_opus_error_analysis.md`
- `config_llm.yaml` with all 7 model configurations
- `scripts/` with batch runners and analysis tools
- Paper generation scripts (`_build_appc_figures.py`, etc.)
- `claude_canonical_runs.json`, `retry_eval_results.json`
- `analysis_segments/` batch analysis
- `claude_opus_retry/` retry data

**Upstream has items not in local (deleted locally):**
- `docs/COMPARISON_REPORT_GEMINI25PRO.md`
- `docs/NEXT_AGENT_PROMPT.md`, `docs/PROGRESS_REPORT.md`
- `overnight/` directory (batch scripts + logs)
- `scripts/run_comparison.sh`, `scripts/run_end2end_gemini.sh`, etc.
- Various `__pycache__` files

---

## 11. Open Issues / Gaps

1. **hessian_sim incomplete:** Missing README.md, plan/, reference_outputs/
2. **E2E results sparse for most models:** Only Claude Code has broad e2e coverage (50+ tasks); other models mostly MISSING in e2e_audit.csv
3. **Claude Code e2e failures:** 27/58 tasks failed (mostly shape mismatches or missing references)
4. **Website repo moved:** agent-imaging-website says moved to inverse_benchmark/website but isn't archived
5. **HuggingFace dataset private:** Need to decide if it should be public for release
6. **tasks.zip (1.1 GB) in repo root:** Should be .gitignored or removed before release
7. **API keys in config_llm.yaml and CLAUDE.md:** Must be stripped before public release
8. **__pycache__ directories** present in working tree

---

## 12. Summary for Release Readiness

### What's in good shape
- 57 tasks with standardized structure (56/57 fully documented)
- Evaluation harness: 4 frameworks, 3 modes, 3 difficulty levels
- Function-mode results: 7 models x 51 tasks = 465 combinations
- Claude Code e2e results: 50+ tasks with report and trajectories
- Paper support: AppC figures, LaTeX generation, metrics tables
- Website: functional Next.js showcase with notebook viewer

### What needs work for release
1. **Clean secrets** from config_llm.yaml, CLAUDE.md
2. **Remove large files** (tasks.zip, __pycache__, .pyc)
3. **Complete hessian_sim** task documentation
4. **Decide HuggingFace access level** (private -> public?)
5. **Consolidate website** (resolve MOVED status)
6. **Fill e2e evaluation gaps** for non-Claude models (if needed for paper)
7. **Audit which results to include** in public release vs keep private
