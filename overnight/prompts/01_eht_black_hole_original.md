# Task 1: EHT Black Hole Static Reconstruction (Closure-Only Imaging)

## Paper
Chael et al. 2018, "Interferometric Imaging Directly with Closure Phases and Closure Amplitudes", ApJ 857:23

## Goal
Reproduce Figures 4, 5, 6, 7 from the paper. The key contribution is imaging using ONLY closure phases and closure amplitudes (not calibrated complex visibilities), which are robust to station-based calibration errors.

## Reference Code
GitHub: https://github.com/achael/eht-imaging
Clone this repo to a temporary directory and study the relevant code. Do NOT use ehtim as an imported package in the final task code. Instead, extract and rewrite the necessary functions as self-contained code.

## Key Functions to Extract from eht-imaging
1. **Closure phase computation**: From complex visibilities on triangles of baselines → bispectrum → closure phase. Formula: V_B = V_12 * V_23 * V_31, closure phase = arg(V_B)
2. **Closure amplitude computation**: From visibility amplitudes on quadrangles of 4 stations. Formula: |V_C|_a = |V_12 * V_34| / |V_13 * V_24|
3. **Closure-only χ² data terms**: Chi-squared for closure phases (Eq. 11 in paper) and log closure amplitudes (Eq. 12 in paper)
4. **Analytic gradients** of closure χ² terms w.r.t. image pixels
5. **RML imaging with closure quantities**: The optimization framework that minimizes closure χ² + regularizer (TV, MEM, L1)
6. **Noise on closure quantities**: σ_ψ for closure phase (Eq. 11), σ_C for closure amplitude (Eq. 12)
7. **Amplitude debiasing**: |V|_debiased = sqrt(|V|²_meas - σ²) (Eq. 9)

## What the Task Should Implement
1. **preprocessing.py**: Load observation data (visibilities, uv-coords, station info), compute closure phases and closure amplitudes from complex visibilities, compute their noise statistics
2. **physics_model.py**: Forward model that maps image → complex visibilities → closure phases and closure amplitudes. Must include:
   - DFT measurement matrix (image → visibilities)
   - Closure phase operator (visibilities → closure phases on all independent triangles)
   - Closure amplitude operator (visibilities → closure amplitudes on all independent quadrangles)
   - Analytic gradients of closure quantities w.r.t. image
3. **solvers.py**:
   - RML solver using closure-only χ² (NOT visibility χ²)
   - Support both closure phase only and closure phase + closure amplitude imaging
   - TV, MEM, L1 regularizers
   - L-BFGS-B optimization with positivity constraint
4. **visualization.py**: Plot closure phases vs time/baseline, reconstructed images, comparison panels, metrics
5. **generate_data.py**: Generate synthetic EHT observation with realistic station gains and phase errors, so that closure quantities differ from calibrated visibilities

## Figures to Reproduce
- **Figure 4**: Reconstruction comparison on simulated data — show that closure-only imaging is robust to gain errors while traditional (visibility) imaging fails
- **Figure 5**: Effect of systematic amplitude errors — closure-only results are independent of gain error level
- **Figure 6**: Application to VLBA data (or simulated equivalent)
- **Figure 7**: Application to ALMA data (or simulated equivalent)

For Figures 6 and 7, if real data is not available, generate realistic synthetic data that demonstrates the same principle: closure-only imaging produces correct results even when station gains are corrupted.

## Data Generation
Generate synthetic EHT observation data:
- 8-9 station array (approximate EHT 2017 configuration)
- Ground truth: crescent/ring model similar to M87*
- Add realistic station-based gain errors (amplitude: 10-50%, phase: random)
- Compute: complex visibilities, closure phases, closure amplitudes, and their noise estimates
- Save both corrupted visibilities AND closure quantities

## Format
Follow EXACTLY the pilot task format in tasks/eht_black_hole/:
```
tasks/eht_black_hole_original/
├── README.md
├── requirements.txt
├── main.py
├── data/
│   ├── raw_data.npz
│   └── meta_data
├── plan/
│   ├── approach.md
│   └── design.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── physics_model.py
│   ├── solvers.py
│   ├── visualization.py
│   └── generate_data.py
├── notebooks/
│   └── eht_black_hole_original.ipynb
└── evaluation/
    ├── reference_outputs/
    │   ├── ground_truth.npy
    │   ├── metrics.json
    │   └── ...
    ├── fixtures/
    │   ├── preprocessing/*.npz
    │   ├── physics_model/*.npz
    │   ├── solvers/*.npz
    │   └── visualization/*.npz
    └── tests/
        ├── test_preprocessing.py
        ├── test_physics_model.py
        ├── test_solvers.py
        ├── test_visualization.py
        └── test_end_to_end.py
```

## Critical Constraints
- Do NOT import or depend on ehtim package at runtime. All necessary functions must be self-contained in src/.
- You SHOULD clone the eht-imaging repo to /tmp/ and extract/copy the relevant code. Copying code from the reference repo is encouraged — the goal is to "clean" the original implementation into our standardized format, not to rewrite from scratch.
- Adapt the extracted code to fit our directory structure and function signatures, remove unnecessary dependencies, but preserve the original algorithm logic as faithfully as possible.
- Each function must be independently testable with fixtures.
- Tests for stochastic functions must use statistical property checks, not seed-matching.
- Keep existing PDFs and reference_website_github.md in the task folder.
- Run `python -m pytest evaluation/tests/ -v` at the end and fix any failures.
