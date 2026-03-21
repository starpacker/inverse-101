# Task 2: EHT Black Hole Dynamic Reconstruction (StarWarps)

## Paper
Bouman et al. 2017, "Reconstructing Video from Interferometric Measurements of Time-Varying Sources", arXiv:1711.01357v2

## Goal
Reproduce Figures 10 and 14 from the paper. The key contribution is the StarWarps algorithm: reconstructing VIDEO from interferometric measurements of a time-varying source using a Gaussian Markov Model that links frames temporally.

## Reference Code
GitHub: https://github.com/achael/eht-imaging
The dynamic imaging functionality was integrated into ehtim. Clone and study the relevant code. Do NOT use ehtim as an imported package.

## Key Concepts from the Paper
1. **Time-varying source model**: Image changes over observation time (e.g., SgrA* orbiting hotspot)
2. **Gaussian Markov Model**: Prior that links consecutive frames: p(x_t | x_{t-1}) ~ N(x_{t-1}, Λ_flow)
3. **Data consistency**: χ(x, y) formulation supporting visibilities, closure phases, visibility amplitudes
4. **Multivariate Gaussian image prior**: N_x(μ, Λ) with power-law covariance Λ where power spectrum ~ 1/(u²+v²)^{a/2}
5. **StarWarps EM algorithm**:
   - E-step: Compute posterior mean and covariance for each frame using forward-backward message passing
   - M-step: Update hyperparameters (flow covariance, image prior)
6. **Earth rotation synthesis**: Different uv-coverage at each time step as baselines rotate

## What the Task Should Implement
1. **preprocessing.py**: Load time-series observation data, organize by time frames, compute data products per frame
2. **physics_model.py**:
   - Per-frame DFT measurement matrix (uv-coverage varies with time)
   - Support for multiple data products: complex visibilities, bispectrum/closure phases, visibility amplitudes
   - Time-varying forward model: image sequence → measurement sequence
3. **solvers.py**:
   - **Static MAP solver** (baseline): Single-frame RML with Gaussian prior (Section IV of paper)
   - **StarWarps dynamic solver**: EM algorithm with Gaussian Markov temporal model (Section V-VI)
     - Forward-backward message passing for posterior inference
     - Multivariate Gaussian prior with power-law covariance
     - Temporal flow regularization between consecutive frames
   - **Simple sliding-window baseline**: Reconstruct each frame independently using data within a time window
4. **visualization.py**: Plot video frames, temporal evolution, uv-coverage per frame, metrics over time
5. **generate_data.py**: Generate synthetic time-varying source:
   - Orbiting hotspot model around black hole (Keplerian motion)
   - Or rotating/evolving crescent model
   - Generate time-stamped visibilities with evolving uv-coverage
   - Include realistic thermal noise

## Figures to Reproduce
- **Figure 10**: Dynamic reconstruction results on simulated data — compare StarWarps vs static methods vs sliding window. Show multiple frames of reconstructed video alongside ground truth.
- **Figure 14**: Application to real VLBA data of M87 jet (or simulated equivalent showing jet-like dynamic structure)

If real data not available, generate synthetic data demonstrating dynamic reconstruction superiority over static methods.

## Data Generation
- Ground truth: sequence of ~10-20 frames showing evolving emission (orbiting hotspot or evolving crescent)
- EHT-like array with 6-9 stations
- Time-stamped visibilities: each frame has different uv-coverage due to Earth rotation
- Thermal noise with realistic SNR
- Save as time-indexed arrays

## Format
Follow EXACTLY the pilot task format (same as Task 1). Directory: tasks/eht_black_hole_dynamic/

## Critical Constraints
- Do NOT import or depend on ehtim package at runtime. All necessary functions must be self-contained in src/.
- You SHOULD clone the eht-imaging repo to /tmp/ and extract/copy the relevant StarWarps and dynamic imaging code. Copying code from the reference repo is encouraged — the goal is to "clean" the original implementation into our standardized format, not to rewrite from scratch.
- Adapt the extracted code to fit our directory structure and function signatures, remove unnecessary dependencies, but preserve the original algorithm logic as faithfully as possible.
- The StarWarps EM algorithm is the core — implement the full forward-backward message passing.
- The Gaussian prior covariance Λ with power-law spectrum is essential (not just diagonal).
- Must demonstrate that dynamic method outperforms static frame-by-frame reconstruction.
- Each function independently testable with fixtures.
- Tests for stochastic functions: statistical property checks only.
- Keep existing PDFs and reference_website_github.md.
- Run `python -m pytest evaluation/tests/ -v` at the end.
