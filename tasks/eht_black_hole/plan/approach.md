# Approach

## Problem Statement

Recover a 64×64 radio image of a black hole from 540 sparse, noisy complex visibility measurements collected by the Event Horizon Telescope (EHT).

## Mathematical Formulation

The forward model follows the van Cittert–Zernike theorem. Each telescope pair measures one complex visibility — a Fourier coefficient of the sky brightness:

$$\mathbf{y} = \mathbf{A}\,\mathbf{x} + \mathbf{n}$$

- **x** ∈ R^{N²}: vectorized sky brightness (N=64, so 4096 unknowns)
- **A** ∈ C^{M×N²}: DFT-like measurement matrix evaluated at (u,v) baseline positions
- **y** ∈ C^M: 540 measured complex visibilities (M ≪ N²)
- **n** ~ CN(0, σ²I): complex Gaussian thermal noise (SNR ≈ 20)

The system is severely underdetermined (540 measurements, 4096 unknowns).

## Solution Strategy

### Step 1: Data Preprocessing

Load the observation data (`raw_data` NPZ containing visibilities and uv-coordinates) and imaging metadata (`meta_data` JSON with image size, pixel scale, noise level).

### Step 2: Forward Model Construction

Build the measurement matrix A by evaluating the DFT kernel at each (u,v) baseline position on the image pixel grid. This is a direct matrix approach (not FFT-based), suitable for small images (N ≤ 128).

Key design choice: omit the Δθ² solid-angle factor from A to avoid numerical underflow (Δθ ≈ 2 μas → Δθ² ≈ 10⁻²³ rad²). This is valid because the images are flux-normalized.

### Step 3: Image Reconstruction

Apply four methods of increasing sophistication:

1. **Dirty Image** (baseline): Back-projection x_dirty = A^H y / max(PSF). No deconvolution. Shows the raw imaging artifacts from sparse uv-coverage.

2. **CLEAN** (Högbom 1974): Iterative deconvolution assuming point-source sky model. For EHT's extremely sparse uv-coverage, a support window (radius ≈ 15 pixels) is mandatory because PSF sidelobes reach ~98% of the main lobe, causing divergence without spatial constraint.

3. **RML-TV**: Regularized Maximum Likelihood with Total Variation penalty:
   - min_{x≥0} ‖Ax−y‖²/(2σ²) + λ·TV(x)
   - λ_TV = 5×10³, solved with L-BFGS-B (500 iterations)
   - TV promotes piecewise-smooth images, good for ring-like structures

4. **RML-MEM**: Same framework with Maximum Entropy regularizer:
   - min_{x≥0} ‖Ax−y‖²/(2σ²) + λ·Σ x_i log(x_i/p_i)
   - λ_MEM = 1×10⁴, flat prior
   - MEM promotes smooth, diffuse emission

### Step 4: Evaluation

Compare reconstructions against ground truth using:
- **NRMSE**: ‖x̂−x‖₂ / ‖x‖₂ (lower is better)
- **NCC**: normalized cross-correlation (higher is better, max 1)
- **Dynamic Range**: peak / RMS(background) (higher means better sensitivity)

All metrics are computed after flux normalization (matching total flux to ground truth).

### Step 5: Visualization

Generate uv-coverage plot, side-by-side reconstruction comparison, and summary panel.

## Expected Results

| Method     | NRMSE  | NCC    |
|------------|--------|--------|
| Dirty Image| ~0.89  | ~0.50  |
| CLEAN      | poor   | poor   |
| RML-TV     | ~0.61  | ~0.82  |
| RML-MEM    | ~0.58  | ~0.87  |

RML methods significantly outperform CLEAN for EHT's sparse coverage. CLEAN performs poorly because its point-source assumption breaks down when the PSF sidelobes dominate. MEM slightly outperforms TV for this smooth ring-like morphology.
