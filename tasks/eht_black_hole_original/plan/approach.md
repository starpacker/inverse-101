# Approach

## Problem Statement

Recover a 64×64 radio image of a black hole from sparse, noisy, and gain-corrupted interferometric measurements using only closure quantities (closure phases and closure amplitudes), which are robust to station-based calibration errors.

## Mathematical Formulation

The forward model follows the van Cittert–Zernike theorem:

$$\mathbf{y} = \mathbf{A}\,\mathbf{x} + \mathbf{n}$$

But in practice, station-based gains corrupt the visibilities:

$$V_{ij}^{\text{obs}} = g_i \, g_j^* \, V_{ij}^{\text{true}} + n_{ij}$$

**Closure quantities** are formed from combinations of visibilities where gains cancel:

- **Closure phase** (triangle i,j,k): φ_ijk = arg(V_ij · V_jk · V_ki). Since |g_i|²|g_j|²|g_k|² is real-positive, arg(B^obs) = arg(B^true).
- **Closure amplitude** (quadrangle i,j,k,l): CA = |V_ij·V_kl|/|V_ik·V_jl|. Gain amplitudes cancel in the ratio.

## Solution Strategy

### Step 1: Data Preprocessing

Load corrupted visibilities, station IDs, and noise estimates. Compute closure phases on all triangles, log closure amplitudes on all quadrangles, and their noise standard deviations via error propagation (Eqs. 11–12, Chael 2018).

### Step 2: Forward Model Construction

Build the DFT measurement matrix A. Extend it with:
- Closure phase operator: image → visibilities → bispectrum → closure phases
- Closure amplitude operator: image → visibilities → log closure amplitudes
- Analytic gradients of closure χ² w.r.t. image pixels

### Step 3: Image Reconstruction

Apply four methods:

1. **Closure Phase Only (TV)**: Uses only closure phase χ² + TV regularizer. Robust to both amplitude and phase gain errors.

2. **Closure Phase + Amplitude (TV)**: Uses both closure phase and log closure amplitude χ² + TV. Most complete closure-only method.

3. **Closure Phase + Amplitude (MEM)**: Same data terms with maximum entropy regularizer. Promotes smooth emission.

4. **Visibility RML (TV)**: Traditional approach using corrupted visibilities. Should fail due to gain errors — included as a comparison baseline.

### Step 4: Evaluation

Compare against ground truth using NRMSE, NCC, dynamic range. The key result: closure-only methods produce good reconstructions (NRMSE < 0.7, NCC > 0.7) while visibility-based reconstruction fails (NRMSE > 0.8) because it cannot account for the unknown gains.

### Step 5: Visualization

Generate comparison panels (Figure 4 equivalent) and closure quantity plots.

## Expected Results

| Method              | NRMSE  | NCC    |
|---------------------|--------|--------|
| CP-only (TV)        | ~0.65  | ~0.75  |
| CP+CA (TV)          | ~0.55  | ~0.85  |
| CP+CA (MEM)         | ~0.55  | ~0.85  |
| Visibility (TV)     | >0.80  | <0.60  |

Closure-only methods significantly outperform visibility-based imaging when station gains are corrupted. Adding closure amplitudes improves over closure-phase-only imaging by constraining the source size and flux distribution.
