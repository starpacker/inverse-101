# Approach: Reflection-Mode Optical Diffraction Tomography

## Problem Statement

Reconstruct the 3D refractive index (RI) distribution of a sample placed on a reflective substrate from intensity measurements under multiple illumination angles. This is reflection-mode Fourier ptychographic tomography (rMS-FPT).

## Physical Model

### Forward Model: rMSBP (Reflection Multi-Slice Beam Propagation)

For each illumination angle, the light propagates through the sample, reflects off the substrate, and propagates back:

1. **Incident field**: Tilted plane wave at angle θ_m = 2πm/n_angles
2. **Forward propagation** (downward through N layers):
   - For each layer i: propagate(dz_layer) → scatter(Δn_i) → propagate(dz_gap)
3. **Reflection**: field *= -1 (π phase shift from perfect mirror)
4. **Backward propagation** (upward, reversed layer order):
   - For each layer i: propagate(dz_gap) → propagate(dz_layer) → scatter(Δn_i)
5. **Detection**: back-propagate to focal plane → apply pupil → intensity = |field|²

### Why Three Steps per Layer (vs. Standard Two-Step BPM)

In transmission-mode ODT the 3D volume is uniformly sliced into contiguous voxels, giving the standard two-step BPM: `propagate(dz) → scatter(Δn)` per slice.

In reflection-mode, the sample is modeled as **thin scattering layers separated by homogeneous gaps** on a mirror:

```
Layer 0 ──gap── Layer 1 ──gap── Layer 2 ──gap── Layer 3 ──gap── Mirror
 0.5μm   10μm    0.5μm   10μm    0.5μm   10μm    0.5μm   10μm
```

Each layer therefore requires three operations: (1) propagate through the layer thickness, (2) apply the scattering phase from the RI contrast, (3) propagate through the gap to the next layer. The gap (`dz_gap=10μm`) is 20× the layer thickness (`dz_layer=0.5μm`), reflecting thin etched structures on spacer layers. Setting `dz_gap=0` recovers the standard two-step BPM. In reflection geometry light traverses every gap twice (down and back up), so the gap propagation has a significant cumulative effect.

### BPM Operators

**Propagation operator P(dz)** (Fourier domain):
```
a = fft2(u)
a *= exp(i·kz·dz) · eva_mask
u = ifft2(a)
```
where kz = sqrt(1 - fx² - fy²) · 2π·res_z

**Scattering operator Q(Δn, dz)** (real space):
```
u *= exp(i · Δn · 2π·res_z/n0 · dz)
```

## Inverse Problem

### Objective Function

Minimize amplitude MSE with 2D TV regularization:
```
L = Σ_m ‖√I_pred^m - √I_meas^m‖² + τ · Σ_z TV_2D(Δn_z)
```

### Solver: FISTA

Fast Iterative Shrinkage-Thresholding Algorithm with:
- Gradient computed via PyTorch autograd through full forward model
- TV proximal operator (Chambolle dual projection, per-slice 2D)
- FISTA momentum: β = (q(k)-1)/q(k+1), q(k) = (1+√(1+4q(k-1)²))/2

**Algorithm**:
```
Initialize: dn = 0, x_prev = 0
For k = 1 to n_iter:
    1. Compute gradient of amplitude MSE (sum over all angles)
    2. Gradient step: x_k = dn - lr · grad
    3. TV proximal: x_k = prox_{lr·τ·TV}(x_k)
    4. FISTA momentum: dn = x_k + β·(x_k - x_prev)
    5. Update: x_prev = x_k
```

## Implementation Strategy

1. **Preprocessing**: Generate 4-layer USAF-like synthetic phantom (Δn = -0.07)
2. **Forward model**: PyTorch implementation of rMSBP with GPU acceleration
3. **Reconstruction**: FISTA with lr=5.0, tv_weight=8e-7, 50 iterations
4. **Evaluation**: NRMSE, NCC, SSIM metrics

## Reference

Zhu et al., "Reflection-mode multi-slice Fourier ptychographic tomography", arXiv:2503.12246 (2025)
