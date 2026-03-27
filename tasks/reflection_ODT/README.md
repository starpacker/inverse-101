# Reflection-Mode Optical Diffraction Tomography (rMS-FPT)

## Overview

This task implements reflection-mode multi-slice Fourier ptychographic tomography (rMS-FPT) for 3D refractive index reconstruction of samples on reflective substrates.

**Reference**: Zhu et al., "Reflection-mode multi-slice Fourier ptychographic tomography", arXiv:2503.12246 (2025)

**GitHub**: https://github.com/bu-cisl/rMSBP-python

## Problem Description

Reconstruct the 3D refractive index (RI) distribution Δn(x,y,z) from intensity measurements under multiple illumination angles. The sample is placed on a reflective substrate (mirror), and light propagates through the sample, reflects, and propagates back before detection.

### Physics

- **Forward model**: Reflection-mode multi-slice beam propagation method (rMSBP)
- **Geometry**: Sample on mirror substrate, illumination from above
- **Measurements**: Intensity images at multiple illumination angles (16 angles, NA=0.28)
- **Sample**: 4-layer phantom with USAF-like patterns, Δn = -0.07

### Sample Geometry: Discrete Layers with Gaps

Unlike transmission-mode ODT where the 3D volume is uniformly sliced into contiguous voxels (standard two-step BPM: propagate → scatter per slice), the reflection-mode geometry models **thin scattering layers separated by homogeneous gaps** on a mirror substrate:

```
            ← illumination
Layer 0 ──gap── Layer 1 ──gap── Layer 2 ──gap── Layer 3 ──gap── Mirror
 0.5μm   10μm    0.5μm   10μm    0.5μm   10μm    0.5μm   10μm
```

This leads to a **three-step BPM** per layer in the forward pass:

| Step | Operation | Physical meaning |
|------|-----------|-----------------|
| `propagate(dz_layer)` | Free-space propagation | Traverse the layer thickness |
| `scatter(Δn, dz_layer)` | Phase accumulation | RI contrast within the layer |
| `propagate(dz_gap)` | Free-space propagation | Traverse gap to the next layer |

The gap distance (`dz_gap=10μm`) is 20× the layer thickness (`dz_layer=0.5μm`), reflecting the physical scenario of thin etched patterns on spacer layers. If `dz_gap=0`, this reduces to the standard two-step BPM. In reflection geometry, light traverses every gap twice (down and back up), so the gap propagation has a significant cumulative effect on the measured intensity.

### Method

- **Inverse solver**: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
- **Regularization**: 2D total variation (per-slice)
- **Loss**: Amplitude MSE (||√I_pred - √I_meas||²)

## Data

### Synthetic Phantom

- **Volume**: 4 layers × 128×128 pixels
- **Voxel size**: 0.274 × 0.274 × 1.0 μm
- **Background RI**: n0 = 1.54
- **RI contrast**: Δn = -0.07 (structures darker than background)
- **Patterns**: Layer 0 (vertical bars), Layer 1 (horizontal bars), Layer 2 (diagonal bars), Layer 3 (concentric rings)

### Metadata (`data/meta_data`)

```json
{
    "volume_shape": [4, 128, 128],
    "res_um": [0.274, 0.274, 1.0],
    "wavelength_um": 0.632,
    "n0": 1.54,
    "NA_obj": 0.28,
    "NA_illu": 0.28,
    "n_angles": 16,
    "dz_layer": 0.5,
    "dz_gap": 10.0,
    "ri_contrast": -0.07
}
```

## Usage

```bash
cd tasks/reflection_ODT
pip install -r requirements.txt
python main.py
```

## Expected Results

- **NRMSE**: ~0.65
- **NCC**: ~0.88
- **SSIM**: ~0.15
- **Loss convergence**: 0.5 → 0.0005 (50 iterations)

## Files

- `main.py` — Full reconstruction pipeline
- `src/preprocessing.py` — Phantom generation
- `src/physics_model.py` — rMSBP forward model (PyTorch)
- `src/solvers.py` — FISTA + TV reconstruction
- `src/visualization.py` — Plotting and metrics
- `notebooks/reflection_ODT.ipynb` — Tutorial notebook
- `evaluation/reference_outputs/` — Precomputed results
