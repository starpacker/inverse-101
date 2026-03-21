# EHT Black Hole Imaging

> Recover the radio image of a supermassive black hole from sparse, noisy
> interferometric measurements collected by a global array of telescopes.

**Domain:** Astronomy
**Modality:** Radio interferometry (VLBI)
**Difficulty:** Medium

---

## Background

The Event Horizon Telescope (EHT) is a Very Long Baseline Interferometry (VLBI) array
that links radio dishes across the globe — from Chile to the South Pole — forming a
virtual Earth-sized telescope at 230 GHz (1.3 mm wavelength) with ~20 microarcsecond
angular resolution.

Each pair of telescopes measures one **complex visibility** — a Fourier coefficient
of the sky brightness distribution. The fundamental challenge: **only ~0.01% of the
Fourier plane is sampled**, and the measurements are corrupted by thermal noise.
Recovering the image requires solving a severely ill-posed inverse problem.

---

## Physical Model

The van Cittert–Zernike theorem relates the measured visibility to the sky brightness:

$$V(u, v) = \iint I(l, m)\, e^{-2\pi i (ul + vm)}\, dl\, dm$$

Discretized on an N×N pixel grid:

$$\mathbf{y} = \mathbf{A}\,\mathbf{x} + \mathbf{n}$$

| Symbol | Description | Size |
|--------|-------------|------|
| **x** | Sky brightness image (vectorized, non-negative) | N² = 4096 |
| **A** | Measurement matrix (DFT at sampled baselines) | M × N² |
| **y** | Complex visibilities | M = 540 |
| **n** | Complex Gaussian thermal noise, CN(0, σ²I) | M |

The system is severely underdetermined: 540 measurements for 4096 unknowns.

---

## Data Description

### `data/raw_data.npz`

NumPy NPZ archive containing:
- `vis_noisy` — complex128 array, shape (540,): noisy complex visibilities
- `uv_coords` — float64 array, shape (540, 2): baseline coordinates in wavelengths

### `data/meta_data`

JSON file with imaging parameters:
- `N` = 64: image size (64×64 pixels)
- `pixel_size_uas` = 2.0: pixel size in microarcseconds
- `pixel_size_rad`: pixel size in radians
- `noise_std`: noise standard deviation σ
- `freq_ghz` = 230.0: observing frequency
- `n_baselines` = 540: number of measured baselines
- `snr` = 20.0: signal-to-noise ratio

### Data Generation

The observation is synthesized from a ground-truth M87\*-like ring image (Gaussian
ring with Doppler-boosted asymmetry) observed by 9 EHT stations over a 6-hour track
via Earth-rotation aperture synthesis.

---

## Method Hint

The standard approach is **Regularized Maximum Likelihood (RML)**:

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \geq 0}\; \frac{1}{2\sigma^2}\|\mathbf{A}\mathbf{x} - \mathbf{y}\|^2 + \lambda\, \mathcal{R}(\mathbf{x})$$

Common regularizers:
- **Total Variation (TV):** promotes piecewise smooth images with sharp edges
- **Maximum Entropy (MEM):** promotes smooth, diffuse emission
- **L1 Sparsity:** promotes compact emission

The optimization is solved with L-BFGS-B with positivity constraints (x ≥ 0).

For comparison, also implement:
- **Dirty Image:** baseline back-projection (no deconvolution)
- **CLEAN:** iterative deconvolution (standard in radio astronomy, but performs poorly with EHT's sparse uv-coverage without a support window constraint)

---

## References

- Högbom, J.A. (1974). *Aperture Synthesis with a Non-Regular Distribution of Interferometer Baselines*. A&AS, 15, 417.
- Thompson, A.R., Moran, J.M., & Swenson, G.W. (2017). *Interferometry and Synthesis in Radio Astronomy* (3rd ed.). Springer.
- EHT Collaboration et al. (2019). *First M87 Event Horizon Telescope Results. Paper I–VI*. ApJL, 875.
- Chael, A.A. et al. (2018). *Interferometric Imaging Directly with Closure Phases and Closure Amplitudes*. ApJ, 857, 23.
