# Task: EHT Black Hole Imaging

> Recover the radio image of a supermassive black hole from sparse, noisy
> interferometric measurements collected by a global array of telescopes.

---

## Background

In April 2019, the **Event Horizon Telescope (EHT) Collaboration** published
the first resolved image of a black hole — the supermassive black hole M87\*
at the center of galaxy Messier 87, 55 million light-years away. Three years
later in 2022, they imaged **Sgr A\***, the black hole at the center of our
own Milky Way.

The EHT is not a single telescope. It is a **Very Long Baseline Interferometry
(VLBI)** array that phase-coherently links radio dishes spread across the globe
— from Chile to Spain, from Hawaii to the South Pole — forming a virtual
telescope the size of Earth with angular resolution of ~20 microarcseconds (μas).

### Why is this an inverse problem?

A filled-aperture telescope (like a camera lens) directly focuses photons to form
an image. A VLBI array cannot do this. Instead, pairs of telescopes measure the
**complex cross-correlation** of the electric field — called a *visibility* —
at discrete spatial frequencies determined by the baseline between them.

The fundamental challenge: **only a tiny fraction of the Fourier plane (uv-plane)
is sampled**, and the measurements are corrupted by noise and instrumental effects.
Recovering the image requires solving an ill-posed inverse problem.

---

## Physical Model

### Van Cittert–Zernike Theorem

Under the assumptions of incoherence and small field of view, the measured
complex visibility $V(u, v)$ is related to the sky brightness $I(l, m)$ by:

$$V(u, v) = \iint I(l, m)\, e^{-2\pi i (ul + vm)}\, dl\, dm$$

where:
- $(l, m)$: direction cosines on the sky (angular coordinates, in radians)
- $(u, v)$: baseline coordinates **in units of wavelengths** (spatial frequencies)
- $V(u, v)$: complex visibility measured by a pair of telescopes

This is a **2D Fourier transform** — the visibilities are Fourier coefficients
of the sky brightness. Each telescope pair samples one point $(u, v)$ in the
Fourier domain, with Earth's rotation tracing out arcs over time.

### Discretized Measurement Model

On an $N \times N$ pixel grid with pixel size $\Delta\theta$:

$$\mathbf{y} = \mathbf{A}\,\mathbf{x} + \mathbf{n}$$

where:
- $\mathbf{x} \in \mathbb{R}^{N^2}$: vectorized image (sky brightness, non-negative)
- $\mathbf{A} \in \mathbb{C}^{M \times N^2}$: measurement matrix with entries
  $A_{kj} = e^{-2\pi i (u_k l_j + v_k m_j)} \cdot (\Delta\theta)^2$
- $\mathbf{y} \in \mathbb{C}^{M}$: measured complex visibilities ($M \ll N^2$)
- $\mathbf{n} \sim \mathcal{CN}(0, \sigma^2 \mathbf{I})$: thermal noise

The system is **severely underdetermined** ($M \approx 10^3$, $N^2 \approx 10^4$–$10^6$).

### Instrumental Effects (Real Data)

In practice, each telescope has complex antenna gains $g_i(t)$ that corrupt the
visibilities:

$$V_{ij}^{\text{obs}} = g_i(t)\, g_j^*(t)\, V_{ij}^{\text{true}} + n_{ij}$$

Calibrating these gains (self-calibration) is performed iteratively alongside imaging.
For this benchmark, we work with **pre-calibrated, gain-corrected visibilities**.

---

## Inverse Problem Formulation

Given noisy measurements $\mathbf{y}$ and knowledge of the sampling positions
$(u_k, v_k)$, recover $\mathbf{x} \geq 0$.

### Data Fidelity (Chi-squared)

$$\chi^2(\mathbf{x}) = \frac{1}{2\sigma^2} \|\mathbf{A}\mathbf{x} - \mathbf{y}\|_2^2$$

### Regularized Maximum Likelihood

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \geq 0}\; \chi^2(\mathbf{x}) + \sum_r \lambda_r\, \mathcal{R}_r(\mathbf{x})$$

Common regularizers:
- **Total Variation (TV):** $\mathcal{R}(\mathbf{x}) = \|\nabla \mathbf{x}\|_1$ — promotes piecewise smooth images
- **Maximum Entropy (MEM):** $\mathcal{R}(\mathbf{x}) = \sum_i x_i \log(x_i / p_i)$ — promotes smooth, positive distributions
- **L1 Sparsity:** $\mathcal{R}(\mathbf{x}) = \|\mathbf{x}\|_1$ — promotes compact emission

### Key Challenges

| Challenge | Description |
|-----------|-------------|
| Sparse sampling | Only ~0.01% of the Fourier plane is measured |
| Noise | Thermal noise at ~20 GHz bandwidths |
| Positivity | Sky brightness must be non-negative |
| Scale ambiguity | Absolute flux calibration is uncertain |
| Dynamic range | Bright features may be $10^3 \times$ brighter than faint ones |

---

## Methods

### 1. Dirty Image (Baseline)

The matched-filter (back-projection) image:
$$\mathbf{x}_{\text{dirty}} = \mathbf{A}^H \mathbf{y}$$

Equivalent to the inverse Fourier transform of the zero-padded visibility data.
Severely blurred by the PSF (dirty beam) due to missing Fourier components.

### 2. CLEAN

The standard algorithm in radio astronomy (Högbom 1974). Assumes the sky
consists of point sources and iteratively deconvolves the PSF:

1. Find the peak in the residual (dirty) image
2. Subtract a fraction (`gain`) of the peak convolved with the PSF
3. Record the component; repeat until convergence
4. Convolve clean components with an idealized Gaussian beam; add residual

Variants: Cotton–Schwab CLEAN (major/minor cycles), multiscale CLEAN.

### 3. RML with Total Variation (RML-TV)

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \geq 0}\; \frac{1}{2\sigma^2}\|\mathbf{A}\mathbf{x}-\mathbf{y}\|^2 + \lambda \|\nabla \mathbf{x}\|_1$$

Solved with L-BFGS-B. Produces sharper edges than MEM, good for ring structures.

### 4. RML with Maximum Entropy (RML-MEM)

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \geq 0}\; \frac{1}{2\sigma^2}\|\mathbf{A}\mathbf{x}-\mathbf{y}\|^2 - \lambda\, H(\mathbf{x})$$

where $H(\mathbf{x}) = -\sum_i x_i \log(x_i/p_i)$ is the entropy relative to
prior $p_i$. Used by the EHT team in their 2019 imaging scripts.

### 5. Deep Learning (Planned)

Score-based diffusion priors, unrolled networks (DEQ-based), and PnP methods.

---

## Data

### Synthetic Data (Default)

Run `generate_data.py` to create:
- A synthetic M87\*-like ring image (N×N pixels, adjustable SNR)
- Simulated EHT uv-coverage (9 stations, ~6h observation, 230 GHz)
- Clean and noisy complex visibilities (`.npy` format)

```bash
python generate_data.py
```

### Real EHT Data (Optional)

The EHT released the 2017 M87\* calibrated visibility data publicly:
- [EHT Data Portal](https://eventhorizontelescope.org/for-astronomers/data)
- Requires the `eht-imaging` package to load UVFITS files

---

## Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| NRMSE | $\|\hat{x} - x\|_2 / \|x\|_2$ | Lower is better |
| NCC | $\hat{x}^T x / (\|\hat{x}\| \|x\|)$ | Higher is better; max 1 |
| Dynamic Range | $\max(x) / \text{rms(background)}$ | Higher means better sensitivity |

All metrics are computed after flux normalization (matching total flux to ground truth).

---

## Quick Start

```python
import sys
sys.path.insert(0, '.')
from src.forward_model import VLBIForwardModel
from src.solvers import CLEANReconstructor, RMLSolver, TVRegularizer
from src.visualization import plot_comparison, compute_metrics
import numpy as np

# Load synthetic data
data = np.load('data/dataset.npz')
image_gt = data['image']
uv_coords = data['uv_coords']
vis_noisy = data['vis_noisy']
noise_std = float(data['noise_std'])
pixel_size_rad = float(data['pixel_size_rad'])

# Build forward model
model = VLBIForwardModel(uv_coords, image_gt.shape[0], pixel_size_rad)

# Reconstruct
clean = CLEANReconstructor(gain=0.1, n_iter=500)
x_clean = clean.reconstruct(model, vis_noisy, noise_std)

rml_tv = RMLSolver(regularizers=[(1e4, TVRegularizer())])
x_tv = rml_tv.reconstruct(model, vis_noisy, noise_std)

# Compare
plot_comparison({'CLEAN': x_clean, 'RML-TV': x_tv}, ground_truth=image_gt)
```

See the full tutorial in [`notebooks/eht_black_hole.ipynb`](notebooks/eht_black_hole.ipynb).

---

## References

- Högbom, J.A. (1974). *Aperture Synthesis with a Non-Regular Distribution of
  Interferometer Baselines*. A&AS, 15, 417.
- Thompson, A.R., Moran, J.M., & Swenson, G.W. (2017). *Interferometry and
  Synthesis in Radio Astronomy* (3rd ed.). Springer.
- EHT Collaboration et al. (2019). *First M87 Event Horizon Telescope Results.
  Paper I–VI*. ApJL, 875.
- Chael, A.A. et al. (2018). *Interferometric Imaging Directly with Closure
  Phases and Closure Amplitudes*. ApJ, 857, 23.
  ([eht-imaging](https://github.com/achael/eht-imaging))
- EHT Collaboration et al. (2022). *First Sgr A* Event Horizon Telescope Results.
  Paper I–VI*. ApJL, 930.
