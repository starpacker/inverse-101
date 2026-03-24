# EHT Black Hole Feature Extraction (α-DPI)

## Background

This task implements **α-deep Probabilistic Inference (α-DPI)** for extracting geometric features from Event Horizon Telescope (EHT) observations of black holes. Unlike pixel-space imaging, α-DPI fits a parametric geometric model (crescent + elliptical Gaussians) to closure quantities, inferring posterior distributions over physical parameters such as ring diameter, width, brightness asymmetry, and position angle.

### Physical Model

The black hole image is modeled as:
- **Crescent**: An asymmetric Gaussian ring parameterized by (diameter, width, asymmetry, position angle)
- **Nuisance Gaussians**: N elliptical Gaussian components for extended emission, each with (x, y, scale, sigma_x, sigma_y, rho)
- Total parameters: 4 + 6N (e.g., 16 for N=2)

### Data: Closure Quantities

The forward model maps geometric parameters → image → visibilities → closure quantities:
- **Closure phases** (ψ): Phase sums around triangles of baselines, robust to station-based phase errors
- **Log closure amplitudes** (log C): Log amplitude ratios around quadrangles, robust to station-based gain errors

### α-DPI Algorithm

α-DPI is a two-step algorithm:
1. **Variational inference**: Train a normalizing flow q_θ(x) to approximate the posterior using α-divergence:
   - θ* = argmin (1/N) Σ [w_n * loss_n], where w_n ∝ softmax(-(1-α)*loss_n)
   - When α=1, reduces to standard KL divergence (mean loss)
   - α < 1 provides more exploration and avoids mode collapse
2. **Importance sampling**: Reweight flow samples by p(y|x)p(x)/q_θ(x) for accurate posteriors

Model selection uses the Evidence Lower Bound (ELBO) to compare models with 0, 1, 2, 3 Gaussian components.

## Data Description

| File | Format | Description |
|------|--------|-------------|
| `data/obs.uvfits` | UVFITS | Synthetic EHT observation (crescent + 2 Gaussians) |
| `data/gt.fits` | FITS | Ground truth image |
| `data/model_params.npy` | NumPy | Ground truth geometric parameters |
| `data/meta_data` | JSON | Imaging and training hyperparameters |

### Key Parameters (from meta_data)
- Image: 64×64 pixels, FOV = 120 μas
- Flow: 16 blocks, seqfrac = 1/16
- Training: 10000 epochs, batch = 2048, lr = 1e-4
- α-DPI: β = 1.0 (α = 1 - β/N_data), warmup start_order = 4, decay_rate = 2000
- Geometric model: simple_crescent_nuisance with n_gaussian = 2

## Method Hints

- **Real-NVP flow**: Maps Gaussian latent → unconstrained params, sigmoid → [0,1] unit params
- **Geometric model**: Unit params → physical params → 2D image (differentiable via PyTorch)
- **Forward model**: Image → NUFFT → visibilities → closure phases + log closure amplitudes
- **α-divergence**: Use importance-weighted softmax for non-KL objectives
- **Data warmup**: Gradually increase data weight: `min(10^(-4+k/2000), 1.0)`

## References

- Sun et al. (2022), "α-deep Probabilistic Inference (α-DPI): Efficient Uncertainty Quantification from Exoplanet Astrometry to Black Hole Feature Extraction", ApJ 932:99
- Sun & Bouman (2020), arXiv:2010.14462 — Deep Probabilistic Imaging
- GitHub: https://github.com/HeSunPU/DPI
