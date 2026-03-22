# EHT Black Hole Probabilistic Imaging (DPI)

> Learn the posterior distribution over radio images of a supermassive black
> hole from sparse interferometric measurements, enabling principled
> uncertainty quantification via a normalizing flow generative model.

**Domain:** Astronomy
**Modality:** Radio interferometry (VLBI)
**Difficulty:** Hard

---

## Background

Standard image reconstruction methods (CLEAN, RML) produce a single point
estimate. But the EHT inverse problem is severely ill-posed: many images are
consistent with the sparse measurements. **Which image features are reliable,
and which are artifacts of the reconstruction?**

Deep Probabilistic Imaging (DPI) addresses this by training a Real-NVP
normalizing flow to approximate the posterior distribution p(image | data).
After training, posterior samples are obtained instantly by pushing random
Gaussian vectors through the learned flow, enabling:

- **Uncertainty maps** (pixel-wise standard deviation)
- **Multi-modal exploration** (distinct plausible images)
- **Feature credibility** (is the asymmetry real?)

---

## Physical Model

The forward model maps images to interferometric observables:

$$V_{ij} = \sum_p I_p \, e^{-2\pi i (u_{ij} l_p + v_{ij} m_p)}$$

Because station-based gain errors corrupt raw visibilities, DPI uses
**closure quantities** that are immune to gain calibration:

- **Closure phase** (triangle of 3 stations):
  $$\phi_{ijk} = \arg(V_{ij}) + \arg(V_{jk}) + \arg(V_{ki})$$

- **Log closure amplitude** (quadrangle of 4 stations):
  $$\ln A_{ijkl} = \ln|V_{ij}| + \ln|V_{kl}| - \ln|V_{ik}| - \ln|V_{jl}|$$

The DPI training loss minimizes the KL divergence from the posterior:

$$\theta^* = \arg\min_\theta \, \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ \mathcal{L}_\text{data}(y, f(G_\theta(z))) + \lambda \mathcal{R}(G_\theta(z)) - \beta \log|\det \frac{\partial G_\theta}{\partial z}| \right]$$

| Symbol | Description |
|--------|-------------|
| G_θ    | Real-NVP normalizing flow (latent z → image x) |
| L_data | Chi-squared on closure phases + log closure amplitudes |
| R(x)   | Image priors (MEM, TSV, L1, flux, centering) |
| β log det | Entropy term — prevents posterior collapse |

---

## Data Description

### `data/obs.uvfits`

SgrA* 2015 ALMA VLBI observation in UVFITS format, containing 938
visibilities at 230 GHz. Loaded via ehtim.

### `data/gt.fits`

128×128 reference ground-truth image in FITS format. Regridded to 32×32
at the task field of view (160 μas).

### `data/meta_data`

JSON file with imaging and training parameters:
- `npix` = 32: image size (32×32 pixels)
- `fov_uas` = 160.0: field of view in microarcseconds
- `n_flow` = 16: number of Real-NVP flow blocks
- `n_epoch` = 30000: training epochs
- `batch_size` = 32: mini-batch size
- `lr` = 1e-4: Adam learning rate
- `logdet_weight` = 1.0: entropy term weight
- `l1_weight`, `tsv_weight`, `flux_weight`, `center_weight`, `mem_weight`: prior weights
- `grad_clip` = 0.1: gradient clipping threshold

---

## Method Hint

**Deep Probabilistic Imaging** uses a Real-NVP normalizing flow:

1. **Architecture**: Stack of flow blocks, each containing ActNorm +
   AffineCoupling (with 2-layer networks: LeakyReLU, BatchNorm1d, ZeroFC).
   Random permutations between blocks.

2. **Training**: Sample z ~ N(0,I), push through flow to get images.
   Minimize data fidelity (closure chi-squared) + priors - entropy (log-det).
   Adam optimizer with gradient clipping.

3. **Positivity**: Softplus activation + learnable scale factor ensures
   non-negative images. Log-determinant correction accounts for the transform.

4. **Posterior sampling**: After training, sample z's and push through
   the flow to get posterior image samples. Compute mean, std, etc.

Key hyperparameters:
- 16 flow blocks, seqfrac=4 (hidden dimension = npix²/8)
- Closure-only data fidelity (no raw visibility loss)
- MEM + TSV + L1 + flux + centering priors

---

## References

- Sun, H. & Bouman, K.L. (2020). *Deep Probabilistic Imaging: Uncertainty Quantification and Multi-modal Solution Characterization for Computational Imaging*. arXiv:2010.14462.
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). *Density estimation using Real-NVP*. ICLR 2017.
- EHT Collaboration et al. (2019). *First M87 Event Horizon Telescope Results*. ApJL, 875.
- Chael, A.A. et al. (2018). *Interferometric Imaging Directly with Closure Phases and Closure Amplitudes*. ApJ, 857, 23.
