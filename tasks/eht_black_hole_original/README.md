# EHT Closure-Only Black Hole Imaging

> Recover the radio image of a supermassive black hole using only closure
> quantities (closure phases and closure amplitudes), which are robust to
> station-based calibration errors that corrupt traditional visibility imaging.

**Domain:** Astronomy
**Modality:** Radio interferometry (VLBI)
**Difficulty:** Hard

---

## Background

The Event Horizon Telescope (EHT) measures complex visibilities — Fourier coefficients of the sky brightness. However, each telescope introduces unknown complex gain errors that corrupt these measurements. Traditional imaging using calibrated visibilities requires accurate gain calibration, which is often unreliable.

**Closure quantities** are combinations of visibilities on which station-based gains cancel exactly:
- **Closure phases** (on triangles of 3 stations): robust to both amplitude and phase gains
- **Closure amplitudes** (on quadrangles of 4 stations): robust to amplitude gains

Chael et al. (2018) showed that imaging *directly* with closure quantities produces results comparable to or better than imaging with calibrated visibilities, without requiring any gain calibration.

---

## Physical Model

Standard VLBI forward model:

$$V_{ij} = \sum_p I_p \, e^{-2\pi i (u_{ij} l_p + v_{ij} m_p)}$$

With gain corruption: $V_{ij}^{\text{obs}} = g_i \, g_j^* \, V_{ij}^{\text{true}} + n_{ij}$

Closure phase (triangle i,j,k): $\phi_{ijk} = \arg(V_{ij} \cdot V_{jk} \cdot V_{ki})$

Closure amplitude (quadrangle i,j,k,l): $\text{CA} = |V_{ij} \cdot V_{kl}| / |V_{ik} \cdot V_{jl}|$

Closure-only chi-squared (Chael 2018):
- **Closure phase** (Eq. 11): $\chi^2_{\text{CP}} = \frac{2}{N_{\text{CP}}} \sum_k \frac{1 - \cos(\phi_k^{\text{obs}} - \phi_k^{\text{model}})}{\sigma_k^2}$
- **Log closure amplitude** (Eq. 12): $\chi^2_{\log\text{CA}} = \frac{1}{N_{\text{CA}}} \sum_k \frac{(\log\text{CA}_k^{\text{obs}} - \log\text{CA}_k^{\text{model}})^2}{\sigma_k^2}$

| Symbol | Description | Size |
|--------|-------------|------|
| **x** | Sky brightness image (vectorized, non-negative) | N² = 4096 |
| **A** | DFT measurement matrix | M × N² |
| **y** | Complex visibilities (gain-corrupted) | M = 540 |
| **φ** | Closure phases | N_tri |
| **CA** | Closure amplitudes | N_quad |

---

## Data Description

### `data/raw_data.npz`

NumPy NPZ archive containing:
- `vis_corrupted` — complex128 array, shape (540,): gain-corrupted visibilities
- `vis_true` — complex128 array, shape (540,): true (noisy but uncorrupted) visibilities
- `uv_coords` — float64 array, shape (540, 2): baseline coordinates in wavelengths
- `station_ids` — int64 array, shape (540, 2): station pair indices per baseline
- `noise_std_per_vis` — float64 array, shape (540,): per-visibility noise standard deviation

### `data/meta_data`

JSON file with imaging parameters:
- `N` = 64: image size (64×64 pixels)
- `pixel_size_uas` = 2.0: pixel size in microarcseconds
- `pixel_size_rad`: pixel size in radians
- `noise_std`: thermal noise standard deviation
- `freq_ghz` = 230.0: observing frequency
- `n_baselines` = 540: number of measured baselines
- `n_stations` = 9: number of EHT stations
- `gain_amp_error` = 0.2: fractional amplitude gain error (20%)
- `gain_phase_error_deg` = 30.0: phase gain error in degrees

### Data Generation

Synthetic EHT observation: M87*-like ring model observed by 9 stations with
realistic Earth-rotation synthesis, thermal noise (SNR=20), and station-based
gain errors (20% amplitude, 30° phase). Closure quantities computed from the
corrupted visibilities are identical to those from the true visibilities,
demonstrating their gain invariance.

---

## Method Hint

The approach is **Regularized Maximum Likelihood (RML) with closure quantities**:

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \geq 0}\; \alpha_{\text{CP}} \chi^2_{\text{CP}}(\mathbf{x}) + \alpha_{\text{CA}} \chi^2_{\log\text{CA}}(\mathbf{x}) + \lambda\, \mathcal{R}(\mathbf{x})$$

Key difference from standard RML: the data fidelity terms use *closure* chi-squared (not visibility chi-squared), making the reconstruction independent of station gains.

Implement and compare:
1. **Closure phase only** + TV regularizer
2. **Closure phase + closure amplitude** + TV regularizer
3. **Closure phase + closure amplitude** + MEM regularizer
4. **Visibility RML** with corrupted visibilities (comparison baseline)

The closure-only methods should produce correct reconstructions while the visibility method fails due to uncorrected gain errors.

---

## References

- Chael, A.A. et al. (2018). *Interferometric Imaging Directly with Closure Phases and Closure Amplitudes*. ApJ, 857, 23.
- EHT Collaboration et al. (2019). *First M87 Event Horizon Telescope Results. Paper I–VI*. ApJL, 875.
- Thompson, A.R., Moran, J.M., & Swenson, G.W. (2017). *Interferometry and Synthesis in Radio Astronomy* (3rd ed.). Springer.
