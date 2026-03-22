# EHT Black Hole Imaging with Closure Quantities

> Recover the radio image of a supermassive black hole from gain-corrupted
> interferometric data using closure phases and log closure amplitudes --
> gain-invariant observables that bypass the need for antenna-based calibration.

**Domain:** Astronomy
**Modality:** Radio interferometry (VLBI)
**Difficulty:** Hard

---

## Background

The Event Horizon Telescope (EHT) is a Very Long Baseline Interferometry (VLBI) array
that links radio dishes across the globe, forming a virtual Earth-sized telescope
operating at 230 GHz (1.3 mm wavelength) with ~20 microarcsecond angular resolution.

Each pair of telescopes measures one **complex visibility** -- a Fourier coefficient of
the sky brightness distribution. In practice, the measured visibilities are corrupted by
**station-based complex gains** arising from atmospheric turbulence, instrumental
instabilities, and calibration errors. A station gain g_i multiplies every visibility
involving station i:

$$V_{ij}^{\text{corr}} = g_i \, g_j^* \, V_{ij}^{\text{true}}$$

Direct fitting of corrupted visibilities produces nonsensical images unless the gains are
first calibrated -- a non-trivial problem when there are no bright calibrator sources
in the field (as with EHT observations of M87*).

**Closure quantities** are algebraic combinations of visibilities that cancel the
station-based gains:

- **Closure phase** (triangle of stations i-j-k):
  $$\phi_C = \arg(V_{ij} \cdot V_{jk} \cdot V_{ki})$$
  The station gains cancel: $\phi_C = \phi_{ij} + \phi_{jk} + \phi_{ki}$.

- **Log closure amplitude** (quadrangle of stations i-j-k-l):
  $$\log CA = \log|V_{ij}| + \log|V_{kl}| - \log|V_{ik}| - \log|V_{jl}|$$
  Again gain-invariant since each station appears equally in numerator and denominator.

Imaging directly from closure quantities avoids calibration entirely, at the cost of
reduced information (fewer independent constraints than raw visibilities).

---

## Physical Model

The van Cittert-Zernike theorem relates the measured visibility to the sky brightness:

$$V(u, v) = \iint I(l, m)\, e^{-2\pi i (ul + vm)}\, dl\, dm$$

Discretized on an N x N pixel grid with the DFT matrix A:

$$\mathbf{y} = \mathbf{A}\,\mathbf{x}$$

The DFT matrix follows the ehtim sign convention (+2*pi*i) and uses a triangle pulse
pixel response function:

$$A_{m,n} = P(u_m, v_m) \, \exp\!\bigl[+2\pi i\,(u_m l_n + v_m m_n)\bigr]$$

where P is the Fourier-domain triangle pulse.

| Symbol | Description | Size |
|--------|-------------|------|
| **x** | Sky brightness image (vectorized, non-negative) | N^2 = 4096 |
| **A** | Measurement matrix (DFT with triangle pulse) | M x N^2 |
| **y_corr** | Gain-corrupted complex visibilities | M = 421 |
| **y_cal** | Calibrated complex visibilities (ground truth) | M = 421 |
| **g** | Station-based complex gains | N_s = 7 |

The gain corruption model is:

$$y_{ij}^{\text{corr}} = g_i \, g_j^* \, y_{ij}^{\text{cal}}$$

with gain amplitudes |g_i| ~ 1 +/- 20% and gain phases arg(g_i) ~ U(-30, +30) degrees.

---

## Data Description

### `data/raw_data.npz`

NumPy NPZ archive containing:
- `vis_cal` -- complex128 array, shape (421,): calibrated visibilities (with thermal noise)
- `vis_corrupt` -- complex128 array, shape (421,): gain-corrupted visibilities
- `uv_coords` -- float64 array, shape (421, 2): baseline coordinates in wavelengths
- `sigma_vis` -- float64 array, shape (421,): per-baseline thermal noise sigma (Jy)
- `station_ids` -- int64 array, shape (421, 2): station pair indices for each baseline

### `data/meta_data`

JSON file with imaging parameters:
- `N` = 64: image size (64 x 64 pixels)
- `pixel_size_uas` = 2.0: pixel size in microarcseconds
- `pixel_size_rad`: pixel size in radians (~9.7e-12)
- `total_flux` = 0.6: total source flux in Jy
- `noise_std`: median per-baseline noise sigma (Jy)
- `freq_ghz` = 230.0: observing frequency
- `n_baselines` = 421: number of measured baselines
- `n_stations` = 7: number of EHT stations
- `station_names`: list of station names (ALMA, APEX, JCMT, LMT, PV, SMA, SMT)
- `gain_amp_error` = 0.2: fractional amplitude gain error (20%)
- `gain_phase_error_deg` = 30.0: max phase gain error in degrees

### Data Generation

The observation is synthesized from a ground-truth M87*-like ring image (Gaussian ring
with Doppler-boosted asymmetry). The array consists of 7 EHT stations from the 2017
campaign. UV coverage is computed using astropy for proper Earth-rotation synthesis over
a 6-hour track. Per-baseline thermal noise is derived from station SEFDs. Station-based
complex gains are applied to create the corrupted visibilities.

---

## Method Hint

The standard approach is **Regularized Maximum Likelihood (RML)** imaging directly from
closure quantities, following Chael et al. (2018):

$$\mathbf{x}^* = \arg\min_{\mathbf{x} \geq 0}\; \sum_d \alpha_d \bigl(\chi^2_d(\mathbf{x}) - 1\bigr) + \sum_r \alpha_r \, \mathcal{R}_r(\mathbf{x})$$

### Data Terms (Chael 2018 Eqs. 11-12)

1. **Closure phase chi-squared** (Eq. 11):
   $$\chi^2_{CP} = \frac{2}{N_{CP}} \sum_{t} \frac{1 - \cos(\phi_t^{\text{obs}} - \phi_t^{\text{model}})}{\sigma_t^2}$$

2. **Log closure amplitude chi-squared** (Eq. 12):
   $$\chi^2_{CA} = \frac{1}{N_{CA}} \sum_{q} \left(\frac{\log CA_q^{\text{obs}} - \log CA_q^{\text{model}}}{\sigma_q}\right)^2$$

### Regularizers

- **Gull-Skilling entropy:** $S(I) = \sum(I - P - I \log(I/P))$
- **Simple entropy:** $S(I) = -\sum I \log(I/P)$
- **Total Variation:** $\text{TV}(I) = \sum \sqrt{(\partial_x I)^2 + (\partial_y I)^2 + \epsilon^2}$

### Solver

L-BFGS-B with positivity constraints (x >= 0). Multi-round optimization (3 rounds of
300 iterations each) matching ehtim's imaging workflow.

### Comparison Methods

For demonstrating the importance of closure quantities under gain corruption, implement
three solvers:

1. **Visibility RML** (baseline): Fit corrupted visibilities directly -- breaks under
   gain corruption.
2. **Amplitude + Closure Phase RML**: Gain-invariant closure phases but gain-sensitive
   visibility amplitudes -- partially robust.
3. **Closure-only RML**: Both closure phases and log closure amplitudes -- fully
   gain-invariant, the main result.

---

## Expected Results

| Method | On calibrated data | On corrupted data |
|--------|-------------------|-------------------|
| | NRMSE / NCC | NRMSE / NCC |
| Vis RML | ~0.26 / ~0.97 | ~5.0 / ~0.01 |
| Amp+CP | ~0.70 / ~0.76 | ~1.7 / ~0.41 |
| Closure-only | ~0.82 / ~0.75 | ~0.85 / ~0.72 |

Key insight: Visibility RML performs best on calibrated data but catastrophically fails
on gain-corrupted data. Closure-only imaging is robust to gain corruption, maintaining
similar performance regardless of calibration state.

---

## References

- Chael, A.A., Johnson, M.D., Bouman, K.L., et al. (2018). *Interferometric Imaging Directly with Closure Phases and Closure Amplitudes*. ApJ, 857, 23.
- EHT Collaboration et al. (2019). *First M87 Event Horizon Telescope Results. Paper I-VI*. ApJL, 875.
- Thompson, A.R., Moran, J.M., & Swenson, G.W. (2017). *Interferometry and Synthesis in Radio Astronomy* (3rd ed.). Springer.
- ehtim: https://github.com/achael/eht-imaging
