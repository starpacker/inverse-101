# EHT Black Hole Dynamic Imaging

## Problem

Reconstruct a time-varying video of a black hole (e.g., SgrA*) from sparse
interferometric measurements obtained by the Event Horizon Telescope (EHT).
Each time frame has a different set of (u,v) baseline measurements due to
Earth rotation, and individual frames are severely under-determined.

This task implements the **StarWarps** algorithm from Bouman et al. (2017),
which uses a Gaussian Markov model with forward-backward message passing
and EM optimization to jointly reconstruct all frames while enforcing
temporal coherence.

## Physics

### Interferometric Imaging

The EHT measures complex visibilities — samples of the Fourier transform of
the sky brightness distribution at spatial frequencies determined by baseline
geometry:

    V(u,v) = ∫∫ I(l,m) e^{-2πi(ul + vm)} dl dm

For N×N images, this is discretized as a DFT:

    y_t = A_t x_t + n_t

where A_t is the per-frame measurement matrix (M_t × N² complex), x_t is the
vectorized image, and n_t is thermal noise with per-baseline variance from
station SEFDs.

### Gaussian Markov Model (StarWarps)

The video is modeled as a first-order Gaussian Markov process:

    x_t = A(θ) x_{t-1} + w_t,    w_t ~ N(0, Q)

where A(θ) is a warp matrix parameterized by affine motion θ, and Q controls
the allowed intensity variation between frames.

### EM Algorithm

1. **E-step**: Forward-backward message passing computes posterior statistics
2. **M-step**: L-BFGS-B optimizes warp parameters θ

## Data

Synthetic rotating crescent (SgrA*-like) observed by EHT 2017 array:
- **Source**: SgrA* (RA=17h45m40s, Dec=-29°00'28")
- **Image**: 30×30 pixels, 3.4 μas/pixel
- **Frames**: 12 frames over 6 hours
- **Rotation**: 90° bright-spot rotation
- **Noise**: Per-baseline SEFD-based thermal noise at 230 GHz

### Data Files

- `data/raw_data.npz`: Per-frame visibilities, UV coordinates, noise
- `data/meta_data`: JSON imaging parameters

## Method Hints

1. Build DFT matrices per frame from UV coordinates
2. Construct Gaussian prior with power-law Fourier covariance
3. Run static per-frame baseline (Gaussian MAP)
4. Run StarWarps with visibility-only data term
5. Compare NRMSE/NCC metrics

## References

- Bouman et al. 2017, arXiv:1711.01357 ("StarWarps")
- Event Horizon Telescope Collaboration, Paper IV (2019), ApJL 875, L4
- Chael et al. 2018, ApJ 857:23 (closure-only imaging)
