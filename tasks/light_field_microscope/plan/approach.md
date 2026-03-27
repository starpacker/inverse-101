# Approach: Artifact-free 3D Deconvolution of Light Field Microscopy Data

## Problem Statement

Recover a 3D fluorescence volume **v** (shape: texH × texW × nDepths) from a single 2D light
field sensor image **m** (shape: imgH × imgW), given the Poisson imaging model:

    m ~ Poisson(A v)

where A is the light field forward operator encoding the wave-optics point spread function
(LFPSF) of the specific microscope configuration.

The key challenge: LFM sampling is depth-dependent. Near the native object plane (Δz ≈ 0),
the MLA samples the light field at the coarsest rate, causing severe grid-pattern aliasing
artifacts in standard Richardson-Lucy deconvolution. The EMS algorithm of Stefanoiu et al.
(2019) addresses this by deriving and applying depth-dependent anti-aliasing filters.

---

## Mathematical Formulation

### Forward Operator (A)

Each entry a_{ji} = |U_sens(o(i), x_s(j))|² is the probability that a photon from voxel i
reaches sensor pixel j. U_sens is computed via three steps:

1. **Debye integral** (Eq. 9 in paper): wavefront at MLA plane from source at depth Δz_FOP:
   - Numerical integration via scipy.integrate.quad_vec
   - Exploits circular symmetry: only one quarter of the PSF is computed, rest by rotation

2. **MLA transmittance** (Eq. 12-14): multiply wavefront by tiled quadratic phase mask
   - t(x_l, y_l) = P(x_l, y_l) · exp(-ik(x²+y²)/(2·f_ml))
   - P = square aperture (uLensMask=1 for this configuration)

3. **Rayleigh-Sommerfeld propagation** (Eq. 15-16): angular spectrum method
   - H_rs = exp(ik·d_mla^sens · sqrt(1 - λ²(f_x²+f_y²)))
   - Applied via FFT convolution

### Anti-Aliasing Filter Width (Eq. 5-7)

For each depth Δz, the depth-dependent filter radius at sensor resolution:
   w_sens_z = min(|γ_z · p_ml - b_z|, r_ml)

where γ_z is the depth-dependent scaling factor and b_z is the micro-lens blur radius.
Backprojected to object space (voxels):
   w_obj_z = w_sens_z · s / p_ml   (s = superResFactor)

### EMS Algorithm (Eq. 27)

Standard Richardson-Lucy + per-depth Lanczos-4 smoothing step:

   v^{q+1} = h_{fw,z} * [ v^q / (A^T 1) · A^T(m / A·v^q) ]

With filter_flag=False (standard RL): no smoothing → artifacts near zero plane.
With filter_flag=True (EMS): depth-adaptive smoothing → artifact-free reconstruction.

---

## Solution Steps

1. **Parse metadata** → build Camera parameter dict with all derived optical quantities
   (fobj, k, objRad, uRad, offsetFobj, etc.)

2. **Compute geometry** in synthetic mode (no calibration image):
   - Build ideal lenslet grid from Camera specs
   - Compute Resolution dict (Nnum, TexNnum, sensor/voxel sizes, depth array, masks)
   - Compute LensletCenters in px / metric / vox coordinates

3. **Compute PSF operators** (one-time expensive computation, ~10-60 min):
   - For each depth: compute Debye PSF at native image plane
   - For each (aa, bb, depth) texture coordinate: shift PSF, apply MLA, propagate to sensor
   - Exploit quarter-symmetry for regular grids
   - Store H as object array of csr_matrix; normalize and store Ht similarly

4. **Generate synthetic data**:
   - Place n_beads Gaussian spheres at random lateral positions and the 5 depth planes
   - Forward project via H to get clean LF image
   - Add Poisson noise → raw lenslet image saved to data/raw_data.npz

5. **Run deconvolution**:
   - Precompute A^T·1 (normalization denominator)
   - Standard RL (8 iter, filter_flag=False) → reconstruction_rl
   - EMS (8 iter, filter_flag=True, Lanczos-4) → reconstruction_ems

6. **Evaluate**:
   - NRMSE and PSNR of each reconstruction vs ground truth volume
   - EMS should show lower NRMSE (fewer artifacts) than RL

---

## Expected Results

| Method | NRMSE | PSNR (dB) | Artifacts |
|--------|-------|-----------|-----------|
| Standard RL | ~0.5–0.7 | ~15–20 | Strong grid artifacts at Δz≈0 |
| EMS (this work) | ~0.3–0.5 | ~20–28 | Artifact-free across all depths |

The visual improvement is the primary result: EMS reconstruction shows clean bead images
at all depth planes, while RL shows characteristic MLA-pitch-spaced grid patterns near Δz=0.

---

## Algorithm Notes

- **Conjugate symmetry**: PSF at depth -Δz is the complex conjugate of PSF at +Δz.
  This halves the number of Debye integrations needed.
- **Quarter-symmetry**: For regular grids, the pattern H[aa,bb,c] at position (aa,bb)
  in the upper-left quadrant determines all four quadrants by reflection.
  This reduces H computation to 1/4 of the texture coordinates.
- **Sparse storage**: Each LFPSF pattern has ~5% nonzero entries; csr_matrix gives
  ~20× memory savings.
- **FFT convolution batching**: forward/backward projection batches convolutions for
  efficiency (step parameter controls batch size).
- **No GPU required**: All operations use numpy/scipy; CuPy acceleration is optional.
