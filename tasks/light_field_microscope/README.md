# Light Field Microscope — Artifact-free 3D Deconvolution

> Reconstruct a 3D fluorescence volume from a single 2D light field image using a wave-optics forward model and a depth-adaptive anti-aliasing deconvolution scheme.

**Domain:** Biology / Microscopy
**Modality:** Light Field Microscopy (LFM)
**Difficulty:** Hard

---

## Background

A **light field microscope (LFM)** inserts a micro-lens array (MLA) into the back focal plane of a conventional wide-field fluorescence microscope. Each micro-lens forms a small sub-image on the sensor that encodes both spatial and angular information about the fluorescent sample. This allows capturing a full 3D volume in a single camera exposure — enabling high-speed volumetric imaging of live specimens (zebrafish larvae, cardiomyocyte organoids, etc.).

### The Aliasing Problem

The LFM sampling pattern is **depth-dependent**: objects at the native object plane (NOP, Δz = 0) are sampled at the coarsest rate, causing severe grid-pattern aliasing artifacts in 3D reconstructions using standard deconvolution methods. The recoverable lateral resolution is non-uniform across depth, and reconstructions near the zero plane are dominated by artifacts rather than true signal.

This task implements and demonstrates the **aliasing-aware EMS (Estimate-Maximize-Smooth)** algorithm from Stefanoiu et al. (2019), which derives depth-dependent anti-aliasing filters and applies them as a smoothing step within Richardson-Lucy deconvolution, achieving artifact-free 3D reconstructions.

---

## Physical Model

### Light Field Forward Model

A fluorescent source point **o**(o_x, o_y, o_z) produces a diffraction pattern at the MLA via the **Debye integral** for circular lens apertures:

$$U_{mla-}(\mathbf{0}, x_{mla}, y_{mla}) = \frac{deM}{o_z^2 \lambda^2} \exp\!\left(-\frac{iu}{4\sin^2(\alpha/2)}\right) \int_0^\alpha P(\theta)\, \exp\!\left(\frac{iu\sin^2(\theta/2)}{2\sin^2(\alpha/2)}\right) J_0\!\left(\frac{\sin\theta}{\sin\alpha} v\right) \sin\theta\, d\theta$$

where:
- $u = \frac{8\pi}{\lambda} \Delta z_{FOP} \sin^2(\alpha/2)$ is the axial optical coordinate
- $v = \frac{2\pi}{\lambda} \sqrt{(x_{mla}-o_x)^2+(y_{mla}-o_y)^2}\sin\alpha$ is the radial optical coordinate
- $\alpha = \arcsin(NA/n)$ is the objective half-angle
- $deM = dof_{mla}/d_{il}^{mla}$ is the demagnification factor
- $J_0$ is the zeroth-order Bessel function

The MLA applies a tiled quadratic phase transmittance:

$$T(x_{mla}, y_{mla}) = \text{rep}_{p_{ml}}\!\left(P(x_l, y_l)\, e^{-ik(x_l^2+y_l^2)/(2f_{ml})}\right)$$

The field propagates from MLA to sensor via the **Rayleigh-Sommerfeld transfer function**:

$$U_{sens} = \mathcal{F}^{-1}\!\left\{\mathcal{F}\{U_{mla+}\} \cdot H_{rs}(f_x, f_y)\right\}, \quad H_{rs} = e^{ikd_{mla}^{sens}\sqrt{1-\lambda^2(f_x^2+f_y^2)}}$$

### Discretized Imaging Model

The light field measurement vector **m** (sensor pixels) arises from the 3D volume **v** (voxels) via a Poisson process:

$$\mathbf{m} \sim \text{Poisson}(A\mathbf{v}), \qquad a_{ji} = |U_{sens}(\mathbf{o}(i), \mathbf{x}_s(j))|^2$$

where $A = (a_{ji})$ is the **light field forward operator** (LFPSF matrix), $a_{ji}$ being the detection probability that a photon emitted at voxel $i$ is counted at sensor pixel $j$.

### Depth-Adaptive Anti-Aliasing Filter

The LFM has a depth-dependent anti-aliasing requirement. The filter radius at sensor resolution is:

$$w_{sens_z} = \min\!\left(|\gamma_z p_{ml} - b_z|,\; r_{ml}\right)$$

where $\gamma_z = d_{mla}^{sens} r_{tl} / (d_{tl}^{mla} B_z)$ is the depth-dependent scaling, $B_z$ is the micro-lens blur radius, and $b_z = r_{ml}|1/z''' - 1/d_{mla}^{sens}|$. Backprojecting to object space at super-resolution factor $s$:

$$w_{obj_z} = w_{sens_z} \cdot s / p_{ml}$$

---

## Data Description

**Synthetic data** is generated from the parameters in `data/meta_data`:

| File | Description |
|------|-------------|
| `data/meta_data` | JSON: microscope & MLA parameters, reconstruction settings |
| `data/raw_data.npz` | Keys: `lf_image` (2D sensor array), `ground_truth` (3D bead volume) |

### Microscope Configuration

| Parameter | Value |
|-----------|-------|
| Objective | 20× / 0.5 NA |
| Tube lens focal length | 165 mm |
| MLA pitch | 150 μm |
| MLA focal length | 3 mm |
| Pixel pitch (sensor) | 6.5 μm |
| Emission wavelength | 525 nm |
| Plenoptic type | 1 (original LFM) |

### Synthetic Scene

- **7 × 7 lenslet grid** (105 × 105 sensor pixels at 15 px/lenslet)
- **5 fluorescent beads** (radius 2 μm) placed at random lateral positions across 5 depth planes (Δz ∈ {−300, −150, 0, 150, 300} μm)
- Poisson noise with scale factor 1000

---

## Method Hints

### Step 1: Build Geometry
Use `preprocessing.py` to parse `meta_data`, build the `Camera` parameter dictionary, and call `compute_geometry()` in synthetic mode (no white calibration image needed).

### Step 2: Compute PSF Operators
Compute the forward operator $H$ and backward operator $H^T$ using wave-optics:
1. Compute the Debye integral PSF at each depth (exploiting conjugate symmetry for ±Δz)
2. Apply MLA quadratic phase transmittance
3. Propagate to sensor via Rayleigh-Sommerfeld (angular spectrum method)
4. Store each per-voxel sensor pattern as a sparse matrix

Due to the quarter-symmetry of regular-grid LFMs, only 1/4 of the patterns need to be computed explicitly.

**Note:** PSF computation is the bottleneck (~10–60 min depending on hardware). Precomputed operators are provided in `evaluation/reference_outputs/operators_H.pkl`.

### Step 3: EMS Deconvolution
Run the Estimate-Maximize-Smooth algorithm for `n_iter` iterations (paper Eq. 27):

$$\mathbf{v}^{q+1} = h_{f_w,z} * \left[\frac{\mathbf{v}^q}{A^T\mathbf{1}} \cdot A^T\!\left(\frac{\mathbf{m}}{A\mathbf{v}^q}\right)\right]$$

where $h_{f_w,z}$ is a Lanczos-4 windowed sinc filter with depth-dependent width $w_{obj_z}$.

Compare with `filter_flag=False` (standard RL) to observe the aliasing artifacts near Δz = 0 that the EMS smoothing step eliminates.

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `depthRange` | [−300, 300] μm | Axial reconstruction range |
| `depthStep` | 150 μm | Depth plane spacing |
| `newSpacingPx` | 15 px | Pixels per lenslet (downsampled) |
| `superResFactor` | 5 | Voxels per lenslet |
| `niter` | 8 | Deconvolution iterations |
| `lanczosWindowSize` | 4 | Lanczos window order |

---

## References

- A. Stefanoiu, J. Page, P. Symvoulidis, G. G. Westmeyer, T. Lasser, **"Artifact-free deconvolution in light field microscopy"**, *Optics Express* 27(22):31644–31666, 2019. DOI: [10.1364/OE.27.031644](https://doi.org/10.1364/OE.27.031644)
- M. Broxton, L. Grosenick, S. Yang, N. Cohen, A. Andalman, K. Deisseroth, M. Levoy, **"Wave optics theory and 3-D deconvolution for the light field microscope"**, *Optics Express* 21(21):25418, 2013.
- pyolaf Python implementation: [github.com/lambdaloop/pyolaf](https://github.com/lambdaloop/pyolaf)
