# Problem Definition: Hessian-SIM Reconstruction

## 1. Physical Background

**Structured Illumination Microscopy (SIM)** overcomes the diffraction limit by illuminating the specimen with sinusoidal patterns at multiple angles and phases. The patterned illumination encodes high-frequency specimen information — normally beyond the optical transfer function (OTF) cutoff at $k_c = 2\text{NA}/\lambda$ — into the observable passband via the **Moiré effect**. By acquiring multiple images with shifted patterns and computationally unmixing the frequency bands, SIM achieves ~2× resolution improvement over conventional widefield microscopy.

However, the Wiener-filtered SIM reconstruction amplifies noise in the extended frequency region, producing artifacts that degrade image quality, especially at low signal-to-noise ratios. **Hessian-SIM** addresses this by applying a second-order regularization (Hessian norm) that penalizes curvature rather than gradients, preserving thin curvilinear biological structures (actin filaments, membrane tubules, microtubules) that first-order methods like Total Variation (TV) tend to over-smooth.

## 2. Forward Model

### SIM Illumination

For each illumination angle $\theta$ and phase shift $\phi_k$, the structured illumination pattern is:

$$I_k(\mathbf{r}) = I_0\left[1 + m \cos(\mathbf{k}_p \cdot \mathbf{r} + \phi_k)\right]$$

where $\mathbf{k}_p$ is the pattern wave vector and $m$ is the modulation depth.

### Image Formation

The observed image $D_k$ for phase $k$ is:

$$D_k = H \otimes (S \cdot I_k) + n$$

where $H$ is the point spread function (PSF), $S$ is the specimen fluorescence distribution, $\otimes$ denotes convolution, and $n$ is noise.

### Fourier Domain Decomposition

Expanding the cosine and taking the Fourier transform:

$$\tilde{D}_k(\mathbf{k}) = \tilde{H}(\mathbf{k}) \left[ \tilde{S}_0(\mathbf{k}) + \frac{m}{2}e^{j\phi_k} \tilde{S}_{+1}(\mathbf{k} - \mathbf{k}_p) + \frac{m}{2}e^{-j\phi_k} \tilde{S}_{-1}(\mathbf{k} + \mathbf{k}_p) \right]$$

where $\tilde{S}_0$ is the conventional (DC) component and $\tilde{S}_{\pm 1}$ are the shifted frequency bands containing high-resolution information.

For $N_\text{phases} = 3$ phase shifts per angle, this forms a linear system:

$$\begin{pmatrix} \tilde{D}_1 \\ \tilde{D}_2 \\ \tilde{D}_3 \end{pmatrix} = \begin{pmatrix} 1 & e^{j\phi_1} & e^{-j\phi_1} \\ 1 & e^{j\phi_2} & e^{-j\phi_2} \\ 1 & e^{j\phi_3} & e^{-j\phi_3} \end{pmatrix} \begin{pmatrix} \tilde{S}_0 \cdot \tilde{H} \\ \tilde{S}_{+1} \cdot \tilde{H} \\ \tilde{S}_{-1} \cdot \tilde{H} \end{pmatrix}$$

Phase separation is achieved by matrix inversion: $\tilde{S}_j = M^{-1}\tilde{D}$.

### Wiener Recombination

After shifting each band to its correct position in the 2× frequency grid and applying Wiener deconvolution:

$$\hat{S}(\mathbf{k}) = \frac{\sum_i w_i \tilde{S}_i(\mathbf{k}) \tilde{H}_i^*(\mathbf{k})}{\sum_i |\tilde{H}_i(\mathbf{k})|^2 + \alpha \lambda^2}$$

where $w_i$ are modulation-depth-dependent weights and $\lambda$ is the Wiener regularization parameter.

### Hessian Denoising (the inverse problem)

The Wiener-SIM output $y$ is treated as a noisy observation. The Hessian-denoised image $x$ solves:

$$\min_x \; \frac{\mu}{2}\|x - y\|_2^2 + \|\mathcal{H}(x)\|_1$$

where the Hessian norm has 6 components (for 3D data with z-axis weight $\sigma$):

$$\|\mathcal{H}(x)\|_1 = \|D_{xx}x\|_1 + \|D_{yy}x\|_1 + \sigma^2\|D_{zz}x\|_1 + 2\|D_{xy}x\|_1 + 2\sigma\|D_{xz}x\|_1 + 2\sigma\|D_{yz}x\|_1$$

## 3. Noise Model

The simulation (Supplementary Figure 8) applies three noise sources to the raw SIM frames:

1. **Poisson shot noise**: $D_k(\mathbf{r}) \sim \text{Poisson}(H \otimes (S \cdot I_k))$ — photon counting statistics
2. **Constant background**: 99 a.u. added to all pixels — models detector offset and autofluorescence
3. **Gaussian read-out noise**: $n \sim \mathcal{N}(0, 20^2)$ — models sCMOS camera read noise

After SIM reconstruction, the noise becomes approximately **Gaussian** due to linear operations (phase separation, Wiener filtering) on the mixed noise. The Wiener filter amplifies noise in the extended frequency region where OTF is weak. The Hessian denoiser uses an $\ell_2$ data fidelity term, appropriate for this post-reconstruction Gaussian noise.

**Preprocessing**: Background (99 a.u.) must be subtracted from all raw frames before reconstruction. Failure to do so biases the DC component and distorts frequency unmixing.

## 4. Data Description

### Input Format
- **File**: TIFF stack, float32
- **Shape**: $(N_\text{frames}, H, W)$ — e.g., $(72, 256, 256)$
- **Organization**: Frames ordered as $[\text{angle}_0\text{phase}_0, \text{angle}_0\text{phase}_1, \text{angle}_0\text{phase}_2, \text{angle}_1\text{phase}_0, \ldots]$, repeating for each timepoint

### Simulation Data (Supplementary Figure 8)
- **Synthetic specimen**: 512×512×12 pixels at 32.5 nm pixel size, fluorescence intensity range [0, 250] a.u.
- **Camera recording**: sCMOS at 65×65 nm pixel size (2× downsampled), producing 256×256 raw frames
- **72 frames** = 12 timepoints × 6 frames/timepoint (2 angles × 3 phases)
- **Noise**: Poisson + background 99 a.u. + Gaussian SD 20 a.u.
- **Ground truth**: Wiener reconstruction from noiseless raw images (not included in data file)

### Metadata

- **Measured OTF**: `data/metadata/488OTF_512.tif` — measured point spread function of the microscope (100×, NA 1.49 objective), stored as the OTF in frequency domain. Must be used instead of synthetic OTF for accurate reconstruction.

### Parameters

| Parameter | Symbol | Value | Unit | Description |
|-----------|--------|-------|------|-------------|
| `nangles` | $N_\theta$ | 2 | — | Number of illumination angles (2-beam SIM) |
| `nphases` | $N_\phi$ | 3 | — | Number of phase shifts per angle |
| `wavelength` | $\lambda$ | 488 | nm | Emission wavelength (Lifeact-EGFP) |
| `NA_detection` | $\text{NA}_\text{det}$ | 1.49 | — | Detection objective NA (100× CFI Apochromat TIRF) |
| `NA_excitation` | $\text{NA}_\text{exc}$ | 0.9 | — | Excitation NA (for SIM pattern generation) |
| `pixel_size` | $\Delta x$ | 65 | nm | Camera pixel size |
| `specimen_pixel_size` | — | 32.5 | nm | Specimen-space pixel size (= camera / 2) |
| `background` | $b$ | 99 | a.u. | Constant background added to all pixels |
| `weilac` | $\lambda_W$ | 2.0 | — | Wiener regularization parameter |
| `spjg` | — | [4, 4, 3] | — | Phase step ratios between phases |
| `mu` | $\mu$ | 150.0 | — | Hessian/TV data fidelity weight |
| `sigma_z` | $\sigma$ | 1.0 | — | Z/t-axis Hessian weight |
| `lambda` | $\lambda_B$ | 0.5 | — | Bregman splitting parameter |

**Important**: The filename `NA0.9` refers to the **excitation NA** (pattern generation), not the detection NA. The OTF must use the detection NA (1.49) since it determines what the camera can resolve. The excitation NA determines only the pattern spatial frequency.

## 5. Method Hint

**Split-Bregman iteration** is used to solve the $\ell_1$-Hessian minimization because:

1. **Hessian over TV**: Second-order regularization preserves thin curvilinear structures (lines, filaments, tubules) that first-order TV tends to erode. For biological imaging, this is critical — most cellular structures are thin and curved.

2. **Split-Bregman efficiency**: By introducing auxiliary variables $d_i = D_i x$ and Bregman variables $b_i$, the non-smooth $\ell_1$ problem decomposes into:
   - An $x$-update with closed-form FFT solution (diagonal in Fourier domain)
   - A $d$-update via simple soft-thresholding (shrinkage operator)
   - Convergence in $\mathcal{O}(100)$ iterations
   - **Critical**: the `frac` variable (data fidelity + adjoint of regularization) must carry across iterations — the adjoint terms from the $d,b$-updates at the bottom of one iteration must be included in the $x$-update at the top of the next iteration.

3. **Alternatives considered**:
   - **Wiener-only**: Fast but noisy, no artifact suppression
   - **TV denoising**: Available as comparison; smoother but loses fine details
   - **Deep learning**: Requires training data; Hessian-SIM is unsupervised
   - **ADMM**: Mathematically equivalent to Split-Bregman for this problem
