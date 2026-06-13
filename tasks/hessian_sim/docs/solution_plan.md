# Solution Plan: Hessian-SIM Reconstruction

## 1. Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  Raw SIM Frames (72, 256, 256) + Measured OTF (512, 512)            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 0: Background Subtraction  (main.py)                         │
│  raw -= 99; raw[raw<0] = 0  (paper: background = 99 a.u.)          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Parameter Estimation  (preprocessing.py)                  │
│  Cross-correlation → sub-pixel refinement → zuobiaox, zuobiaoy      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: Modulation & Phase Estimation  (preprocessing.py)         │
│  EMD histogram smoothing → c6 (modulation depth), angle6 (phase)    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: Wiener-SIM Reconstruction  (preprocessing.py)             │
│  Phase separation → frequency shift (Moiré) → Wiener recombination  │
│  Output: super-resolved stack y (12, 512, 512)                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Stage 4: Hessian    │  │  Stage 5: TV         │
│  Denoising           │  │  Denoising           │
│  (solver.py)         │  │  (solver.py)         │
│  Split-Bregman,      │  │  Split-Bregman,      │
│  2nd order, 100 iter │  │  1st order, 100 iter │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Hessian-SIM output  │  │  TV-SIM output       │
│  (12, 512, 512)      │  │  (12, 512, 512)      │
└──────────────────────┘  └──────────────────────┘
```

## 2. Optimization Problems

### 2.1 Wiener-SIM Reconstruction

**Phase separation**: Given $N_\phi = 3$ phase-shifted images per angle, the phase matrix $M$ encodes the relationship:

$$\mathbf{D} = M \cdot \mathbf{S}, \quad M = \begin{pmatrix} 1 & e^{j\phi_1} & e^{-j\phi_1} \\ 1 & e^{j\phi_2} & e^{-j\phi_2} \\ 1 & e^{j\phi_3} & e^{-j\phi_3} \end{pmatrix}$$

Separated components: $\tilde{S}_j = M^{-1} \tilde{D}$

**Frequency shift** (Moiré unmixing): Each separated component $\tilde{S}_{\pm 1}$ is shifted to its correct frequency position by multiplication in real space:

$$\tilde{S}_{\pm 1}^{\text{shifted}}(\mathbf{k}) = \mathcal{F}\left[\mathcal{F}^{-1}[\tilde{S}_{\pm 1}] \cdot e^{\pm j\mathbf{k}_p \cdot \mathbf{r}}\right]$$

This places the high-frequency information at the correct position in the 2× expanded frequency grid.

**Wiener recombination**:

$$\hat{S}(\mathbf{k}) = \frac{\sum_{i=1}^{N_\theta \cdot N_\phi} w_i \cdot \tilde{S}_i(\mathbf{k}) \cdot \tilde{H}_i^*(\mathbf{k})}{\sum_{i=1}^{N_\theta \cdot N_\phi} |\tilde{H}_i(\mathbf{k})|^2 + 0.005 \cdot N_\theta \cdot \lambda_W^2} \cdot A(\mathbf{k})$$

where $w_i$ are the modulation-depth weights (xishu), $\tilde{H}_i$ are the shifted OTFs, and $A(\mathbf{k}) = \cos(\pi|\mathbf{k}|/2k_{\max})$ is the apodization function.

### 2.2 Hessian Denoising (Main Inverse Problem)

**Objective function**:

$$\min_x \; \frac{\mu}{2}\|x - y\|_2^2 + \|D_{xx}x\|_1 + \|D_{yy}x\|_1 + \sigma^2\|D_{zz}x\|_1 + 2\|D_{xy}x\|_1 + 2\sigma\|D_{xz}x\|_1 + 2\sigma\|D_{yz}x\|_1$$

where $D_{ij}$ are second-order finite difference operators.

**Split-Bregman formulation**: Introduce auxiliary variables $d_i$ and Bregman variables $b_i$ for $i \in \{xx, yy, zz, xy, xz, yz\}$:

$$\min_{x, d_i} \; \frac{\mu}{2}\|x - y\|_2^2 + \sum_i \alpha_i \|d_i\|_1 + \frac{\lambda}{2}\sum_i \alpha_i \|d_i - D_i x - b_i\|_2^2$$

where $\alpha_i \in \{1, 1, \sigma^2, 2, 2\sigma, 2\sigma\}$.

## 3. Solver Derivation

### x-update (linear, solved in Fourier domain)

Setting $\nabla_x = 0$:

$$\hat{x} = \mathcal{F}^{-1}\left[\frac{(\mu/\lambda)\hat{y} + \sum_i \alpha_i \hat{D}_i^H(\hat{d}_i - \hat{b}_i)}{(\mu/\lambda) + \sum_i \alpha_i |\hat{D}_i|^2}\right]$$

The denominator $(\mu/\lambda) + \sum_i \alpha_i |\hat{D}_i|^2$ is precomputed once since it depends only on the operator spectra, not the data.

### d-update (soft thresholding / proximal operator)

For each component $i$:

$$d_i = \text{shrink}(D_i x + b_i, \; 1/\lambda) = \text{sign}(u) \cdot \max(|u| - 1/\lambda, \; 0)$$

where $u = D_i x + b_i$.

### b-update (Bregman variable)

$$b_i \leftarrow b_i + (D_i x - d_i)$$

### Iteration structure (CRITICAL)

The `frac` variable must **carry across iterations**. The numerator of the x-update has two parts: the data fidelity term $(\mu/\lambda)\hat{y}$ and the adjoint of the regularization $\sum_i \alpha_i D_i^T(d_i - b_i)$. The adjoint terms are computed at the bottom of each iteration and must be included in the x-update at the top of the **next** iteration:

```
frac = (mu/lambda) * y                          # initial: data only
for it = 1 to N:
    x = IFFT[ FFT(frac) / divide ]              # x-update uses frac from PREVIOUS iter
    frac = (mu/lambda) * y                       # reset to data term
    for each Hessian component i:
        d_i = shrink(D_i x + b_i, 1/lambda)     # d-update
        b_i += D_i x - d_i                       # b-update
        frac += alpha_i * D_i^T(d_i - b_i)      # accumulate adjoint → NEXT iter's x-update
```

**Common bug**: if `frac` is reset at the **top** of the loop (before the x-update), the adjoint terms are discarded and the regularization has no effect on x. The data fidelity overwhelms, producing x ≈ y.

### TV Denoising

Same structure but with first-order operators $D_x, D_y$ only (zbei = 0 disables z-axis):

$$\min_x \; \frac{\mu}{2}\|x - y\|_2^2 + \|D_x x\|_1 + \|D_y x\|_1$$

## 4. Code Structure Table

| Function | File | Mathematical Operation |
|----------|------|----------------------|
| `generate_otf()` | `physics.py` | $H(k) = \max(0, 1 - |k|/k_c)$ (synthetic, for reference only) |
| `shift_otf(H, kx, ky, n)` | `physics.py` | $\mathcal{F}[\mathcal{F}^{-1}[H] \cdot e^{j(k_x x + k_y y)}]$ |
| `dft_conv(h, g)` | `physics.py` | $\mathcal{F}^{-1}[\mathcal{F}[h] \cdot \mathcal{F}[g]]$ (linear conv) |
| `pad_to_size(a, shape)` | `physics.py` | Zero-pad centered: $n \to 2n$ |
| `_fdiff(x, dim)` | `solver.py` | $(D_+ x)_i = x_{i+1} - x_i$ (module-private) |
| `_bdiff(x, dim)` | `solver.py` | $(D_- x)_i = x_i - x_{i-1}$ (module-private) |
| `compute_merit(...)` | `physics.py` | $\sum \tilde{S}_0^* \cdot \tilde{S}_{+1}^{\text{shifted}} / N_{\text{overlap}}$ |
| `emd_decompose(s)` | `physics.py` | EMD: $s = \sum_k \text{IMF}_k + r$ |
| `estimate_sim_parameters()` | `preprocessing.py` | Cross-correlation peak $\to$ sub-pixel refinement of $\mathbf{k}_p$ |
| `estimate_modulation_and_phase()` | `preprocessing.py` | EMD histogram $\to$ peak: $c_6$ (depth), $\phi_6$ (phase) |
| `wiener_sim_reconstruct()` | `preprocessing.py` | $M^{-1}\tilde{D} \to e^{j\mathbf{k}_p \cdot \mathbf{r}}$ shift $\to$ Wiener filter |
| `running_average(stack, w)` | `preprocessing.py` | $\bar{x}_t = \frac{1}{w}\sum_{i=-w/2}^{w/2} x_{t+i}$ |
| `hessian_denoise(y, mu, sigma, lamda)` | `solver.py` | Split-Bregman: $x, d_i, b_i$ updates (2nd order, $\lambda_B$=0.5) |
| `tv_denoise(y, mu)` | `solver.py` | Split-Bregman: $x, d, b$ updates (1st order) |
| `plot_comparison(...)` | `visualization.py` | 5-panel image + FFT (widefield, raw, Wiener, Hessian, TV) |
| `plot_line_profiles(...)` | `visualization.py` | Center-row/col intensity profiles |
| `plot_hessian_vs_tv(...)` | `visualization.py` | Zoomed Hessian vs TV + difference map |

## 5. Data Flow Diagram

```
Raw TIF (72, 256, 256) float32 + Measured OTF (512, 512) uint16
    │
    ├──► Background subtraction: raw -= 99; clip to 0  [main.py]
    │
    ├──► estimate_sim_parameters()
    │       │ Input:  raw[:6] + measured OTF (512,512)
    │       │ Steps:  avg frames → FFT → phase separate → cross-correlate
    │       │         → bandpass filter → sub-pixel refine
    │       └──► zuobiaox (6,1), zuobiaoy (6,1)
    │
    ├──► estimate_modulation_and_phase()
    │       │ Input:  raw[:6] + measured OTF + zuobiao
    │       │ Steps:  dual-cutoff OTF (fc_ang=120, fc_con=105) → shift
    │       │         → complex ratio map → histogram → EMD smooth → peak find
    │       └──► c6 (4,), angle6 (4,)
    │
    └──► wiener_sim_reconstruct()  [per timepoint, 12 total]
            │ Input:  raw[t*6:(t+1)*6] + measured OTF + zuobiao + c6 + angle6
            │ Steps:  sigmoid mask → zero-pad (256→512)
            │         → FFT → phase separate (M^{-1} with per-angle deph)
            │         → OTF mask → zero-pad (512→1024)
            │         → frequency shift (* Irtest)  ← Moiré unmixing
            │         → OTF support mask → phase correct (/ re)
            │         → Wiener combine (Σ w·S·H* / (ΣH² + αλ²))
            │         → apodize → IFFT → crop (1024→512)
            └──► sim_result (12, 512, 512) float32
                    │
                    ├──► hessian_denoise()
                    │       │ Steps:  normalize → precompute FFT(D_ij)
                    │       │         → 100 iterations:
                    │       │           x = F^{-1}[frac / divide]
                    │       │           d_i = shrink(D_i x + b_i, 1/λ)
                    │       │           b_i += D_i x - d_i
                    │       └──► hessian_result (12, 512, 512)
                    │
                    └──► tv_denoise()
                            │ Steps:  normalize → precompute FFT(D_x, D_y)
                            │         → 100 iterations:
                            │           x = F^{-1}[frac / divide]
                            │           d = shrink(Dx + b, 1/λ)
                            │           b += Dx - d
                            └──► tv_result (12, 512, 512)
```
