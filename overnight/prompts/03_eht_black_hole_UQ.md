# Task 3: EHT Black Hole Probabilistic Imaging / Uncertainty Quantification (DPI)

## Paper
Sun & Bouman 2020, "Deep Probabilistic Imaging: Uncertainty Quantification and Multi-modal Solution Characterization for Computational Imaging", arXiv:2010.14462v2

## Goal
Reproduce Figures 5, 6, 7 from the paper. The key contribution is Deep Probabilistic Imaging (DPI): using a normalizing flow generative model to learn the posterior distribution of images given interferometric measurements, enabling uncertainty quantification.

## Reference Code
GitHub: https://github.com/HeSunPU/DPI
Website: https://imaging.cms.caltech.edu/dpi/
Clone this repo and study the code. Extract and adapt the relevant functions. Do NOT directly import the DPI package.

## Key Concepts from the Paper
1. **Flow-based generative model**: Real-NVP architecture with 48 affine coupling layers
   - Maps latent z ~ N(0,1) → image x = G_θ(z)
   - Exact log-likelihood via change of variables: log q_θ(x) = log π(z) - log|det(dG_θ/dz)|
2. **KL divergence training loss** (Eq. 7-9):
   θ* = argmin E_{z~N(0,1)} { L(y, f(G_θ(z))) + λR(G_θ(z)) - β log|det(dG_θ(z)/dz)| }
   - Data fidelity: L(y, f(x)) — chi-squared between measurements and forward model
   - Regularizer: R(x) — image prior (e.g., TV, Gaussian)
   - Entropy: -log|det(...)| encourages diversity in generated samples
3. **Convex case** (calibrated visibilities): Linear forward model, Gaussian likelihood (Eq. 16)
4. **Non-convex case** (closure quantities): Nonlinear forward model with closure phases and amplitudes (Eq. 15)
5. **Architecture A** (linear output) vs **Architecture B** (softplus output, non-negative images)

## What the Task Should Implement
1. **preprocessing.py**: Load EHT observation data, compute closure quantities if needed, prepare measurement vectors
2. **physics_model.py**:
   - Linear DFT forward model (visibilities)
   - Closure phase and closure amplitude forward operators
   - Noise covariance matrix construction from SEFDs
   - Chi-squared data fidelity terms (both convex and non-convex)
3. **solvers.py** (THIS IS THE CORE):
   - **Real-NVP normalizing flow**: 48 affine coupling layers
     - Affine coupling layer: split input, apply scale-shift transform
     - Scale-shift network: U-Net style with 5 hidden layers, LeakyReLU, batch norm, skip connections
     - Random shuffle between layers
   - **DPI training loop**:
     - Sample z ~ N(0,1), generate x = G_θ(z)
     - Compute loss = data_fidelity + λ*regularizer - β*log_det_jacobian
     - Optimize with Adam, batch size 32, ~20000 epochs
   - **Posterior sampling**: After training, sample z's and push through G_θ to get posterior image samples
   - **Posterior statistics**: Compute mean, std, covariance from samples
4. **visualization.py**: Plot posterior mean, std deviation, sample images, corner plots of features
5. **generate_data.py**: Generate synthetic black hole image + noisy visibilities

## Figures to Reproduce
- **Figure 5**: DPI posterior mean, std dev, and covariance for convex reconstruction (calibrated visibilities). Compare with analytical Gaussian posterior.
- **Figure 6**: DPI results for non-convex reconstruction (closure quantities). Show posterior samples capturing multi-modal solutions.
- **Figure 7**: DPI applied to real EHT M87* data (or realistic simulation). Show posterior uncertainty on black hole features.

## Data Generation
- Ground truth: 32×32 black hole crescent image (blurred to interferometer resolution)
- Synthetic visibilities from 9-station EHT array
- Both calibrated (complex) and uncalibrated (closure) versions
- Noise based on telescope SEFDs

## Requirements (GPU)
This task requires PyTorch with GPU support:
```
numpy
scipy
matplotlib
torch
torchvision
```

## Format
Follow EXACTLY the pilot task format. Directory: tasks/eht_black_hole_UQ/

## Critical Constraints
- Do NOT import the DPI package at runtime. All necessary functions must be self-contained in src/.
- You SHOULD clone the DPI repo (https://github.com/HeSunPU/DPI) to /tmp/ and extract/copy the relevant normalizing flow, training loop, and forward model code. Copying code is encouraged — the goal is to "clean" the original implementation into our standardized format, not to rewrite from scratch.
- Adapt the extracted code to fit our directory structure and function signatures, remove unnecessary dependencies, but preserve the original algorithm logic as faithfully as possible.
- The Real-NVP architecture must match: 48 affine coupling layers, U-Net scale-shift functions.
- The entropy term (-β * log_det_jacobian) is CRITICAL — without it the model collapses to a point estimate.
- Must produce posterior SAMPLES, not just a point estimate. This is the whole point of DPI.
- GPU training code must work (use torch.cuda.is_available() for device selection).
- Image size: 32×32 pixels (as in the paper).
- Keep existing PDFs and reference_website_github.md.
- Run `python -m pytest evaluation/tests/ -v` at the end.
