# Task 4: EHT Black Hole Feature Extraction (α-DPI)

## Paper
Sun et al. 2022, "α-deep Probabilistic Inference (α-DPI): Efficient Uncertainty Quantification from Exoplanet Astrometry to Black Hole Feature Extraction", ApJ 932:99

## Goal
Reproduce Figure 7 from the paper. The key contribution is α-DPI applied to black hole feature extraction: fitting a geometric crescent model to EHT closure quantities to infer posterior distributions of black hole geometric parameters (diameter, width, asymmetry, position angle).

## Reference Code
GitHub: https://github.com/HeSunPU/DPI (same repo as Task 3, different application)
Website: https://imaging.cms.caltech.edu/dpi/

## Key Concepts from the Paper
1. **α-DPI**: Two-step algorithm combining α-divergence VI + importance sampling
   - Step 1: Train normalizing flow with α-divergence loss (Eq. 1):
     θ* = argmin (1/N) Σ [exp[(1-α)(log p(y|x_n) + log p(x_n) - log q_θ(x_n))]]
   - Step 2: Importance sampling to reweight samples: w(x_j) = p(y|x_j)p(x_j) / q_θ(x_j)
2. **Geometric crescent model** for black hole:
   - Asymmetric ring (crescent) + elliptical Gaussians for extended emission
   - Parameters: diameter, width, asymmetry, position angle, central emission
   - Model selection via ELBO to choose number of Gaussian components (0-3)
3. **Closure quantities as data**: closure phases and log closure amplitudes
   - Robust to station-based calibration errors
   - Nonlinear forward model from geometric parameters → image → closure quantities
4. **α parameter**: Controls exploration vs exploitation. α=0.5 works well; α=1.0 is KL divergence (prone to mode collapse)

## What the Task Should Implement
1. **preprocessing.py**: Load EHT data, compute closure phases and closure amplitudes, compute noise estimates (σ_ψ, σ_C from SEFDs)
2. **physics_model.py**:
   - **Geometric crescent model**: Parameterized black hole image from (diameter, width, asymmetry, position_angle, ...) → 2D image
   - **Elliptical Gaussian model**: For extended emission components
   - **Combined model**: Crescent + N elliptical Gaussians → image
   - **Image → closure quantities**: DFT → visibilities → closure phases/amplitudes
   - All must be differentiable (for gradient-based optimization with PyTorch)
3. **solvers.py** (CORE):
   - **Normalizing flow** (same Real-NVP architecture as DPI, but mapping to geometric parameter space, not pixel space)
   - **α-divergence training loss** (Eq. 1 in paper)
   - **Importance sampling** (Step 2): reweight flow samples by p(y|x)p(x)/q_θ(x)
   - **ELBO computation** for model selection (Eq. 4)
   - Training with Adam optimizer
4. **visualization.py**:
   - Corner plots of posterior parameters (diameter, width, asymmetry, position angle)
   - Posterior image samples
   - ELBO vs model complexity
   - Closure quantity fits (observed vs model-predicted)
5. **generate_data.py**: Generate synthetic EHT observation of crescent + Gaussians model

## Figure to Reproduce
- **Figure 7**: α-DPI applied to simulated EHT data of black hole crescent model
  - ELBO comparison for models with 0, 1, 2, 3 Gaussian ellipses (top panel)
  - Corner plot of posterior geometric parameters: diameter, width, asymmetry, position angle (bottom panel)
  - Show that α-DPI recovers true parameters and captures posterior uncertainty

## Data Generation
- Ground truth geometric model: crescent (d=40μas, w=10μas, asymmetry, PA) + 2 elliptical Gaussians
- Generate synthetic closure phases and closure amplitudes for 9-station EHT array
- Noise from realistic SEFDs
- Image size for rendering: 64×64 pixels, FOV ~160 μas

## Requirements (GPU)
```
numpy
scipy
matplotlib
torch
```

## Format
Follow EXACTLY the pilot task format. Directory: tasks/eht_black_hole_feature_extraction/

## Critical Constraints
- Do NOT import the DPI package at runtime. All necessary functions must be self-contained in src/.
- You SHOULD clone the DPI repo (https://github.com/HeSunPU/DPI) to /tmp/ and extract/copy the relevant α-DPI, normalizing flow, geometric model, and forward model code. Copying code is encouraged — the goal is to "clean" the original implementation into our standardized format, not to rewrite from scratch.
- Adapt the extracted code to fit our directory structure and function signatures, remove unnecessary dependencies, but preserve the original algorithm logic as faithfully as possible.
- The geometric crescent model must be differentiable (use PyTorch tensors throughout).
- α-divergence (NOT just KL divergence) is essential — implement the full α-divergence loss.
- Importance sampling step is critical for accurate posteriors.
- ELBO computation for model selection must be included.
- The parameter space is low-dimensional (~4-10 params), not pixel space like Task 3.
- Keep existing PDFs and reference_website_github.md.
- Run `python -m pytest evaluation/tests/ -v` at the end.
