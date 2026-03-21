# Task 5: EHT Black Hole 4D Tomographic Reconstruction (BH-NeRF)

## Papers
1. Levis et al. 2022, "Gravitationally Lensed Black Hole Emission Tomography", CVPR 2022 (PRIMARY — implement this first)
2. Levis et al. 2024, Nature Astronomy (SECONDARY — add as additional test if possible)

## Goal
Reproduce Figures 7 and 8 from the CVPR 2022 paper. The key contribution is BH-NeRF: using neural radiance fields to recover the 3D+time (4D) emission field around a black hole from sparse EHT measurements, accounting for gravitational lensing.

## Reference Code
GitHub: https://github.com/aviadlevis/bhnerf
Website: https://imaging.cms.caltech.edu/bhnerf/
Clone and study. The code uses JAX. You may implement in either JAX or PyTorch.

## Key Concepts from the Paper
1. **3D emission field**: e(t, x) as continuous function of time and 3D coordinate, parameterized by MLP neural network
   - e_0(x) = MLP_θ(γ(x)) where γ is positional encoding (Eq. 7-8)
   - MLP: 4 layers, 128 units wide, ReLU activations
   - Positional encoding: L=3 (low degree, suitable for smooth volumetric fields)

2. **Keplerian orbital dynamics** (Eq. 1-3):
   - Angular velocity: ω(r) = 1/(2π√(r³/GM)) ∝ r^{-3/2}
   - Shearing: inner orbits move faster → structure stretches over time
   - e(t, x) = e_0(R_{ξ,φ} x) where φ(t,r) = tω(r)
   - Rotation axis ξ is unknown, jointly optimized

3. **Gravitational lensing ray tracing** (Section 3.2):
   - Light rays near black hole follow curved paths (General Relativity)
   - Pre-computed ray paths Γ_n for each image pixel
   - Pixel intensity: p_n(t) = Σ e(t, x_i) Δs_i along curved ray path

4. **EHT measurement model** (Eq. 6):
   - y(t) = F_t I(t) + ε where F_t is time-varying DFT matrix
   - Different uv-coverage at each time frame

5. **Optimization** (Eq. 9):
   - L(θ, ξ) = Σ_t ||y(t) - F_t I_{θ,ξ}(t)||²_Σ
   - Joint optimization over MLP weights θ and rotation axis ξ

6. **Symmetric local minimum** (Figure 6):
   - Initialize ξ from both hemispheres, pick solution with lower loss

## What the Task Should Implement
1. **preprocessing.py**: Load time-stamped EHT measurements, organize by observation time
2. **physics_model.py** (CORE — the forward model is complex):
   - **Keplerian dynamics**: Compute rotation angle φ(t,r) for shearing emission
   - **Rotation transformation**: R_{ξ,φ} rotation matrix about axis ξ by angle φ
   - **Gravitational lensing ray tracer**:
     - Pre-compute curved ray paths in Schwarzschild/Kerr metric
     - Or use simplified ray tracing (straight rays + lensing correction)
     - For each image pixel, trace ray through 3D volume
   - **Volume rendering**: Integrate emission along each ray path
   - **DFT measurement**: Image → complex visibilities at sampled (u,v) points
3. **solvers.py**:
   - **BH-NeRF**: MLP network for volumetric emission
     - Positional encoding with L=3
     - 4 layers, 128 units, ReLU
   - **Training loop**: Adam optimizer, jointly optimize θ (MLP weights) and ξ (rotation axis)
   - **Multi-initialization**: Run from multiple ξ initializations to avoid symmetric local minima
4. **visualization.py**:
   - 3D volume rendering at different times
   - 2D projected images vs ground truth
   - Recovered rotation axis vs true axis
   - Loss curves
5. **generate_data.py**: Generate synthetic hotspot observation:
   - SgrA*-like black hole (M = 4×10⁶ solar masses, zero angular momentum)
   - 1-3 orbiting hotspots at r ≈ 1.16 × r_ms (marginally stable orbit)
   - Gaussian hotspots with σ = 0.4 × GM/c²
   - 128 temporal frames
   - 64³ voxel grid for ground truth
   - EHT 2017 array uv-coverage

## Figures to Reproduce
- **Figure 7**: Single hotspot reconstruction — show recovered 3D emission at multiple time steps vs ground truth. Compare EHT vs ngEHT coverage.
- **Figure 8**: Multiple hotspot reconstruction — show that method handles multiple orbiting features.

## Data Generation (Simplified)
Since full GR ray tracing is complex, acceptable simplifications:
- Use pre-computed ray paths (can approximate with analytical formulas for Schwarzschild metric)
- Or use straight-ray approximation with gravitational deflection correction
- Generate ground truth emission as 3D Gaussian hotspots in Keplerian orbit
- Project through (simplified) ray paths to get images
- Apply DFT at EHT baselines to get visibilities

## Requirements (GPU)
```
numpy
scipy
matplotlib
torch  # or jax + jaxlib
```

## Format
Follow EXACTLY the pilot task format. Directory: tasks/eht_black_hole_tomography/

## Critical Constraints
- Do NOT import the bhnerf package at runtime. All necessary functions must be self-contained in src/.
- You SHOULD clone the bhnerf repo (https://github.com/aviadlevis/bhnerf) to /tmp/ and extract/copy the relevant NeRF, ray tracing, Keplerian dynamics, and forward model code. Copying code is encouraged — the goal is to "clean" the original implementation into our standardized format, not to rewrite from scratch.
- Adapt the extracted code to fit our directory structure and function signatures, remove unnecessary dependencies, but preserve the original algorithm logic as faithfully as possible.
- The implementation should use JAX or PyTorch (your choice, but must support GPU).
- Gravitational lensing ray tracing can be simplified but must be present (not just straight rays).
- Keplerian shearing dynamics are essential — the whole point is 4D (3D+time) reconstruction.
- Rotation axis ξ must be jointly optimized (not assumed known).
- Multi-initialization strategy for avoiding symmetric local minima.
- If full GR ray tracing is too complex, use analytical approximations for Schwarzschild metric.
- Keep existing PDFs and reference_website_github.md.
- Run `python -m pytest evaluation/tests/ -v` at the end.
- For the Nature Astronomy paper test (secondary): if time permits, add real M87* or SgrA* data test.
