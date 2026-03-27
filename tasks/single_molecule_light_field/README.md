# Single-Molecule Light Field Microscopy (SMLFM): 3D Super-Resolution Reconstruction

## Problem

Fourier Light Field Microscopy (FLFM) is a technique that achieves 3D single-molecule super-resolution by replacing the standard camera with a micro-lens array (MLA) placed at the back focal plane of the imaging system.

Each micro-lens samples a different **view** of the sample — a different angle of incidence — so the same fluorescent molecule simultaneously appears at slightly different lateral positions in several sub-images. The lateral shift between views (parallax) carries depth information: molecules above or below focus shift in opposite directions, and the shift magnitude encodes how far they are from focus.

The computational task is the **inverse problem**: given 2D localisations (x, y) detected by standard SMLM fitting software (e.g. ThunderStorm or PeakFit) across thousands of camera frames, recover the 3D positions (x, y, z) of individual fluorescent molecules.

## Imaging Physics

The Fourier relay maps the objective back focal plane (BFP) onto the MLA plane. Each micro-lens therefore samples a unique solid angle of the emission cone, parameterised by normalised pupil coordinates (u, v):

```
  u = (x_bfp / BFP_radius),  v = (y_bfp / BFP_radius)
  BFP_radius = NA * f_obj * (f_fourier / f_tube)  [microns]
```

A point source at 3D position (x₀, y₀, z) forms an image at position (xᵢ, yᵢ) through microlens i:

```
  xᵢ = x₀  +  uᵢ / ρ  +  z · αᵤᵢ
  yᵢ = y₀  +  vᵢ / ρ  +  z · αᵥᵢ
```

where:
- **ρ** (rho_scaling) = magnification / BFP_radius  converts image-plane microns → normalised pupil coords
- **α(u, v)** is the angular sensitivity — how much a unit depth change shifts the apparent lateral position in a given view. For a spherical wavefront:

```
  α(u, v)  =  -(NA/n) / √(1 − ρ² (NA/n)²)  ×  (u, v)
```

The most accurate model (INTEGRATE_SPHERE) phase-averages this expression over the finite aperture of each micro-lens.

## Dataset

`data/raw_data.csv` — 2D localisations from a fixed BCR (B-cell receptor) B cell, imaged on a hexagonal FLFM. Localised using ImageJ GDSC SMLM2 PeakFit plugin.

- **150,996 localisations** in **5,000 frames**
- Hexagonal MLA: 7 views in the back focal plane
- 60× water-immersion objective, NA 1.27

Source: `hexSMLFM` repository (TheLeeLab, University of Cambridge).
Reference paper: [R. R. Sims et al., Optica 7, 1065 (2020)](https://doi.org/10.1364/OPTICA.397172)

## Method Hints

### Pipeline steps

1. **Load & centre** 2D localisations (pixels → microns; subtract mean X, Y)
2. **Build optics model** — `FourierMicroscope` derives rho_scaling, magnification, etc.
3. **Build MLA lattice** — hexagonal grid, rotated by `mla_rotation` degrees to match data
4. **Assign lenses** — each localisation mapped to its nearest micro-lens by k-NN in (u, v) space
5. **Filter** — remove localisations outside BFP, or outside rho/spot-size/photon thresholds
6. **Compute α** — phase-averaged sphere model (INTEGRATE_SPHERE)
7. **Aberration correction** — first-pass fitting on first 1000 frames, compute per-view systematic offset
8. **Full 3D fitting** — overdetermined linear system solved by OLS for every frame
9. **Z calibration** — multiply raw optical Z by `z_calib` (determined from bead scans)

### Key fitting algorithm

For each camera frame, for each candidate seed localisation:

1. Search for companion localisations in other views that satisfy:
   - Angle between (Δx, Δy) and (Δu, Δv) within `angle_tolerance`
   - Disparity `(dxy − duv/ρ) / duv` within `dist_search` of best-fit Z
2. Collect matched group; solve OLS for (x₀, y₀, z)
3. Accept if fit residual < `threshold`; remove used localisations from pool
4. Repeat until fewer than `min_views` candidates remain

### MLA rotation

For this dataset `mla_rotation = 30.8°`. The MLA lattice is rotated (not the data) so the lens centres align with the observed localisations. The alignment can be verified visually.

## Evaluation

| Metric | Value |
|--------|-------|
| 3D localisations (total) | 24,778 |
| 3D localisations (filtered: lateral err <200 nm, ≥4 views) | 21,931 |
| Median lateral fit error | ~52 nm |
| Median axial fit error | ~55 nm |
| Median views per molecule | 6 |
| Z range | −3.3 to +5.7 µm |

These values match the precision reported in Sims et al. (2020) for similar acquisition conditions.
