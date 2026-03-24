# EHT Black Hole Tomography (BH-NeRF)

Recover the 3D volumetric emission structure of a black hole accretion flow from time-series interferometric measurements using a neural radiance field (NeRF) with Keplerian orbital dynamics and gravitational lensing.

| Attribute | Value |
|-----------|-------|
| Domain | Astronomy |
| Modality | Radio interferometry (VLBI) |
| Difficulty | Hard |

## Background

The Event Horizon Telescope (EHT) captures sparse radio interferometric measurements of supermassive black holes. Orbiting hot plasma ("hotspots") near the black hole create time-varying emission that is gravitationally lensed and Doppler-boosted. The challenge is to recover the full 3D emission field from these sparse 2D projections, leveraging the physical constraint that the emission follows Keplerian orbital dynamics.

## Physical Model

### Emission representation
The static 3D emission field is represented as a continuous function parameterized by an MLP neural network with positional encoding:

$$e_0(\mathbf{x}) = \text{sigmoid}(\text{MLP}_\theta(\gamma(\mathbf{x})) - 10)$$

where $\gamma(\mathbf{x})$ is a positional encoding with $L=3$ frequency bands.

### Keplerian dynamics
The emission at time $t$ is obtained by inverse-rotating the query coordinates according to Keplerian angular velocity:

$$e(t, \mathbf{x}) = e_0(R_{\xi}(-\Omega(r) \cdot t) \cdot \mathbf{x})$$

where $\Omega(r) = \sqrt{M}/r^{3/2}$ is the Keplerian angular velocity and $\xi$ is the (unknown) rotation axis.

### Gravitational lensing
Light rays near the black hole follow curved paths in the Schwarzschild metric. Pre-computed ray paths $\Gamma_n$ for each image pixel trace through the 3D volume. The pixel intensity is:

$$I_n(t) = \sum_i g^2(\mathbf{x}_i) \cdot e(t, \mathbf{x}_i) \cdot \Delta\tau_i \cdot \Sigma_i$$

where $g$ is the Doppler factor, $\Delta\tau$ is the proper time step, and $\Sigma = r^2$ for Schwarzschild.

### Measurement model
The EHT measures complex visibilities via discrete Fourier transform:

$$\mathbf{y}(t) = \mathbf{A}(t) \cdot \mathbf{I}(t) + \boldsymbol{\varepsilon}$$

### Optimization
Joint optimization over MLP weights $\theta$ and rotation axis $\xi$:

$$\mathcal{L}(\theta, \xi) = \sum_t \left\| \mathbf{y}(t) - \mathbf{A}(t) \cdot \mathbf{I}_{\theta,\xi}(t) \right\|^2_{\Sigma}$$

## Data Description

### `data/raw_data.npz`
| Key | Shape | Description |
|-----|-------|-------------|
| `ray_x`, `ray_y`, `ray_z` | (num_alpha, num_beta, ngeo) | 3D Cartesian coordinates along rays |
| `ray_r` | (num_alpha, num_beta, ngeo) | Radial coordinate |
| `ray_t_geo` | (num_alpha, num_beta, ngeo) | Coordinate time along ray (slow light) |
| `ray_dtau` | (num_alpha, num_beta, ngeo) | Mino time differential |
| `ray_Sigma` | (num_alpha, num_beta, ngeo) | Metric factor (r^2 for Schwarzschild) |
| `Omega` | (num_alpha, num_beta, ngeo) | Keplerian angular velocity on ray points |
| `g_doppler` | (num_alpha, num_beta, ngeo) | Doppler boosting factor |
| `t_frames` | (n_frames,) | Observation times in units of M |
| `images_true` | (n_frames, num_alpha, num_beta) | Ground truth image-plane movie |
| `emission_true` | (res, res, res) | Ground truth 3D emission at t=0 |
| `rot_axis_true` | (3,) | True rotation axis |

### `data/meta_data`
JSON file with simulation parameters (spin, inclination, fov, resolution) and training hyperparameters (net_depth, net_width, posenc_deg, lr, n_iters).

## Method Hint

Use a coordinate-based neural network (NeRF) with:
- Positional encoding (L=3 frequency bands)
- 4-layer MLP with 128 hidden units and skip connections
- Keplerian velocity warp for temporal dynamics
- Volume rendering along pre-computed gravitational lensed ray paths
- Adam optimizer with polynomial learning rate schedule
- Multi-initialization of the rotation axis to avoid symmetric local minima

## References

1. Levis, A., Bharadwaj, P., Dugan, K.P., Vardhan, R., Dahlbom, D., Bouman, K.L. (2022). "Gravitationally Lensed Black Hole Emission Tomography." CVPR 2022. [arXiv:2204.03715]
2. Levis, A., et al. (2024). "Orbital polarimetric tomography of a flare near the Sagittarius A* supermassive black hole." Nature Astronomy.
