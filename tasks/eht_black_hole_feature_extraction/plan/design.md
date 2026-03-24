# Code Architecture: α-DPI Feature Extraction

## Module Overview

```
src/
├── preprocessing.py    # Data loading and closure quantity extraction
├── physics_model.py    # Geometric models and NUFFT forward model
├── solvers.py          # Real-NVP flow and α-DPI solver
├── visualization.py    # Corner plots, ELBO, metrics
└── generate_data.py    # Synthetic data generation
```

## Data Flow

```
obs.uvfits → preprocessing → closure_indices + nufft_params
                                    ↓
z ~ N(0,I) → RealNVP.reverse → sigmoid → GeometricModel → image
                                    ↓
                    image → NUFFTForwardModel → (vis, cphase, logcamp)
                                    ↓
                    Loss_angle_diff(cphase) + Loss_logca_diff2(logcamp)
                                    ↓
                    α-divergence reweighting → gradient update
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str = "data") -> dict
def load_ground_truth(data_dir: str, npix: int, fov_uas: float) -> np.ndarray
def load_metadata(data_dir: str = "data") -> dict
def extract_closure_indices(obs, snrcut: float = 0.0) -> dict
def compute_nufft_params(obs, npix: int, fov_uas: float) -> dict
def estimate_flux(obs) -> float
def prepare_data(data_dir: str = "data") -> tuple
```

### physics_model.py

```python
class SimpleCrescentParam2Img(nn.Module):
    def __init__(self, npix, fov=120, r_range=[10,40], width_range=[1,40])
    def compute_features(self, params) -> tuple  # (r, sigma, s, eta)
    def forward(self, params) -> Tensor  # (B, npix, npix)

class SimpleCrescentNuisanceParam2Img(nn.Module):
    def __init__(self, npix, n_gaussian=1, fov=120, ...)
    def compute_features(self, params) -> tuple
    def forward(self, params) -> Tensor  # (B, npix, npix)

class NUFFTForwardModel(nn.Module):
    def __init__(self, npix, ktraj_vis, pulsefac_vis, ...)
    def forward(self, images) -> (vis, visamp, cphase, logcamp)

def Loss_angle_diff(sigma, device) -> Callable
def Loss_logca_diff2(sigma, device) -> Callable
def Loss_visamp_diff(sigma, device) -> Callable
```

### solvers.py

```python
class ActNorm(nn.Module): ...
class ZeroFC(nn.Module): ...
class AffineCoupling(nn.Module): ...
class Flow(nn.Module): ...
class RealNVP(nn.Module):
    def forward(self, input) -> (output, logdet)
    def reverse(self, out) -> (input, logdet)

class AlphaDPISolver:
    def __init__(self, npix, fov_uas, n_flow, ..., alpha, beta, ...)
    def reconstruct(self, obs_data, closure_indices, nufft_params, flux_const) -> dict
    def sample(self, n_samples) -> dict
    def extract_physical_params(self, params_unit) -> np.ndarray
    def importance_resample(self, obs_data, closure_indices, nufft_params, n_samples) -> dict
    def compute_elbo(self, obs_data, closure_indices, nufft_params, n_samples) -> float
```

### visualization.py

```python
def compute_feature_metrics(params_physical, ground_truth_params, ...) -> dict
def print_feature_metrics(metrics) -> None
def plot_corner(params_physical, param_names, ground_truth, ...) -> Figure
def plot_elbo_comparison(elbos, model_names, ...) -> Figure
def plot_posterior_images(images, n_show, ...) -> Figure
def plot_loss_curves(loss_history, ...) -> Figure
```
