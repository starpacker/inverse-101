# Code Design

## File Structure

```
main.py                  # Pipeline orchestration
src/
  preprocessing.py       # Data loading, closure quantity computation
  physics_model.py       # Closure forward model with gradients
  solvers.py             # Closure-only and visibility RML solvers
  visualization.py       # Plotting utilities and metrics
  generate_data.py       # Synthetic data with gain corruption
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str = "data") -> dict:
    """Load raw_data.npz. Returns dict with vis_corrupted, vis_true, uv_coords, station_ids, noise_std_per_vis."""

def load_metadata(data_dir: str = "data") -> dict:
    """Load meta_data JSON. Returns dict with N, pixel sizes, gain errors, etc."""

def find_triangles(station_ids, n_stations) -> (ndarray, ndarray):
    """Find closure phase triangles. Returns (triangles, tri_stations)."""

def find_quadrangles(station_ids, n_stations) -> (ndarray, ndarray):
    """Find closure amplitude quadrangles. Returns (quadrangles, quad_stations)."""

def compute_closure_phases(vis, triangles, station_ids) -> ndarray:
    """Visibilities → closure phases (radians) on triangles."""

def compute_log_closure_amplitudes(vis, quadrangles) -> ndarray:
    """Visibilities → log closure amplitudes on quadrangles."""

def closure_phase_sigma(vis, noise_std_per_vis, triangles) -> ndarray:
    """Closure phase noise σ_ψ (Eq. 11, Chael 2018)."""

def closure_amplitude_sigma(vis, noise_std_per_vis, quadrangles) -> ndarray:
    """Log closure amplitude noise σ_logCA (Eq. 12, Chael 2018)."""

def prepare_data(data_dir) -> tuple:
    """Combined loader: returns (obs, closure_data, metadata)."""
```

### physics_model.py

```python
class ClosureForwardModel:
    def __init__(self, uv_coords, image_size, pixel_size_rad, station_ids, triangles, quadrangles):
        """Build DFT matrix A and pre-extract rows for triangle/quadrangle baselines."""

    def forward(self, image) -> ndarray: """image → visibilities"""
    def adjoint(self, vis) -> ndarray:   """visibilities → back-projected image"""
    def dirty_image(self, vis) -> ndarray: """Normalized dirty image"""
    def psf(self) -> ndarray:            """Point spread function"""

    def model_closure_phases(self, image) -> ndarray:
        """image → model closure phases"""
    def closure_phase_chisq(self, image, cphases_obs, sigma_cp) -> float:
        """Closure phase χ² (Eq. 11)"""
    def closure_phase_chisq_grad(self, image, cphases_obs, sigma_cp) -> ndarray:
        """Gradient of closure phase χ² w.r.t. image"""

    def model_log_closure_amplitudes(self, image) -> ndarray:
    def log_closure_amp_chisq(self, image, log_camps_obs, sigma_logca) -> float:
    def log_closure_amp_chisq_grad(self, image, log_camps_obs, sigma_logca) -> ndarray:

    def visibility_chisq(self, image, vis_obs, noise_std) -> float:
    def visibility_chisq_grad(self, image, vis_obs, noise_std) -> ndarray:
```

### solvers.py

```python
class ClosurePhaseOnlySolver:
    def reconstruct(self, model, closure_data, x0=None) -> ndarray:
        """Closure phase only RML. Returns (N,N) image."""

class ClosurePhasePlusAmpSolver:
    def reconstruct(self, model, closure_data, x0=None) -> ndarray:
        """Closure phase + log closure amplitude RML."""

class VisibilityRMLSolver:
    def reconstruct(self, model, vis, noise_std, x0=None) -> ndarray:
        """Traditional visibility RML (comparison baseline)."""

class TVRegularizer:
    def value_and_grad(self, x) -> tuple: """(float, ndarray)"""

class MaxEntropyRegularizer:
    def value_and_grad(self, x) -> tuple: """(float, ndarray)"""

class L1SparsityRegularizer:
    def value_and_grad(self, x) -> tuple: """(float, ndarray)"""
```

### visualization.py

```python
def plot_uv_coverage(uv_coords, title="", ax=None) -> Axes
def plot_image(image, title="", ax=None, pixel_size_uas=None) -> Axes
def plot_closure_phases(cphases, ...) -> Figure
def plot_closure_amplitudes(log_camps, ...) -> Figure
def plot_comparison(reconstructions, ground_truth=None, ...) -> Figure
def plot_gain_robustness(results_by_gain, ground_truth, ...) -> Figure
def compute_metrics(estimate, ground_truth) -> dict
def print_metrics_table(metrics) -> None
```

## Data Flow

```
data/raw_data.npz + data/meta_data
        │
        ▼
  preprocessing.py  ──→  obs dict + closure_data dict + metadata
        │
        ▼
  physics_model.py  ──→  ClosureForwardModel (A matrix + closure operators)
        │
        ▼
  solvers.py        ──→  reconstructed images (N,N) per method
        │                  (closure-only vs visibility comparison)
        ▼
  visualization.py  ──→  comparison plots + metrics table
        │
        ▼
  output/reconstruction.npy
```
