# Code Design

## File Structure

```
main.py                  # Pipeline orchestration
src/
  preprocessing.py       # Data loading and preparation
  physics_model.py       # VLBI forward model (measurement operator)
  solvers.py             # Inverse problem solvers and regularizers
  visualization.py       # Plotting utilities and metrics
  generate_data.py       # Synthetic data generation (optional)
```

## Function Signatures

### preprocessing.py

```python
def load_observation(data_dir: str = "data") -> dict:
    """
    Load raw_data.npz from data directory.

    Returns: dict with keys
      'vis_noisy'  : np.ndarray, shape (M,), complex128 — noisy visibilities
      'uv_coords'  : np.ndarray, shape (M, 2), float64  — baseline positions [wavelengths]
    """

def load_metadata(data_dir: str = "data") -> dict:
    """
    Load meta_data JSON from data directory.

    Returns: dict with keys
      'N'              : int   — image size
      'pixel_size_uas' : float — pixel size in microarcseconds
      'pixel_size_rad' : float — pixel size in radians
      'noise_std'      : float — noise standard deviation σ
      'freq_ghz'       : float — observing frequency
      'n_baselines'    : int   — number of baselines M
    """

def prepare_data(data_dir: str = "data") -> tuple:
    """
    Combined loader: returns (vis_noisy, uv_coords, metadata).
    """
```

### physics_model.py

```python
class VLBIForwardModel:
    """
    Linear measurement operator for VLBI imaging.

    Attributes:
      A  : np.ndarray, shape (M, N²), complex128 — measurement matrix
      uv : np.ndarray, shape (M, 2)  — baseline coordinates
      N  : int                        — image side length
      M  : int                        — number of baselines
    """

    def __init__(self, uv_coords: np.ndarray, image_size: int, pixel_size_rad: float):
        """Build measurement matrix A from (u,v) positions and pixel grid."""

    def forward(self, image: np.ndarray) -> np.ndarray:
        """image (N,N) → visibilities (M,) complex. Computes y = A @ x."""

    def adjoint(self, vis: np.ndarray) -> np.ndarray:
        """visibilities (M,) complex → image (N,N) real. Computes A^H @ y."""

    def dirty_image(self, vis: np.ndarray) -> np.ndarray:
        """Normalized back-projection: A^H y / max(PSF)."""

    def psf(self) -> np.ndarray:
        """Point Spread Function (dirty beam): A^H 1 / max(A^H 1)."""

    def add_noise(self, vis: np.ndarray, snr: float = 20.0, rng=None) -> tuple:
        """Add thermal noise. Returns (vis_noisy, noise_std)."""
```

### solvers.py

```python
class DirtyImageReconstructor:
    def reconstruct(self, model, vis: np.ndarray, noise_std: float) -> np.ndarray:
        """Baseline back-projection. Returns (N,N) dirty image."""

class CLEANReconstructor:
    def __init__(self, gain=0.05, n_iter=50, threshold=1e-4,
                 clean_beam_fwhm=None, support_radius=None):
        """
        Högbom CLEAN. support_radius is critical for sparse arrays like EHT.
        """

    def reconstruct(self, model, vis: np.ndarray, noise_std: float) -> np.ndarray:
        """Iterative deconvolution. Returns (N,N) CLEAN image."""

class RMLSolver:
    def __init__(self, regularizers=None, n_iter=500, positivity=True):
        """
        Regularized Maximum Likelihood via L-BFGS-B.
        regularizers: list of (weight, regularizer) tuples.
        """

    def reconstruct(self, model, vis: np.ndarray, noise_std: float,
                    x0: np.ndarray = None) -> np.ndarray:
        """Optimization-based reconstruction. Returns (N,N) image."""

class TVRegularizer:
    def __init__(self, epsilon: float = 1e-6): ...
    def value_and_grad(self, x: np.ndarray) -> tuple:
        """Returns (float, ndarray) — value and gradient of TV(x)."""

class MaxEntropyRegularizer:
    def __init__(self, prior: np.ndarray = None, epsilon: float = 1e-12): ...
    def value_and_grad(self, x: np.ndarray) -> tuple:
        """Returns (float, ndarray) — value and gradient of -H(x)."""

class L1SparsityRegularizer:
    def __init__(self, epsilon: float = 1e-8): ...
    def value_and_grad(self, x: np.ndarray) -> tuple:
        """Returns (float, ndarray) — value and gradient of ‖x‖₁."""
```

### visualization.py

```python
def plot_uv_coverage(uv_coords: np.ndarray, title="EHT uv-Coverage", ax=None) -> Axes:
    """Plot (u,v)-plane sampling pattern with conjugate points."""

def plot_image(image: np.ndarray, title="", ax=None, pixel_size_uas=None) -> Axes:
    """Display 2D image with optional physical axis labels (μas)."""

def plot_visibilities(vis: np.ndarray, uv_coords: np.ndarray) -> Figure:
    """Amplitude and phase vs. baseline length (two-panel plot)."""

def plot_comparison(reconstructions: dict, ground_truth=None,
                    pixel_size_uas=None, metrics=None) -> Figure:
    """Side-by-side comparison of multiple reconstruction methods."""

def plot_summary_panel(model, vis_noisy, reconstructions, ground_truth=None,
                       pixel_size_uas=None, metrics=None) -> Figure:
    """Full summary: uv-coverage + dirty image + methods."""

def compute_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute image-quality metrics after flux normalization.
    Returns dict with 'nrmse', 'ncc', 'dynamic_range'.
    """

def print_metrics_table(metrics: dict) -> None:
    """Print formatted comparison table."""
```

### main.py

```python
def main():
    """
    Orchestrate the full reconstruction pipeline:
      1. prepare_data("data")        → vis_noisy, uv_coords, metadata
      2. VLBIForwardModel(...)        → model
      3. Reconstruct with 4 methods   → dict of images
      4. compute_metrics(...)         → metrics dict
      5. plot_comparison(...)         → visualization
      6. Save output/reconstruction.npy
    """
```

## Data Flow

```
data/raw_data.npz + data/meta_data
        │
        ▼
  preprocessing.py  ──→  vis_noisy (M,), uv_coords (M,2), metadata dict
        │
        ▼
  physics_model.py  ──→  VLBIForwardModel (contains A matrix)
        │
        ▼
  solvers.py        ──→  reconstructed images (N,N) per method
        │
        ▼
  visualization.py  ──→  comparison plots + metrics table
        │
        ▼
  output/reconstruction.npy
```
