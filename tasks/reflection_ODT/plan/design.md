# Design: Reflection-Mode ODT Code Architecture

## Module Structure

### src/preprocessing.py

```python
def generate_phantom(metadata: dict) -> np.ndarray
    """Generate 4-layer USAF-like synthetic phantom.
    Returns: (Nz, Ny, Nx) array with Δn = -0.07 for structures"""

def load_metadata(data_dir: str) -> dict
    """Load imaging parameters from data/meta_data JSON"""

def prepare_data(data_dir: str) -> tuple[np.ndarray, dict]
    """Load metadata and generate phantom. Returns (phantom, metadata)"""
```

### src/physics_model.py

```python
@dataclass
class ReflectionBPMConfig:
    volume_shape: tuple  # (Nz, Ny, Nx)
    res: tuple           # Normalized (dx·n0/λ, dy·n0/λ, dz·n0/λ)
    n0: float
    NA_obj: float
    NA_illu: float
    wavelength_um: float
    res_um: tuple
    n_angles: int
    dz_layer: float      # BPM step for layer scattering
    dz_gap: float        # BPM step for gaps

    @classmethod
    def from_metadata(cls, metadata: dict) -> ReflectionBPMConfig

class ReflectionBPMForwardModel:
    def __init__(self, config: ReflectionBPMConfig, device: str)

    def _compute_c_gamma(self) -> torch.Tensor
        """cos(gamma) = sqrt(1 - fx² - fy²)"""

    def _compute_kz(self) -> torch.Tensor
        """kz = c_gamma · 2π·res_z"""

    def _compute_evanescent_mask(self) -> torch.Tensor
    def _compute_pupil(self) -> torch.Tensor

    def _make_incident_field(self, angle_idx: int) -> torch.Tensor
        """Tilted plane wave for illumination angle m"""

    def _bpm_propagate(self, u: torch.Tensor, dz: float) -> torch.Tensor
        """P operator: fft → exp(i·kz·dz)·eva → ifft"""

    def _bpm_scatter(self, u: torch.Tensor, dn_slice: torch.Tensor, dz: float) -> torch.Tensor
        """Q operator: u *= exp(i·Δn·2π·res_z/n0·dz)"""

    def _reflect(self, u: torch.Tensor) -> torch.Tensor
        """Mirror reflection: u *= -1"""

    def forward_single(self, dn_volume: torch.Tensor, angle_idx: int) -> torch.Tensor
        """Full reflection pipeline for one angle. Returns intensity (Ny, Nx)"""

    def forward(self, dn_volume: torch.Tensor) -> torch.Tensor
        """All angles. Returns (n_angles, Ny, Nx)"""
```

### src/solvers.py

```python
def tv_2d_proximal(volume: torch.Tensor, tau: float, n_iter: int = 20) -> torch.Tensor
    """Chambolle dual projection per-slice 2D TV denoising"""

class ReflectionBPMReconstructor:
    def __init__(self, n_iter: int, lr: float, tv_weight: float,
                 positivity: bool, device: str)

    def reconstruct(self, measurements: torch.Tensor, model) -> tuple[np.ndarray, list]
        """FISTA reconstruction. Returns (dn_recon, loss_history)"""
```

### src/visualization.py

```python
def plot_ri_slices(volume: np.ndarray, ...) -> Figure
def plot_comparison(ground_truth: np.ndarray, reconstruction: np.ndarray, ...) -> Figure
def plot_loss_history(losses: list, ...) -> Figure
def plot_measurements(measurements: np.ndarray, ...) -> Figure
def compute_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> dict
def print_metrics_table(metrics: dict) -> None
```

### src/generate_data.py

```python
def generate_measurements(data_dir: str, output_path: str, device: str) -> None
    """Load phantom, run forward model, save measurements"""
```

## Pipeline (main.py)

1. Load metadata and generate phantom
2. Build ReflectionBPMForwardModel
3. Simulate measurements (16 angles)
4. Reconstruct via FISTA (50 iter, lr=5.0, tv=8e-7)
5. Evaluate metrics (NRMSE, NCC, SSIM)
6. Save outputs and visualizations
