# Design: Light Field Microscope — Code Architecture

## File Structure

```
tasks/light_field_microscope/
├── README.md
├── requirements.txt
├── main.py                            # Pipeline orchestration
├── plan/
│   ├── approach.md
│   └── design.md
├── data/
│   ├── meta_data                      # JSON (no extension)
│   └── raw_data.npz                   # lf_image, ground_truth
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Camera params, geometry, calibration
│   ├── physics_model.py               # Wave-optics PSF, H/Ht operators, projection
│   ├── solvers.py                     # Anti-aliasing filters, EMS deconvolution
│   ├── visualization.py              # Volume rendering, metrics, comparison plots
│   └── generate_data.py              # Synthetic bead volume + LFM forward simulation
├── evaluation/
│   └── reference_outputs/
│       ├── ground_truth.npy
│       ├── lf_image.npy
│       ├── reconstruction_rl.npy
│       ├── reconstruction_ems.npy
│       ├── operators_H.pkl
│       ├── operators_Ht.pkl
│       └── metrics.json
└── notebooks/
    └── light_field_microscope.ipynb
```

---

## Data Flow

```
meta_data (JSON)
    │
    ├─ load_metadata() ──────────────────────────── preprocessing.py
    │
    ├─ set_camera_params() ──────────────────────── preprocessing.py
    │   Returns: Camera dict
    │
    ├─ compute_geometry() ───────────────────────── preprocessing.py
    │   Returns: (LensletCenters, Resolution, LensletGridModel, NewGridModel)
    │
    ├─ compute_lf_operators() ───────────────────── physics_model.py  [SLOW]
    │   Returns: (H, Ht)  — object arrays of csr_matrix
    │
    ├─ generate_bead_volume() ───────────────────── generate_data.py
    │   Returns: gt_volume (texH, texW, nDepths)
    │
    ├─ forward_project() ────────────────────────── physics_model.py
    │   Returns: lf_image_clean (imgH, imgW)
    │
    ├─ add_poisson_noise() ──────────────────────── generate_data.py
    │   Returns: lf_image (imgH, imgW)
    │
    └─ Saved: data/raw_data.npz (lf_image, ground_truth)
                │
                ├─ ems_deconvolve(..., filter_flag=False) ── solvers.py
                │   Returns: reconstruction_rl (texH, texW, nDepths)
                │
                └─ ems_deconvolve(..., filter_flag=True)  ── solvers.py
                    Returns: reconstruction_ems (texH, texW, nDepths)
```

---

## Module: preprocessing.py  (EXISTS — no changes needed)

```python
def load_metadata(path: str = "data/meta_data") -> dict
def set_camera_params(metadata: dict, new_spacing_px: int) -> dict
def build_grid(grid_model: dict, grid_type: str) -> np.ndarray
def set_grid_model(spacing_px, first_pos_shift_row, u_max, v_max,
                   h_offset, v_offset, rot, orientation, grid_type) -> dict
def process_white_image(white_image: np.ndarray, spacing_px: float,
                        grid_type: str) -> tuple
def fix_mask(mask: np.ndarray, new_lenslet_spacing: np.ndarray,
             grid_type: str) -> np.ndarray
def compute_patch_mask(n_spacing, grid_type, pixel_size, patch_rad, nnum) -> np.ndarray
def compute_resolution(lenslet_grid_model, texture_grid_model, Camera,
                       depth_range, depth_step) -> dict
def compute_lens_centers(new_grid_model, texture_grid_model,
                         sensor_res, grid_type) -> dict
def compute_geometry(Camera, white_image, depth_range, depth_step,
                     super_res_factor, img_size=None) -> tuple
    # Returns (LensletCenters, Resolution, LensletGridModel, NewLensletGridModel)
def retrieve_transformation(lenslet_grid_model, new_grid_model) -> np.ndarray
def format_transform(fix_all) -> np.ndarray
def get_transformed_shape(img_shape, ttnew) -> np.ndarray
def transform_image(img, ttnew, lens_offset) -> np.ndarray
def prepare_data(metadata_path: str = "data/meta_data") -> tuple
    # Returns (Camera, metadata)
```

---

## Module: physics_model.py  (CREATE from pyolaf/lf.py + project.py)

```python
# ── PSF Computation ──────────────────────────────────────────────────────────

def compute_psf_size(max_depth: float, Camera: dict) -> float:
    """Geometric PSF radius at MLA in units of lenslet pitch.
    Port of LFM_computePSFsize."""

def get_used_lenslet_centers(psf_size: float, lenslet_centers: dict) -> dict:
    """Crop lenslet centers to PSF extent for efficiency.
    Port of LFM_getUsedCenters."""

def compute_psf_single_depth(p1: float, p2: float, p3: float,
                              Camera: dict, Resolution: dict) -> np.ndarray:
    """
    Debye integral PSF at native image plane for source at (p1, p2, p3).
    Returns complex array shape (2*half+1, 2*half+1).
    Port of LFM_calcPSF.
    """

def compute_psf_all_depths(Camera: dict, Resolution: dict) -> np.ndarray:
    """
    PSF wave stack for all depths. Exploits conjugate symmetry (PSF(-z) = conj(PSF(z))).
    Returns complex array shape (yspace, xspace, nDepths).
    Port of LFM_calcPSFAllDepths.
    """

# ── MLA Transmittance ────────────────────────────────────────────────────────

def compute_ulens_transmittance(Camera: dict, Resolution: dict) -> np.ndarray:
    """Single micro-lens complex phase transmittance. Port of LFM_ulensTransmittance."""

def compute_mla_transmittance(Camera: dict, Resolution: dict,
                               ulens_pattern: np.ndarray) -> np.ndarray:
    """
    Full MLA array transmittance by tiling ulens_pattern at lenslet centers.
    Port of LFM_mlaTransmittance.
    Returns complex array shape (imgH, imgW).
    """

# ── Sensor Propagation ───────────────────────────────────────────────────────

def propagate_to_sensor(field: np.ndarray, sensor_res: np.ndarray,
                         z: float, wavelength: float,
                         ideal_sampling: bool = False) -> np.ndarray:
    """
    Rayleigh-Sommerfeld angular spectrum propagation.
    Port of prop2Sensor. Returns complex sensor field.
    """

# ── Forward / Backward Operators ─────────────────────────────────────────────

def compute_forward_patterns(psf_wave_stack: np.ndarray, mlarray: np.ndarray,
                              Camera: dict, Resolution: dict) -> np.ndarray:
    """
    Forward operator H[aa, bb, depth] = csr_matrix of sensor pattern
    for texture coordinate (aa, bb) at given depth.
    Shape: (TexNnum_half[0], TexNnum_half[1], nDepths), dtype=object.
    Port of LFM_computeForwardPatternsWaves.
    """

def compute_backward_patterns(H: np.ndarray, Resolution: dict,
                               Camera: dict) -> np.ndarray:
    """
    Backward operator Ht[aa_sen, bb_sen, depth] = csr_matrix.
    Shape: (Nnum_half[0], Nnum_half[1], nDepths), dtype=object.
    Ports LFM_computeBackwardPatterns + normalizeHt.
    """

def compute_lf_operators(Camera: dict, Resolution: dict,
                          lenslet_centers: dict) -> tuple:
    """
    Top-level: compute (H, Ht) for the given LFM configuration.
    Port of LFM_computeLFMatrixOperators.
    Returns (H, Ht).
    """

# ── Projection ───────────────────────────────────────────────────────────────

def _fft_conv2d(a: np.ndarray, b: np.ndarray, mode: str = "same") -> np.ndarray:
    """2D FFT convolution (numpy, no CuPy). Port of cufftconv numpy path."""

def forward_project(H: np.ndarray, volume: np.ndarray,
                    lenslet_centers: dict, Resolution: dict,
                    img_size: np.ndarray, Camera: dict) -> np.ndarray:
    """
    3D volume → 2D light field image via H.
    Iterates over depths and texture coords, accumulates FFT convolutions.
    Returns float32 array shape img_size.
    Port of LFM_forwardProject (numpy path, no CuPy).
    """

def backward_project(Ht: np.ndarray, lf_image: np.ndarray,
                     lenslet_centers: dict, Resolution: dict,
                     tex_size: np.ndarray, Camera: dict) -> np.ndarray:
    """
    2D light field image → 3D volume via Ht.
    Returns float32 array shape (texH, texW, nDepths).
    Port of LFM_backwardProject (numpy path, no CuPy).
    """
```

---

## Module: solvers.py  (CREATE from pyolaf/aliasing.py + fftpack.py + examples/)

```python
def compute_depth_adaptive_widths(Camera: dict, Resolution: dict) -> np.ndarray:
    """
    Anti-aliasing filter widths per depth.
    Returns int array shape (nDepths, 2): [width_y_voxels, width_x_voxels].
    Port of LFM_computeDepthAdaptiveWidth.
    """

def build_lanczos_filters(volume_size: np.ndarray,
                           widths: np.ndarray, n: int = 4) -> np.ndarray:
    """
    Pre-compute Lanczos-n windowed sinc filters in FFT domain.
    Returns complex array shape (texH, texW, nDepths).
    Port of lanczosfft (numpy path, no CuPy).
    """

def ems_deconvolve(H: np.ndarray, Ht: np.ndarray,
                   lf_image: np.ndarray,
                   lenslet_centers: dict, Resolution: dict, Camera: dict,
                   n_iter: int = 8,
                   filter_flag: bool = True,
                   lanczos_n: int = 4) -> np.ndarray:
    """
    Estimate-Maximize-Smooth deconvolution (paper Eq. 27).

    Each iteration:
      1. Forward project current volume estimate: A·v^q
      2. Compute error ratio: m / (A·v^q) · (A·1)
      3. Backward project error: A^T(error)
      4. Multiply update: v^q * (A^T error) / (A^T 1)
      5. [EMS only] Apply depth-adaptive Lanczos filter per depth slice

    Parameters
    ----------
    filter_flag : bool
        True → EMS (artifact-free).
        False → standard Richardson-Lucy (with aliasing artifacts).

    Returns
    -------
    np.ndarray
        Reconstructed volume, shape (texH, texW, nDepths), float32.
    """
```

---

## Module: visualization.py  (CREATE)

```python
def plot_lf_image(lf_image: np.ndarray,
                  title: str = "Light Field Image") -> plt.Figure:
    """Display raw 2D sensor image. Grayscale with inferno colormap."""

def plot_volume_slices(volume: np.ndarray, depths: np.ndarray,
                       title: str = "", vmax: float = None) -> plt.Figure:
    """Grid of lateral (XY) slices at each depth plane."""

def plot_volume_mip(volume: np.ndarray, title: str = "") -> plt.Figure:
    """Max-intensity projections: XY (top), XZ (side), YZ (front)."""

def plot_reconstruction_comparison(gt: np.ndarray,
                                   rl_vol: np.ndarray,
                                   ems_vol: np.ndarray,
                                   depths: np.ndarray,
                                   metrics: dict) -> plt.Figure:
    """
    Side-by-side columns: Ground Truth | RL (artifacts) | EMS (clean).
    One row per depth plane.
    """

def compute_metrics(estimate: np.ndarray,
                    ground_truth: np.ndarray) -> dict:
    """
    Returns {'nrmse': float, 'psnr': float}.
    NRMSE = ||estimate - gt||_F / ||gt||_F
    PSNR = 20 * log10(max(gt) / RMSE)
    """

def print_metrics_table(metrics: dict) -> None:
    """Formatted table to stdout."""
```

---

## Module: generate_data.py  (CREATE)

```python
def generate_bead_volume(metadata: dict,
                          Resolution: dict) -> np.ndarray:
    """
    Create a 3D fluorescent bead volume of shape (texH, texW, nDepths).

    Beads are modeled as 3D Gaussians with sigma = bead_radius_um / texRes.
    Positions are drawn at fixed lateral random positions and assigned to
    depth planes (one bead per depth + extra random beads).
    Uses metadata['synthetic_data'] for n_beads, bead_radius_um, etc.
    """

def add_poisson_noise(image: np.ndarray, scale: float,
                      rng: np.random.Generator = None) -> np.ndarray:
    """
    Scale image by 'scale' and draw from Poisson distribution.
    Returns float array normalized to [0, 1].
    """

def generate_synthetic_dataset(metadata_path: str = "data/meta_data",
                                 output_data_path: str = "data/raw_data.npz",
                                 output_ref_dir: str = "evaluation/reference_outputs") -> None:
    """
    Full pipeline:
      1. Load metadata, build Camera, compute geometry
      2. Compute LF operators H, Ht (or load cached from output_ref_dir)
      3. Generate bead volume
      4. Forward project + add Poisson noise → lf_image
      5. Save data/raw_data.npz: {lf_image, ground_truth}
      6. Save operators_H.pkl, operators_Ht.pkl to output_ref_dir
    """

if __name__ == "__main__":
    generate_synthetic_dataset()
```

---

## Module: src/__init__.py  (UPDATE)

```python
from . import preprocessing
from . import physics_model
from . import solvers
from . import visualization
from . import generate_data
```

---

## main.py

```python
def main():
    # Step 1: Load data and camera params
    Camera, metadata = prepare_data("data/meta_data")
    data = np.load("data/raw_data.npz")
    lf_image = data["lf_image"]
    ground_truth = data["ground_truth"]

    # Step 2: Compute geometry (synthetic mode)
    img_size = np.array(lf_image.shape)
    LensletCenters, Resolution, _, _ = compute_geometry(
        Camera, np.array([]), depth_range, depth_step, super_res_factor, img_size)
    tex_size = np.array(Resolution['TexNnum'][:2] * Resolution['usedLensletCenters']['vox'].shape[:2])

    # Step 3: Load or compute H, Ht
    ops_path_H = "evaluation/reference_outputs/operators_H.pkl"
    if os.path.exists(ops_path_H):
        H, Ht = load_operators(ops_path_H)
    else:
        H, Ht = compute_lf_operators(Camera, Resolution, LensletCenters)

    # Step 4: Deconvolution — RL (no filter) + EMS (with filter)
    vol_rl  = ems_deconvolve(H, Ht, lf_image, ..., filter_flag=False)
    vol_ems = ems_deconvolve(H, Ht, lf_image, ..., filter_flag=True)

    # Step 5: Evaluate
    metrics = {
        "rl":  compute_metrics(vol_rl,  ground_truth),
        "ems": compute_metrics(vol_ems, ground_truth),
    }
    print_metrics_table(metrics)

    # Step 6: Save
    os.makedirs("output", exist_ok=True)
    np.save("output/reconstruction_ems.npy", vol_ems)
    np.save("output/reconstruction_rl.npy", vol_rl)
```

---

## Key Conventions

- **No CuPy**: all code uses numpy/scipy only (CuPy is optional in pyolaf but omitted here)
- **No `matplotlib.use('Agg')` in src/**: only in main.py for headless runs
- **Sparse matrices**: H, Ht entries are `scipy.sparse.csr_matrix`
- **Object arrays**: H has `dtype=object`, indexed as `H[aa, bb, depth]`
- **Coordinate convention**: row = y = vertical, col = x = horizontal throughout
- **Depths**: stored as Δz in μm relative to NOP; positive = above NOP
