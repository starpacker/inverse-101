"""
Solvers — EMS Deconvolution with Depth-Adaptive Anti-Aliasing
=============================================================

Implements the Estimate-Maximize-Smooth (EMS) algorithm from
Stefanoiu et al., Optics Express 27(22):31644, 2019 (Eq. 27).

Standard Richardson-Lucy deconvolution produces grid-pattern aliasing
artifacts near the native object plane (Δz ≈ 0) due to depth-dependent
under-sampling. EMS adds a depth-adaptive Lanczos smoothing step per
iteration to suppress these artifacts.

Adapted from pyolaf/aliasing.py and pyolaf/examples/deconvolve_image.py
(github.com/lambdaloop/pyolaf, Lili Karashchuk)
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from tqdm import trange

from .physics_model import forward_project, backward_project


# ═══════════════════════════════════════════════════════════════════════════════
# Depth-Adaptive Anti-Aliasing Filter Widths
# ═══════════════════════════════════════════════════════════════════════════════

def compute_depth_adaptive_widths(Camera: dict, Resolution: dict) -> np.ndarray:
    """
    Compute the depth-dependent anti-aliasing filter width for each depth plane.

    Implements Eqs. (5–7) from Stefanoiu et al. 2019 via geometric ray tracing:
      - Trace a marginal ray from object through objective → tube lens → MLA → sensor
      - Compute the micro-lens blur radius b_z and the MLA-to-sensor magnification λ_z
      - Filter width w_sens_z = min(|λ_z·p_ml - b_z|, r_ml)  (at sensor resolution)
      - Backproject to object (voxel) space: w_obj_z = w_sens_z · s / p_ml

    Port of LFM_computeDepthAdaptiveWidth from pyolaf/aliasing.py.

    Parameters
    ----------
    Camera : dict
        Camera parameters from preprocessing.set_camera_params.
        Required keys: fobj, dof, objRad, Delta_ot, ftl, tube2mla, fm, mla2sensor,
                       lensPitch.
    Resolution : dict
        Must contain 'depths' (array of Δz values in μm) and 'TexNnum' ([texH, texW]).

    Returns
    -------
    np.ndarray
        Integer filter widths, shape (nDepths, 2): column 0 = Y width, column 1 = X width.
        Values are odd integers (in voxels) for symmetric filter support.
    """
    dz = Resolution["depths"]
    zobj = (Camera["fobj"] - dz).astype("float")

    # Avoid division by zero at focal planes
    for i in range(len(zobj)):
        if np.isclose(zobj[i], Camera["fobj"]) or np.isclose(zobj[i], Camera["dof"]):
            zobj[i] = zobj[i] + 0.00001 * Camera["fobj"]

    # Objective image distance z1
    z1 = (zobj * Camera["fobj"]) / (zobj - Camera["fobj"])

    # Effective tube-lens radius (ray clipping at tube lens aperture)
    tube_rad = Camera["objRad"] * Camera["Delta_ot"] * np.abs(1.0 / z1 - 1.0 / Camera["Delta_ot"])

    # Tube-lens image distance z2
    z2 = Camera["ftl"] * (Camera["Delta_ot"] - z1) / (Camera["Delta_ot"] - z1 - Camera["ftl"])

    # Main blur at MLA plane B_z
    B = tube_rad * Camera["tube2mla"] * np.abs(1.0 / z2 - 1.0 / Camera["tube2mla"])

    # MLA image distance z3
    z3 = Camera["fm"] * (Camera["tube2mla"] - z2) / (Camera["tube2mla"] - z2 - Camera["fm"])

    # Micro-lens blur radius b_z
    b = Camera["lensPitch"] / 2 * np.abs(1.0 / z3 - 1.0 / Camera["mla2sensor"])

    # MLA-to-sensor magnification λ_z
    lam = z2 * Camera["mla2sensor"] / (Camera["tube2mla"] * np.abs(Camera["tube2mla"] - z2))

    # Filter width at sensor (in physical units)
    d = Camera["lensPitch"]
    pinhole_filt_rad = d * np.abs(lam)
    final_rad = np.abs(pinhole_filt_rad - b)

    # Clip to half-lenslet pitch
    w_sens = np.minimum(d / 2, final_rad)

    # Convert to object (voxel) space
    widths_x = w_sens * Resolution["TexNnum"][1] / d
    widths_y = w_sens * Resolution["TexNnum"][0] / d

    # Round to odd integers
    widths = np.zeros((len(w_sens), 2), dtype="int64")
    widths[:, 0] = np.floor(widths_y * 2)
    widths[:, 1] = np.floor(widths_x * 2)
    widths[widths % 2 == 0] += 1

    return widths


# ═══════════════════════════════════════════════════════════════════════════════
# Lanczos Anti-Aliasing Filter (FFT domain)
# ═══════════════════════════════════════════════════════════════════════════════

def build_lanczos_filters(volume_size: np.ndarray,
                           widths: np.ndarray, n: int = 4) -> np.ndarray:
    """
    Pre-compute Lanczos-n windowed sinc filters in the FFT domain for each depth.

    The Lanczos-n filter is: h(x,y) = sinc(x/w_x) · sinc(y/w_y)
                                       · sinc(x/(n·w_x)) · sinc(y/(n·w_y))
    truncated to zero outside a circle of radius 3·w_x voxels.

    Stored in FFT domain for efficient application via element-wise multiply.
    Port of lanczosfft from pyolaf/aliasing.py (numpy path, no CuPy).

    Parameters
    ----------
    volume_size : np.ndarray
        [texH, texW, nDepths].
    widths : np.ndarray
        Filter widths per depth, shape (nDepths, 2), from compute_depth_adaptive_widths.
    n : int
        Lanczos window order (default 4).

    Returns
    -------
    np.ndarray
        Complex array shape (texH, texW, nDepths): FFT of normalized Lanczos kernel.
    """
    nd = volume_size[2]
    size2 = np.int64(np.floor(np.array(volume_size) / 2))

    x, y = np.meshgrid(np.arange(-size2[1], size2[1] + 1),
                        np.arange(-size2[0], size2[0] + 1))
    dxy = np.sqrt(x**2 + y**2)

    kernel_fft = np.zeros(volume_size, dtype="complex128")

    for i in range(nd):
        w_y = float(widths[i, 0])
        w_x = float(widths[i, 1])

        fy = 1.0 / w_y if w_y > 0 else 0.0
        fx = 1.0 / w_x if w_x > 0 else 0.0

        x_f = x * fx
        y_f = y * fy

        kernel_sinc = np.sinc(x_f) * np.sinc(y_f)

        # Lanczos window: sinc(x/n) in each axis
        window = np.sinc(x_f / n) * np.sinc(y_f / n)

        # Zero outside radius 3·w_x (use x-direction width for isotropic cutoff)
        window[dxy > 3 * widths[i, 1]] = 0.0

        kernel = kernel_sinc * window
        total = kernel.sum()
        if total != 0:
            kernel /= total

        kernel_fft[:, :, i] = fft2(kernel)

    return kernel_fft


# ═══════════════════════════════════════════════════════════════════════════════
# EMS Deconvolution
# ═══════════════════════════════════════════════════════════════════════════════

def ems_deconvolve(H: np.ndarray, Ht: np.ndarray,
                   lf_image: np.ndarray,
                   lenslet_centers: dict, Resolution: dict, Camera: dict,
                   n_iter: int = 8,
                   filter_flag: bool = True,
                   lanczos_n: int = 4) -> np.ndarray:
    """
    Estimate-Maximize-Smooth (EMS) deconvolution (paper Eq. 27).

    Iterative update rule:

        v^{q+1} = h_{fw,z} * [ v^q / (A^T·1) · A^T( m / (A·v^q) · (A·1) ) ]

    where h_{fw,z} is the depth-adaptive Lanczos-n filter applied per depth
    slice (skipped when filter_flag=False → standard Richardson-Lucy).

    Port of the deconvolution loop from pyolaf/examples/deconvolve_image.py.

    Parameters
    ----------
    H : np.ndarray
        Forward operator, object array of csr_matrix, shape (TexNnum_half[0], TexNnum_half[1], nDepths).
    Ht : np.ndarray
        Backward operator, same structure (Nnum_half[0], Nnum_half[1], nDepths).
    lf_image : np.ndarray
        Observed 2D sensor image, float32, shape (imgH, imgW).
    lenslet_centers : dict
        Lenslet center arrays, keys 'px' and 'vox'.
    Resolution : dict
        Resolution dict from preprocessing.compute_geometry.
    Camera : dict
        Camera parameter dict.
    n_iter : int
        Number of EM iterations (default 8).
    filter_flag : bool
        True  → EMS: apply depth-adaptive Lanczos smoothing (artifact-free).
        False → standard RL: no smoothing (aliasing artifacts near Δz≈0).
    lanczos_n : int
        Lanczos window order (default 4).

    Returns
    -------
    np.ndarray
        Reconstructed 3D volume, shape (texH, texW, nDepths), float32.
    """
    img_size = np.array(lf_image.shape)

    # Volume size
    nDepths = len(Resolution["depths"])
    tex_size = np.ceil(img_size * np.array(Resolution["texScaleFactor"])).astype("int32")
    tex_size = tex_size + (1 - tex_size % 2)
    volume_size = np.append(tex_size, nDepths).astype("int32")

    # Pre-build depth-adaptive filters (if EMS mode)
    if filter_flag:
        print("  Building depth-adaptive Lanczos filters...")
        widths = compute_depth_adaptive_widths(Camera, Resolution)
        kernel_fft = build_lanczos_filters(volume_size, widths, n=lanczos_n)

    # Uniform initial estimate
    init_volume = np.ones(volume_size, dtype="float32")

    # Precompute normalization denominators: A·1 and A^T(A·1)
    print("  Precomputing normalization (A·1 and A^T(A·1))...")
    ones_fwd = forward_project(H, init_volume, lenslet_centers, Resolution, img_size, Camera)
    ones_back = backward_project(Ht, ones_fwd, lenslet_centers, Resolution, tex_size, Camera)

    # Initialize reconstruction
    recon = np.ones(volume_size, dtype="float32")

    print(f"  Running {'EMS' if filter_flag else 'Richardson-Lucy'} deconvolution "
          f"({n_iter} iterations)...")

    for i in range(n_iter):
        print(f"  Iteration {i + 1}/{n_iter}")

        # E-step: forward project current estimate
        if i == 0:
            lf_guess = ones_fwd
        else:
            lf_guess = forward_project(H, recon, lenslet_centers, Resolution, img_size, Camera)

        # Compute error ratio: m / (A·v^q) · (A·1)
        error_lf = np.zeros_like(lf_guess)
        nonzero = lf_guess != 0
        error_lf[nonzero] = lf_image[nonzero] / lf_guess[nonzero] * ones_fwd[nonzero]
        error_lf[~np.isfinite(error_lf)] = 0.0

        # M-step: backward project error
        error_back = backward_project(Ht, error_lf, lenslet_centers, Resolution, tex_size, Camera)

        # Normalize
        nonzero_b = ones_back != 0
        error_back[nonzero_b] /= ones_back[nonzero_b]
        error_back[~np.isfinite(error_back)] = 0.0

        # Multiplicative update
        recon = recon * error_back

        # S-step: depth-adaptive smoothing (EMS only)
        if filter_flag:
            for j in range(nDepths):
                slice_j = recon[:, :, j]
                filtered = np.abs(fftshift(ifft2(kernel_fft[:, :, j] * fft2(slice_j))))
                recon[:, :, j] = filtered.astype("float32")

        recon[~np.isfinite(recon)] = 0.0

    return recon
