"""
Physics Model — Wave-Optics LFM Forward/Backward Operators
===========================================================

Implements the generalized light field point spread function (LFPSF) from
Stefanoiu et al., Optics Express 27(22):31644, 2019.

Pipeline per depth Δz:
  1. Debye integral  → wavefront U_{mla-} at the MLA plane
  2. MLA transmittance T → U_{mla+} = U_{mla-} · T
  3. Rayleigh-Sommerfeld propagation → sensor field U_{sens}
  4. |U_{sens}|² → detection probability a_{ji}

The forward operator H and backward operator Ht are stored as object arrays of
scipy.sparse.csr_matrix, one per (texture_coord, depth) combination.

Adapted from pyolaf/lf.py and pyolaf/project.py
(github.com/lambdaloop/pyolaf, Lili Karashchuk)
"""

import time
import numpy as np
from scipy import integrate, special
from scipy.fft import fft2, ifft2
from scipy.ndimage import shift
from scipy.signal import convolve2d
from scipy.sparse import csr_matrix, coo_matrix
from tqdm import trange


# ═══════════════════════════════════════════════════════════════════════════════
# PSF Size Estimation (geometric optics)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_psf_size(max_depth: float, Camera: dict) -> float:
    """
    Estimate the PSF radius at the MLA for the given maximum axial depth.

    Uses geometric ray tracing through objective → tube lens → MLA.

    Parameters
    ----------
    max_depth : float
        Maximum axial depth (um) relative to NOP (before offsetFobj correction).
    Camera : dict
        Camera parameter dictionary from preprocessing.set_camera_params.

    Returns
    -------
    float
        PSF radius in units of lenslet pitch (rounded up, +2 margin).
    """
    max_depth = max_depth - Camera["offsetFobj"]
    zobj = Camera["fobj"] - max_depth
    if zobj == Camera["fobj"] or zobj == Camera["dof"]:
        zobj = zobj + 0.00001 * Camera["fobj"]

    z1 = (zobj * Camera["fobj"]) / (zobj - Camera["fobj"])
    tube_rad = Camera["objRad"] * Camera["Delta_ot"] * np.abs(1.0 / z1 - 1.0 / Camera["Delta_ot"])
    z2 = Camera["ftl"] * (Camera["Delta_ot"] - z1) / (Camera["Delta_ot"] - z1 - Camera["ftl"])
    blur_rad = tube_rad * Camera["tube2mla"] * np.abs(1.0 / z2 - 1.0 / Camera["tube2mla"])

    psf_size = np.ceil(blur_rad / Camera["lensPitch"]) + 2
    print(f"  PSF radius ≈ {psf_size:.0f} lenslet pitches")
    return psf_size


def get_used_lenslet_centers(psf_size: float, lenslet_centers: dict) -> dict:
    """
    Extract the subset of lenslet centers within the PSF footprint.

    Parameters
    ----------
    psf_size : float
        PSF radius in lenslet-pitch units (from compute_psf_size).
    lenslet_centers : dict
        Full lenslet center arrays ('px' and 'vox') from preprocessing.

    Returns
    -------
    dict
        {'px': ndarray, 'vox': ndarray} — cropped to PSFsize+3 neighborhood.
    """
    used_lens = np.array([psf_size + 3, psf_size + 3])
    center_of_matrix = np.round(0.01 + np.array(lenslet_centers["px"].shape[:2]) / 2).astype(int)

    idx_y = np.arange(center_of_matrix[0] - used_lens[0] - 1,
                      center_of_matrix[0] + used_lens[0]).astype(int)
    idx_x = np.arange(center_of_matrix[1] - used_lens[1] - 1,
                      center_of_matrix[1] + used_lens[1]).astype(int)

    idx_y = idx_y[(idx_y >= 0) & (idx_y < lenslet_centers["px"].shape[0])]
    idx_x = idx_x[(idx_x >= 0) & (idx_x < lenslet_centers["px"].shape[1])]

    return {
        "px":  lenslet_centers["px"][idx_y[:, None], idx_x, :],
        "vox": lenslet_centers["vox"][idx_y[:, None], idx_x, :],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Debye Integral PSF
# ═══════════════════════════════════════════════════════════════════════════════

def compute_psf_single_depth(p1: float, p2: float, p3: float,
                              Camera: dict, Resolution: dict) -> np.ndarray:
    """
    Compute the PSF at the native image plane using the Debye integral.

    Implements Eq. (9) from Stefanoiu et al. 2019 for a source point (p1, p2, p3).
    Exploits four-fold symmetry: only one quadrant is integrated, the rest
    are obtained by rotation.

    Parameters
    ----------
    p1, p2 : float
        Source transverse coordinates (um). Typically 0 for on-axis PSF.
    p3 : float
        Source axial coordinate (um), offset from NOP.
    Camera : dict
        Camera parameters.
    Resolution : dict
        Resolution parameters (yspace, xspace must already be set).

    Returns
    -------
    np.ndarray
        Complex PSF, shape (len(yspace), len(xspace)).
    """
    k = 2 * np.pi * Camera["n"] / Camera["WaveLength"]
    alpha = np.arcsin(Camera["NA"] / Camera["n"])
    demag = 1.0 / Camera["M"]

    ylength = len(Resolution["yspace"])
    xlength = len(Resolution["xspace"])
    center_pt = np.ceil(ylength / 2).astype(int)

    yspace = Resolution["yspace"][:center_pt]
    xspace = Resolution["xspace"][:center_pt]

    d1 = Camera["dof"] - p3
    u = 4 * k * p3 * (np.sin(alpha / 2) ** 2)
    Koi = (demag / ((d1 * Camera["WaveLength"]) ** 2)
           * np.exp(-1j * u / (4 * np.sin(alpha / 2) ** 2)))

    x, y = np.meshgrid(xspace, yspace)
    r = np.sqrt((y + Camera["M"] * p1) ** 2 + (x + Camera["M"] * p2) ** 2) / Camera["M"]
    v = k * r * np.sin(alpha)

    def integrand(theta, alpha, u, v):
        return (np.sqrt(np.cos(theta)) * (1 + np.cos(theta))
                * np.exp(1j * u / 2 * np.sin(theta / 2) ** 2 / np.sin(alpha / 2) ** 2)
                * special.j0(np.sin(theta) / np.sin(alpha) * v)
                * np.sin(theta))

    I0, _ = integrate.quad_vec(integrand, 0, alpha, args=(alpha, u, v), limit=100)

    # Fill upper-left quadrant (exploiting diagonal symmetry)
    pattern = np.zeros((center_pt, center_pt), dtype="complex128")
    for a in range(center_pt):
        pattern[a, a:] = Koi * I0[a, a:]

    # Reconstruct full PSF by 4× rotation
    patA = pattern
    patAt = np.fliplr(patA)
    p3d = np.zeros((xlength, ylength, 4), dtype="complex128")
    p3d[:center_pt, :center_pt, 0] = pattern
    p3d[:center_pt, center_pt - 1:, 0] = patAt
    p3d[:, :, 1] = np.rot90(p3d[:, :, 0], -1)
    p3d[:, :, 2] = np.rot90(p3d[:, :, 0], -2)
    p3d[:, :, 3] = np.rot90(p3d[:, :, 0], -3)

    # Sum, then fix diagonal overlaps by picking max-magnitude quadrant
    psf = np.sum(p3d, axis=2)
    for i in range(p3d.shape[0]):
        j_mirror = psf.shape[1] - i - 1
        psf[i, i] = p3d[i, i, np.argmax(np.abs(p3d[i, i, :]))]
        psf[i, j_mirror] = p3d[i, j_mirror, np.argmax(np.abs(p3d[i, j_mirror, :]))]

    return psf


def compute_psf_all_depths(Camera: dict, Resolution: dict) -> np.ndarray:
    """
    Compute the PSF wave stack for all reconstruction depths.

    Exploits conjugate symmetry: PSF(-Δz) = conj(PSF(+Δz)), so each unique
    |depth| is integrated only once.

    Parameters
    ----------
    Camera : dict
        Camera parameters (offsetFobj applied to shift depths for defocused LFMs).
    Resolution : dict
        Must contain 'depths', 'yspace', 'xspace'.

    Returns
    -------
    np.ndarray
        Complex array, shape (len(yspace), len(xspace), nDepths).
    """
    # Apply defocus offset (zero for plenoptic=1)
    Resolution["depths"] = Resolution["depths"] + Camera["offsetFobj"]

    ny = len(Resolution["yspace"])
    nx = len(Resolution["xspace"])
    nd = len(Resolution["depths"])
    psf_stack = np.zeros((ny, nx, nd), dtype="complex128")

    print("  Computing wave-optics PSF at each depth:")
    for i in range(nd):
        compute_this = True
        src_idx = 0
        if i > 0:
            prev = np.where(np.abs(Resolution["depths"][:i]) == np.abs(Resolution["depths"][i]))[0]
            if prev.size > 0:
                compute_this = False
                src_idx = prev[0]

        if compute_this:
            tic = time.time()
            psf = compute_psf_single_depth(0, 0, Resolution["depths"][i], Camera, Resolution)
            print(f"    depth {i+1}/{nd} (Δz={Resolution['depths'][i]:.1f} μm): {time.time()-tic:.1f}s")
        else:
            if Resolution["depths"][i] == Resolution["depths"][src_idx]:
                psf = psf_stack[:, :, src_idx]
            else:
                psf = np.conj(psf_stack[:, :, src_idx])
            print(f"    depth {i+1}/{nd} (Δz={Resolution['depths'][i]:.1f} μm): reused conjugate")

        psf_stack[:, :, i] = psf

    return psf_stack


# ═══════════════════════════════════════════════════════════════════════════════
# MLA Transmittance
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ulens_transmittance(Camera: dict, Resolution: dict) -> np.ndarray:
    """
    Single micro-lens complex phase transmittance t(x_l, y_l).

    Implements Eq. (14): t = P(x,y) · exp(-ik/(2f_ml) · (x²+y²))
    P is the square aperture mask (uLensMask=1) or circular aperture.

    Parameters
    ----------
    Camera : dict
    Resolution : dict
        Must contain 'yMLspace', 'xMLspace', 'sensMask', 'maskFlag'.

    Returns
    -------
    np.ndarray
        Complex array, shape (len(yMLspace), len(xMLspace)).
    """
    pattern = np.zeros((len(Resolution["yMLspace"]), len(Resolution["xMLspace"])),
                       dtype="complex128")
    for a, y in enumerate(Resolution["yMLspace"]):
        for b, x in enumerate(Resolution["xMLspace"]):
            pattern[a, b] = np.exp(-1j * Camera["k"] / (2 * Camera["fm"]) * (y**2 + x**2))

    if Resolution["maskFlag"] == 1:
        pattern[Resolution["sensMask"] == 0] = 0
    else:
        x_g, y_g = np.meshgrid(Resolution["xMLspace"], Resolution["yMLspace"])
        pattern[np.sqrt(x_g**2 + y_g**2) >= Camera["lensPitch"] / 2 - 3] = 0

    return pattern


def compute_mla_transmittance(Camera: dict, Resolution: dict,
                               ulens_pattern: np.ndarray) -> np.ndarray:
    """
    Build the full MLA transmittance T by tiling the single-lens pattern.

    Implements Eq. (13): T = rep_{p_ml}(t(x_l, y_l)).
    Constructs an extended array, places delta functions at each lenslet center,
    then convolves with ulens_pattern to replicate the lens at each position.

    Parameters
    ----------
    Camera : dict
    Resolution : dict
        Must contain 'usedLensletCenters', 'yspace', 'xspace', 'yMLspace', 'xMLspace'.
    ulens_pattern : np.ndarray
        Single-lens transmittance from compute_ulens_transmittance.

    Returns
    -------
    np.ndarray
        Complex MLA transmittance, shape (len(yspace), len(xspace)).
    """
    ny = len(Resolution["yspace"])
    nx = len(Resolution["xspace"])
    ny_ml = len(Resolution["yMLspace"])
    nx_ml = len(Resolution["xMLspace"])

    ny_ext = ny + 2 * ny_ml
    nx_ext = nx + 2 * nx_ml

    centers_off = np.zeros((*Resolution["usedLensletCenters"]["px"].shape[:2], 2), dtype="int64")
    centers_off[..., 0] = (np.round(Resolution["usedLensletCenters"]["px"][..., 0])
                           + np.ceil(ny_ext / 2)).astype(int)
    centers_off[..., 1] = (np.round(Resolution["usedLensletCenters"]["px"][..., 1])
                           + np.ceil(nx_ext / 2)).astype(int)

    MLcenters = np.zeros((ny_ext, nx_ext), dtype="complex128")
    for a in range(centers_off.shape[0]):
        for b in range(centers_off.shape[1]):
            cy, cx = centers_off[a, b, 0], centers_off[a, b, 1]
            if 1 <= cy <= ny_ext and 1 <= cx <= nx_ext:
                MLcenters[cy - 1, cx - 1] = 1.0 + 0j

    MLARRAY_ext = convolve2d(MLcenters, ulens_pattern, mode="same")

    r0 = int(np.ceil(ny_ext / 2) - np.floor(ny / 2)) - 1
    c0 = int(np.ceil(nx_ext / 2) - np.floor(nx / 2)) - 1
    MLARRAY = MLARRAY_ext[r0:r0 + ny, c0:c0 + nx]
    return MLARRAY


# ═══════════════════════════════════════════════════════════════════════════════
# Rayleigh-Sommerfeld Propagation
# ═══════════════════════════════════════════════════════════════════════════════

def propagate_to_sensor(field: np.ndarray, sensor_res: np.ndarray,
                         z: float, wavelength: float,
                         ideal_sampling: bool = False) -> np.ndarray:
    """
    Propagate a wavefield from the MLA to the sensor using the
    Rayleigh-Sommerfeld angular spectrum method.

    Implements Eq. (15-16): U_sens = F^{-1}{ F{U_mla+} · H_rs }
    where H_rs = exp(ik·z·sqrt(1 - λ²(f_x²+f_y²))).

    Parameters
    ----------
    field : np.ndarray
        Complex wavefield at MLA plane, shape (Ny, Nx).
    sensor_res : np.ndarray
        Pixel size [dy, dx] in um.
    z : float
        Propagation distance (um): mla2sensor.
    wavelength : float
        Emission wavelength (um).
    ideal_sampling : bool
        If True, resample to ideal Nyquist rate before propagation (slower).
        If False, propagate at native sampling rate (faster, default).

    Returns
    -------
    np.ndarray
        Complex wavefield at sensor, same shape as input.
    """
    if z == 0:
        return field

    Ny, Nx = field.shape
    k = 2 * np.pi / wavelength

    if ideal_sampling:
        from scipy.ndimage import zoom
        Lx = Ny * sensor_res[0]
        Ly = Nx * sensor_res[1]
        ideal_rate = np.array([wavelength * np.sqrt(z**2 + (Lx / 2)**2) / Lx,
                                wavelength * np.sqrt(z**2 + (Ly / 2)**2) / Ly])
        ideal_n = np.ceil(np.array([Lx, Ly]) / ideal_rate).astype(int)
        ideal_n = ideal_n + (1 - ideal_n % 2)
        rate = np.array([Lx, Ly]) / ideal_n

        du = 1.0 / (ideal_n[0] * float(rate[0]))
        dv = 1.0 / (ideal_n[1] * float(rate[1]))
        u = np.hstack([np.arange(np.ceil(ideal_n[0] / 2)), np.arange(-np.floor(ideal_n[0] / 2), 0)]) * du
        v = np.hstack([np.arange(np.ceil(ideal_n[1] / 2)), np.arange(-np.floor(ideal_n[1] / 2), 0)]) * dv
        H = np.exp(1j * np.sqrt(1 - wavelength**2 * (np.tile(u, (len(v), 1)).T**2
                                                       + np.tile(v, (len(u), 1))**2)) * z * k)
        field_z = zoom(field, ideal_n, order=3)
        out = ifft2(fft2(field_z.T, norm="ortho") * H, norm="ortho").T
        out = zoom(out, field.shape, order=3)
    else:
        du = 1.0 / (Ny * float(sensor_res[0]))
        dv = 1.0 / (Nx * float(sensor_res[1]))
        u = np.hstack([np.arange(np.ceil(Ny / 2)), np.arange(-np.floor(Ny / 2), 0)]) * du
        v = np.hstack([np.arange(np.ceil(Nx / 2)), np.arange(-np.floor(Nx / 2), 0)]) * dv
        H = np.exp(1j * np.sqrt(1 - wavelength**2 * (np.tile(u, (len(v), 1)).T**2
                                                       + np.tile(v, (len(u), 1))**2)) * z * k)
        out = np.exp(1j * k * z) * ifft2(fft2(field, norm="ortho") * H, norm="ortho")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Image Shift Helper
# ═══════════════════════════════════════════════════════════════════════════════

def _imshift(image: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    """Shift a 2D (or 3D) image by (shift_x, shift_y) using linear interpolation."""
    if image.ndim == 2:
        return shift(image, (shift_x, shift_y), order=1)
    out = np.zeros_like(image)
    for i in range(image.shape[-1]):
        out[..., i] = shift(image[..., i], (shift_x, shift_y), order=1)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Forward / Backward Patterns
# ═══════════════════════════════════════════════════════════════════════════════

def compute_forward_patterns(psf_wave_stack: np.ndarray, mlarray: np.ndarray,
                              Camera: dict, Resolution: dict) -> np.ndarray:
    """
    Compute the forward projection operator H.

    For each (aa, bb) texture coordinate and each depth c:
      1. Shift the native-plane PSF to position (aa, bb)
      2. Multiply by MLA transmittance
      3. Propagate to sensor via RS diffraction
      4. Shift back to center; store |field|² as sparse matrix

    Exploits quarter-symmetry for regular grids: only the upper-left
    quadrant (aa < TexNnum_half, bb < TexNnum_half) is computed explicitly.

    Parameters
    ----------
    psf_wave_stack : np.ndarray
        Complex PSF stack, shape (ny, nx, nDepths).
    mlarray : np.ndarray
        Complex MLA transmittance, shape (ny, nx).
    Camera : dict
    Resolution : dict
        Must contain 'TexNnum', 'TexNnum_half', 'texScaleFactor', 'sensorRes', 'depths'.

    Returns
    -------
    np.ndarray
        Object array H, shape (coords_range[0], coords_range[1], nDepths).
        Each element is a csr_matrix.
    """
    if Camera["range"] == "quarter":
        coords_range = Resolution["TexNnum_half"]
    else:
        coords_range = Resolution["TexNnum"]

    half_coord = Resolution["TexNnum_half"] // Resolution["texScaleFactor"]
    sensor_res = Resolution["sensorRes"]
    nd = len(Resolution["depths"])

    H = np.empty((coords_range[0], coords_range[1], nd), dtype=object)

    for c in trange(nd, ncols=70, desc="  Forward patterns"):
        psf_ref = psf_wave_stack[:, :, c]
        for i in range(coords_range[0]):
            aa_tex = i + 1
            aa_sensor = aa_tex / Resolution["texScaleFactor"][0]
            for j in range(coords_range[1]):
                bb_tex = j + 1
                bb_sensor = bb_tex / Resolution["texScaleFactor"][1]

                shift_x = round(aa_sensor - half_coord[0])
                shift_y = round(bb_sensor - half_coord[1])

                psf_shifted = _imshift(psf_ref, shift_x, shift_y)
                psf_mla = psf_shifted * mlarray
                lf_psf = propagate_to_sensor(psf_mla, sensor_res,
                                              Camera["mla2sensor"], Camera["WaveLength"],
                                              ideal_sampling=False)
                lf_psf = _imshift(lf_psf, -shift_x, -shift_y)
                H[i, j, c] = csr_matrix(np.abs(lf_psf) ** 2)

    return H


def _sconv2_single_point_flip(size_a: tuple, point: tuple, B: np.ndarray,
                               flip_x: bool, flip_y: bool, shape: str) -> np.ndarray:
    """
    Sparse convolution of a single point with a (possibly flipped) filter B.
    Adapted from pyolaf/lf.py::sconv2singlePointFlip.
    """
    m, n = size_a
    p, q = B.shape
    i, j = point

    ky, kx = np.nonzero(B)
    bvals = B[ky, kx].astype(float)
    if flip_x:
        ky = p - ky - 1
    if flip_y:
        kx = q - kx - 1
    ky += 1
    kx += 1

    I = i + ky - 1
    J = j + kx - 1
    C_vals = bvals

    shape = shape.lower()
    if shape == "full":
        out = coo_matrix((C_vals, (I, J)), shape=(m + p - 1, n + q - 1))
    elif shape == "same":
        cy = int(np.ceil((p + 1) / 2))
        cx = int(np.ceil((q + 1) / 2))
        II = I - cy
        JJ = J - cx
        mask = (II >= 0) & (II < m) & (JJ >= 0) & (JJ < n)
        out = coo_matrix((C_vals[mask], (II[mask], JJ[mask])), shape=(m, n))
    elif shape == "valid":
        mn0 = max(m - max(0, p - 1), 0)
        mn1 = max(n - max(0, q - 1), 0)
        II = I - p
        JJ = J - q
        mask = (II >= 0) & (II < mn0) & (JJ >= 0) & (JJ < mn1)
        out = coo_matrix((C_vals[mask], (II[mask], JJ[mask])), shape=(mn0, mn1))
    else:
        raise ValueError(f"Unknown shape mode: {shape}")

    return out.toarray()


def _backproject_single_pixel(H: np.ndarray, Resolution: dict, img_size: np.ndarray,
                               tex_size: np.ndarray, current_pixel: list,
                               lenslet_centers: dict, crange: str) -> np.ndarray:
    """
    Compute the object-space response from a single sensor pixel.

    For each texture coordinate (aa, bb) and depth c, computes the contribution
    of the sensor pixel to volume voxels via the rotated PSF H[aa, bb, c].
    """
    nd = H.shape[2]
    backproj = np.zeros((tex_size[0], tex_size[1], nd))

    lens_vox_y = lenslet_centers["vox"][:, :, 0]
    lens_vox_x = lenslet_centers["vox"][:, :, 1]
    lens_sen_y = lenslet_centers["px"][:, :, 0]
    lens_sen_x = lenslet_centers["px"][:, :, 1]

    for c in range(nd):
        slice_c = np.zeros(tex_size)
        for aa in range(Resolution["TexNnum"][0]):
            for bb in range(Resolution["TexNnum"][1]):
                if Resolution["texMask"][aa, bb] == 0:
                    continue

                aa_new, flip_x = aa, False
                if aa > Resolution["TexNnum_half"][0] - 1 and crange == "quarter":
                    aa_new = Resolution["TexNnum"][0] - aa - 1
                    flip_x = True

                bb_new, flip_y = bb, False
                if bb > Resolution["TexNnum_half"][1] - 1 and crange == "quarter":
                    bb_new = Resolution["TexNnum"][1] - bb - 1
                    flip_y = True

                Ht_aa_bb = np.flip(np.flip(H[aa_new, bb_new, c].toarray(), 0), 1)
                temp = _sconv2_single_point_flip(
                    tuple(img_size), current_pixel, Ht_aa_bb, flip_x, flip_y, "same")

                ly = np.round(lens_vox_y - Resolution["TexNnum_half"][0] + aa).astype(int)
                lx = np.round(lens_vox_x - Resolution["TexNnum_half"][1] + bb).astype(int)
                valid_tex = ((lx < tex_size[1]) & (lx >= 0) & (ly < tex_size[0]) & (ly >= 0))

                sy = -1 + lens_sen_y + np.round((-Resolution["TexNnum_half"][0] + aa + 1)
                                                  / Resolution["texScaleFactor"][0])
                sx = -1 + lens_sen_x + np.round((-Resolution["TexNnum_half"][1] + bb + 1)
                                                  / Resolution["texScaleFactor"][1])
                sy, sx = sy.astype(int), sx.astype(int)
                valid_img = ((sx < img_size[1]) & (sx >= 0) & (sy < img_size[0]) & (sy >= 0))

                valid = valid_img & valid_tex
                if np.sum(valid) > 0:
                    slice_c[ly[valid], lx[valid]] += temp[sy[valid], sx[valid]]

        backproj[:, :, c] += slice_c

    return backproj


def compute_backward_patterns(H: np.ndarray, Resolution: dict,
                               Camera: dict) -> np.ndarray:
    """
    Compute the backward projection operator Ht, then normalize.

    For each sensor pixel (aa_sen, bb_sen) within one lenslet:
      1. Back-project via H (rotated PSF) onto the volume
      2. Center-shift the result
      3. Normalize so each pixel's total contribution sums to 1

    Parameters
    ----------
    H : np.ndarray
        Forward operator from compute_forward_patterns.
    Resolution : dict
    Camera : dict
        Needs 'range'.

    Returns
    -------
    np.ndarray
        Ht object array, shape (Nnum_half[0], Nnum_half[1], nDepths).
    """
    nd = H.shape[2]
    img_size = np.array(H[0, 0, 0].shape)

    tex_size = np.ceil(img_size * np.array(Resolution["texScaleFactor"])).astype(int)
    tex_size = tex_size + (1 - tex_size % 2)

    offset_img = np.ceil(img_size / 2).astype(int)
    offset_vol = np.ceil(tex_size / 2).astype(int)

    lc = {
        "px":  np.copy(Resolution["usedLensletCenters"]["px"]),
        "vox": np.copy(Resolution["usedLensletCenters"]["vox"]),
    }
    lc["px"][:, :, 0] += offset_img[0]
    lc["px"][:, :, 1] += offset_img[1]
    lc["vox"][:, :, 0] += offset_vol[0]
    lc["vox"][:, :, 1] += offset_vol[1]

    crange = Camera["range"]
    if crange == "quarter":
        coords_range = Resolution["Nnum_half"]
    else:
        coords_range = Resolution["Nnum"]

    Ht = np.empty((coords_range[0], coords_range[1], nd), dtype=object)

    for aa in trange(coords_range[0], ncols=70, desc="  Backward patterns"):
        aa_tex = int(np.ceil((1 + aa) * Resolution["texScaleFactor"][0]))
        for bb in range(coords_range[1]):
            bb_tex = int(np.ceil((1 + bb) * Resolution["texScaleFactor"][1]))

            cur_px = [aa + offset_img[0] - Resolution["Nnum_half"][0],
                      bb + offset_img[1] - Resolution["Nnum_half"][1]]
            bp = _backproject_single_pixel(H, Resolution, img_size, tex_size, cur_px, lc, crange)

            for c in range(nd):
                sy = int(np.round(Resolution["TexNnum_half"][0] - aa_tex))
                sx = int(np.round(Resolution["TexNnum_half"][1] - bb_tex))
                shifted = _imshift(bp[:, :, c], sy, sx)
                Ht[aa, bb, c] = csr_matrix(shifted)

    # Normalize Ht so each sensor pixel's total contribution sums to 1
    for aa in range(Ht.shape[0]):
        for bb in range(Ht.shape[1]):
            s = sum(Ht[aa, bb, c].sum() for c in range(nd))
            if not np.isclose(s, 0):
                for c in range(nd):
                    Ht[aa, bb, c] = Ht[aa, bb, c] / s

    return Ht


def _threshold_small_values(H: np.ndarray, tol: float = 0.005) -> np.ndarray:
    """Zero out values below tol*max and renormalize each PSF pattern."""
    for a in range(H.shape[0]):
        for b in range(H.shape[1]):
            for c in range(H.shape[2]):
                arr = H[a, b, c].toarray()
                mx = arr.max()
                arr[arr < mx * tol] = 0
                s = arr.sum()
                if s > 0:
                    arr /= s
                H[a, b, c] = csr_matrix(arr)
    return H


def compute_lf_operators(Camera: dict, Resolution: dict,
                          lenslet_centers: dict) -> tuple:
    """
    Top-level entry point: compute (H, Ht) for the LFM configuration.

    Steps:
      1. Estimate PSF footprint size; extract used lenslet centers
      2. Set up sensor/ML coordinate grids in Resolution
      3. Compute Debye PSF stack at all depths
      4. Compute MLA transmittance MLARRAY
      5. Compute forward patterns H; threshold small values
      6. Compute backward patterns Ht; normalize

    Parameters
    ----------
    Camera : dict
    Resolution : dict
    lenslet_centers : dict

    Returns
    -------
    H, Ht : np.ndarray (object dtype)
    """
    # PSF footprint
    depth_range = np.array(Resolution["depthRange"]) + Camera["offsetFobj"]
    p = np.argmax(np.abs(depth_range))
    psf_size = compute_psf_size(depth_range[p], Camera)

    used_centers = get_used_lenslet_centers(psf_size, lenslet_centers)
    Resolution["usedLensletCenters"] = used_centers

    # Coordinate spaces
    half = int(max(Resolution["Nnum"][1] * psf_size,
                   2 * Resolution["Nnum"][1]))
    print(f"  PSF image size: {2*half+1} × {2*half+1} pixels")

    Resolution["yspace"] = Resolution["sensorRes"][0] * np.arange(-half, half + 1)
    Resolution["xspace"] = Resolution["sensorRes"][1] * np.arange(-half, half + 1)
    Resolution["yMLspace"] = Resolution["sensorRes"][0] * np.arange(
        -Resolution["Nnum_half"][0] + 1, Resolution["Nnum_half"][0])
    Resolution["xMLspace"] = Resolution["sensorRes"][1] * np.arange(
        -Resolution["Nnum_half"][1] + 1, Resolution["Nnum_half"][1])

    # PSF stack
    print("Step 1/3: Computing wave-optics PSF at all depths...")
    psf_stack = compute_psf_all_depths(Camera, Resolution)

    # MLA transmittance
    print("Step 2/3: Computing MLA transmittance...")
    ulens = compute_ulens_transmittance(Camera, Resolution)
    mlarray = compute_mla_transmittance(Camera, Resolution, ulens)

    # Forward patterns
    print("Step 3a/3: Computing forward patterns H...")
    H = compute_forward_patterns(psf_stack, mlarray, Camera, Resolution)
    H = _threshold_small_values(H, tol=0.005)

    # Backward patterns
    print("Step 3b/3: Computing backward patterns Ht...")
    Ht = compute_backward_patterns(H, Resolution, Camera)

    return H, Ht


# ═══════════════════════════════════════════════════════════════════════════════
# FFT Convolution (numpy, no CuPy)
# ═══════════════════════════════════════════════════════════════════════════════

def _fft_conv2d(a: np.ndarray, b: np.ndarray, mode: str = "same") -> np.ndarray:
    """
    2D FFT-based convolution (numpy fallback of pyolaf/fftpack.py::cufftconv).

    Parameters
    ----------
    a, b : np.ndarray
        2D arrays to convolve.
    mode : str
        'full', 'same', or 'valid'.

    Returns
    -------
    np.ndarray
        Convolution result.
    """
    from scipy.fft import next_fast_len
    s1 = np.array(a.shape[-2:])
    s2 = np.array(b.shape[-2:])
    shape = s1 + s2 - 1
    fshape = [next_fast_len(int(d), True) for d in shape]

    sp1 = np.fft.rfft2(a, fshape)
    sp2 = np.fft.rfft2(b, fshape)
    ret = np.fft.irfft2(sp1 * sp2, fshape)

    # Crop
    if mode == "full":
        cropped = ret[:shape[0], :shape[1]]
    elif mode == "same":
        start = (np.array(shape) - s1) // 2
        cropped = ret[start[0]:start[0] + s1[0], start[1]:start[1] + s1[1]]
    elif mode == "valid":
        vs = s1 - s2 + 1
        start = (np.array(shape) - vs) // 2
        cropped = ret[start[0]:start[0] + vs[0], start[1]:start[1] + vs[1]]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return cropped


# ═══════════════════════════════════════════════════════════════════════════════
# Forward and Backward Projection (fast, uses precomputed H/Ht)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_indices_forward(lenslet_centers: dict, Resolution: dict,
                          img_size: np.ndarray, tex_size: np.ndarray) -> tuple:
    """Precompute texture→image index mappings for forward projection."""
    texnum = Resolution["TexNnum"]
    texnum_half = Resolution["TexNnum_half"]
    tex_scale = Resolution["texScaleFactor"]

    offset_img = np.ceil(img_size / 2)
    offset_vol = np.ceil(tex_size / 2)
    lY = lenslet_centers["vox"][:, :, 0] + offset_vol[0]
    lX = lenslet_centers["vox"][:, :, 1] + offset_vol[1]

    idx_tex, idx_img = {}, {}
    for aa in range(texnum[0]):
        for bb in range(texnum[1]):
            lYc = np.round(lY - texnum_half[0] + aa + 1)
            lXc = np.round(lX - texnum_half[1] + bb + 1)

            lYp = np.ceil((lYc - offset_vol[0]) / tex_scale[0] + offset_img[0]) - 1
            lXp = np.ceil((lXc - offset_vol[1]) / tex_scale[1] + offset_img[1]) - 1

            valid = ((lYc < tex_size[0]) & (lYc >= 0) & (lXc < tex_size[1]) & (lXc >= 0)
                     & (lYp < img_size[0]) & (lYp >= 0) & (lXp < img_size[1]) & (lXp >= 0))

            idx_tex[(aa, bb)] = (lYc[valid].astype(int), lXc[valid].astype(int))
            idx_img[(aa, bb)] = (lYp[valid].astype(int), lXp[valid].astype(int))

    return idx_tex, idx_img


def forward_project(H: np.ndarray, volume: np.ndarray,
                    lenslet_centers: dict, Resolution: dict,
                    img_size: np.ndarray, Camera: dict) -> np.ndarray:
    """
    Forward project a 3D volume to a 2D light field image: m_hat = A·v.

    For each depth c and texture coordinate (aa, bb):
      1. Scatter volume voxels to a temporary image at lenslet positions
      2. Convolve with H[aa, bb, c] PSF via FFT
      3. Accumulate into projection image

    Parameters
    ----------
    H : np.ndarray
        Forward operator (object array of csr_matrix).
    volume : np.ndarray
        3D volume, shape (texH, texW, nDepths), float32.
    lenslet_centers : dict
    Resolution : dict
    img_size : np.ndarray
        Output image size [H, W].
    Camera : dict

    Returns
    -------
    np.ndarray
        Light field image, shape img_size, float32.
    """
    texnum = Resolution["TexNnum"]
    texMask = Resolution["texMask"]
    tex_size = np.array(volume.shape[:2])
    nd = H.shape[2]
    crange = Camera["range"]

    idx_tex, idx_img = _get_indices_forward(lenslet_centers, Resolution, img_size, tex_size)

    # Extract dense filters
    texnum_half = Resolution["TexNnum_half"]
    fshape = H[0, 0, 0].shape
    filters = np.zeros((texnum[0], texnum[1], nd, fshape[0], fshape[1]), dtype="float32")
    for c in range(nd):
        for aa in range(texnum[0]):
            aa_new = aa
            flip_x = False
            if crange == "quarter" and aa >= texnum_half[0]:
                aa_new = texnum[0] - aa - 1
                flip_x = True
            for bb in range(texnum[1]):
                if texMask[aa, bb] == 0:
                    continue
                bb_new = bb
                flip_y = False
                if crange == "quarter" and bb >= texnum_half[1]:
                    bb_new = texnum[1] - bb - 1
                    flip_y = True
                h = H[aa_new, bb_new, c].toarray()
                if flip_x:
                    h = np.flipud(h)
                if flip_y:
                    h = np.fliplr(h)
                filters[aa, bb, c] = h

    projection = np.zeros(img_size, dtype="float32")
    for c in trange(nd, ncols=70, desc="  Forward project"):
        vol_c = volume[:, :, c]
        for aa in range(texnum[0]):
            for bb in range(texnum[1]):
                if texMask[aa, bb] == 0:
                    continue
                it = idx_tex[(aa, bb)]
                ii = idx_img[(aa, bb)]
                tmp = np.zeros(img_size, dtype="float32")
                tmp[ii] = vol_c[it]
                projection += _fft_conv2d(tmp, filters[aa, bb, c], mode="same").astype("float32")

    return projection


def _get_indices_backward(lenslet_centers: dict, Resolution: dict,
                           img_size: np.ndarray, tex_size: np.ndarray) -> tuple:
    """Precompute image→texture index mappings for backward projection."""
    num = Resolution["Nnum"]
    num_half = Resolution["Nnum_half"]
    tex_scale = Resolution["texScaleFactor"]

    offset_img = np.ceil(img_size / 2)
    offset_vol = np.ceil(tex_size / 2)
    lY = lenslet_centers["px"][:, :, 0] + offset_img[0]
    lX = lenslet_centers["px"][:, :, 1] + offset_img[1]

    idx_tex, idx_img = {}, {}
    for aa in range(num[0]):
        for bb in range(num[1]):
            lYp = np.round(lY - num_half[0] + aa)
            lXp = np.round(lX - num_half[1] + bb)

            lYv = np.ceil((lYp - offset_img[0] + 1) * tex_scale[0] + offset_vol[0]) - 1
            lXv = np.ceil((lXp - offset_img[1] + 1) * tex_scale[1] + offset_vol[1]) - 1

            valid = ((lYv < tex_size[0]) & (lYv >= 0) & (lXv < tex_size[1]) & (lXv >= 0)
                     & (lYp < img_size[0]) & (lYp >= 0) & (lXp < img_size[1]) & (lXp >= 0))

            idx_img[(aa, bb)] = (lYp[valid].astype(int), lXp[valid].astype(int))
            idx_tex[(aa, bb)] = (lYv[valid].astype(int), lXv[valid].astype(int))

    return idx_tex, idx_img


def backward_project(Ht: np.ndarray, lf_image: np.ndarray,
                     lenslet_centers: dict, Resolution: dict,
                     tex_size: np.ndarray, Camera: dict) -> np.ndarray:
    """
    Backward project a 2D light field image to a 3D volume: v_hat = A^T · m.

    Parameters
    ----------
    Ht : np.ndarray
        Backward operator (object array of csr_matrix).
    lf_image : np.ndarray
        2D sensor image, float32.
    lenslet_centers : dict
    Resolution : dict
    tex_size : np.ndarray
        Output texture size [texH, texW].
    Camera : dict

    Returns
    -------
    np.ndarray
        Back-projected volume, shape (texH, texW, nDepths), float32.
    """
    num = Resolution["Nnum"]
    num_half = Resolution["Nnum_half"]
    sensMask = Resolution["sensMask"]
    nd = Ht.shape[2]
    img_size = np.array(lf_image.shape)
    crange = Camera["range"]

    idx_tex, idx_img = _get_indices_backward(lenslet_centers, Resolution, img_size, tex_size)

    # Extract dense backward filters
    fshape = Ht[0, 0, 0].shape
    filters = np.zeros((num[0], num[1], nd, fshape[0], fshape[1]), dtype="float32")
    for c in range(nd):
        for aa in range(num[0]):
            aa_new = aa
            flip_x = False
            if crange == "quarter" and aa >= num_half[0]:
                aa_new = num[0] - aa - 1
                flip_x = True
            for bb in range(num[1]):
                if sensMask[aa, bb] == 0:
                    continue
                bb_new = bb
                flip_y = False
                if crange == "quarter" and bb >= num_half[1]:
                    bb_new = num[1] - bb - 1
                    flip_y = True
                h = Ht[aa_new, bb_new, c].toarray()
                if flip_x:
                    h = np.flipud(h)
                if flip_y:
                    h = np.fliplr(h)
                filters[aa, bb, c] = h

    backproj = np.zeros((tex_size[0], tex_size[1], nd), dtype="float32")
    for c in trange(nd, ncols=70, desc="  Backward project"):
        for aa in range(num[0]):
            for bb in range(num[1]):
                if sensMask[aa, bb] == 0:
                    continue
                ii = idx_img[(aa, bb)]
                it = idx_tex[(aa, bb)]
                tmp = np.zeros(tex_size, dtype="float32")
                tmp[it] = lf_image[ii]
                backproj[:, :, c] += _fft_conv2d(tmp, filters[aa, bb, c], mode="same").astype("float32")

    return backproj
