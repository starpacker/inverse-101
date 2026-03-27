"""
Generate Data — Synthetic USAF 1951 Resolution Target and Light Field Image
===========================================================================

Creates a synthetic USAF 1951 resolution target volume and simulates the light
field measurement via the wave-optics forward model. Used to produce the
reference dataset (data/raw_data.npz) for this benchmark task.

The scene places a flat USAF target at the native object plane (z=0), with three
groups of bar triplets at spatial frequencies 128, 256, and 513 lp/mm. This
directly demonstrates the aliasing artifacts in standard RL deconvolution at the
native object plane, and the artifact-free reconstruction via EMS.
"""

import os
import pickle
import numpy as np

from .preprocessing import load_metadata, set_camera_params, compute_geometry
from .physics_model import compute_lf_operators, forward_project


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Bead Volume
# ═══════════════════════════════════════════════════════════════════════════════

def generate_bead_volume(metadata: dict, Resolution: dict,
                          rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a 3D fluorescent bead volume of shape (texH, texW, nDepths).

    Places one bead per depth plane at random lateral positions. Each bead
    is modeled as a 2D Gaussian (xy) with sigma = bead_radius_um / voxel_size_um.
    Beads do not span multiple depth planes (bead radius << depth step).

    Parameters
    ----------
    metadata : dict
        Parsed meta_data; uses 'synthetic_data' sub-dict for bead parameters.
    Resolution : dict
        Resolution dict from preprocessing.compute_geometry.
        Must contain 'texRes', 'depths'.
    rng : np.random.Generator, optional
        Random number generator (seeded externally for reproducibility).

    Returns
    -------
    np.ndarray
        Float32 volume, shape (texH, texW, nDepths), values in [0, 1].
    """
    syn = metadata["synthetic_data"]
    bead_radius_um = syn.get("bead_radius_um", 2.0)
    bead_intensity = syn.get("bead_intensity", 1.0)
    background = syn.get("background", 0.01)

    if rng is None:
        rng = np.random.default_rng(syn.get("random_seed", 42))

    tex_res = Resolution["texRes"]   # [dy_um, dx_um, dz_um]
    depths = Resolution["depths"]
    nd = len(depths)

    # Volume dimensions: we'll figure out texH, texW from the calling code
    # which passes them as tex_size. We derive them from depths count and
    # the sensor grid stored in the Resolution dict.
    # TexNnum = [texH_per_lenslet, texW_per_lenslet], but actual volume size
    # comes from the caller via lenslet_centers shape * TexNnum.
    # Here we use a simple default based on what generate_synthetic_dataset passes.
    tex_size = Resolution.get("_tex_size", None)
    if tex_size is None:
        raise ValueError("Resolution must contain '_tex_size' key. "
                         "Call generate_synthetic_dataset instead.")

    texH, texW = int(tex_size[0]), int(tex_size[1])
    volume = np.full((texH, texW, nd), background, dtype="float32")

    # Sigma in voxels
    sigma_y = bead_radius_um / tex_res[0]
    sigma_x = bead_radius_um / tex_res[1]

    # Coordinate grids
    yy = np.arange(texH)
    xx = np.arange(texW)
    Xg, Yg = np.meshgrid(xx, yy)

    for d in range(nd):
        # Random lateral position (keep bead 3-sigma away from edges)
        margin_y = max(1, int(3 * sigma_y))
        margin_x = max(1, int(3 * sigma_x))
        cy = rng.integers(margin_y, texH - margin_y)
        cx = rng.integers(margin_x, texW - margin_x)

        bead_xy = bead_intensity * np.exp(
            -0.5 * (((Yg - cy) / sigma_y) ** 2 + ((Xg - cx) / sigma_x) ** 2)
        )
        volume[:, :, d] += bead_xy.astype("float32")

    # Clip to [0, 1]
    volume = np.clip(volume, 0.0, 1.0)
    return volume


# ═══════════════════════════════════════════════════════════════════════════════
# USAF 1951 Resolution Target
# ═══════════════════════════════════════════════════════════════════════════════

def _build_usaf_pattern(H: int, W: int, voxel_um: float) -> np.ndarray:
    """
    Build a USAF 1951-like resolution test target pattern.

    Three bar groups at different spatial frequencies fill the H×W grid.
    Each group has 3 vertical bars (period along X) and 3 horizontal bars
    (period along Y), mimicking the standard USAF 1951 element layout.

    Layout (for the default 35×35 grid at voxel_um ≈ 0.975 μm):
      - V bars: top section, groups 1–3 side by side, spanning the full width
        (5×4 + 5×2 + 5×1 = 35 columns)
      - H bars: below the V bars, within each group's column band

    Bar widths and corresponding spatial frequencies (at voxel_um = 1.5 μm):
      Group 1  bw=4 →  6 μm bar width →  ~83 lp/mm  (USAF Group 6, Elem 4)
      Group 2  bw=2 →  3 μm bar width →  ~167 lp/mm (USAF Group 7, Elem 2)
      Group 3  bw=1 →  1.5 μm bar width → ~333 lp/mm (near optical limit)

    Parameters
    ----------
    H, W : int
        Grid dimensions in voxels.
    voxel_um : float
        Voxel size in μm (used only for annotation; does not affect layout).

    Returns
    -------
    np.ndarray
        Float32 binary pattern, shape (H, W), values in {0, 1}.
    """
    pattern = np.zeros((H, W), dtype=np.float32)

    # (bar_width_vox, v_bar_length, h_bar_length)
    # Groups tile the full width: 5*4 + 5*2 + 5*1 = 35
    groups = [(4, 10, 10), (2, 8, 8), (1, 6, 5)]
    n_bars = 3
    v_row0 = 1  # V bars top margin
    h_row0 = v_row0 + max(vbl for _, vbl, _ in groups) + 2  # H bars start row

    c0 = 0
    for bw, vbl, hbl in groups:
        # V bars: 3 vertical bars side by side (period along X)
        for i in range(n_bars):
            x0 = c0 + i * 2 * bw
            x1 = x0 + bw
            y1 = v_row0 + vbl
            if y1 <= H and x1 <= W:
                pattern[v_row0:y1, x0:x1] = 1.0

        # H bars: 3 horizontal bars stacked (period along Y)
        for i in range(n_bars):
            y0 = h_row0 + i * 2 * bw
            y1 = y0 + bw
            x1 = c0 + hbl
            if y1 <= H and x1 <= W:
                pattern[y0:y1, c0:x1] = 1.0

        c0 += 5 * bw

    return pattern


def generate_usaf_volume(metadata: dict, Resolution: dict) -> np.ndarray:
    """
    Generate a USAF 1951 resolution target at a single depth plane.

    Places the test pattern at the depth plane nearest to the target depth
    specified in metadata['usaf_data']['target_depth'] (default 0 μm = native
    object plane). All other depth planes are set to the background value.

    Parameters
    ----------
    metadata : dict
        Parsed meta_data; uses 'usaf_data' sub-dict for parameters.
    Resolution : dict
        Resolution dict from compute_geometry.
        Must contain '_tex_size', 'texRes', 'depths'.

    Returns
    -------
    np.ndarray
        Float32 volume, shape (texH, texW, nDepths).
    """
    usaf = metadata.get("usaf_data", {})
    target_depth = float(usaf.get("target_depth", 0.0))
    background = float(usaf.get("background", 0.0))

    tex_size = Resolution.get("_tex_size")
    if tex_size is None:
        raise ValueError("Resolution must contain '_tex_size' key. "
                         "Call generate_synthetic_dataset instead.")

    texH, texW = int(tex_size[0]), int(tex_size[1])
    tex_res = Resolution["texRes"]
    depths = list(Resolution["depths"])
    nd = len(depths)
    voxel_um = float(tex_res[0])

    depth_idx = int(np.argmin(np.abs(np.array(depths) - target_depth)))

    pattern = _build_usaf_pattern(texH, texW, voxel_um)

    volume = np.full((texH, texW, nd), background, dtype=np.float32)
    volume[:, :, depth_idx] = pattern
    return volume


# ═══════════════════════════════════════════════════════════════════════════════
# Poisson Noise
# ═══════════════════════════════════════════════════════════════════════════════

def add_poisson_noise(image: np.ndarray, scale: float,
                       rng: np.random.Generator = None) -> np.ndarray:
    """
    Add Poisson shot noise to a normalized image.

    Scales the image to photon counts, draws from a Poisson distribution,
    then normalizes back to [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input float image (values in approximately [0, 1]).
    scale : float
        Photon count scaling factor (higher → less noise).
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy image, float32, normalized to [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    counts = rng.poisson((image * scale).astype("float64")).astype("float32")
    max_counts = float(counts.max())
    if max_counts > 0:
        counts /= max_counts
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# Full Synthetic Dataset Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(metadata_path: str = "data/meta_data",
                                 output_data_path: str = "data/raw_data.npz",
                                 output_ref_dir: str = "evaluation/reference_outputs") -> None:
    """
    Generate the complete synthetic benchmark dataset.

    Steps
    -----
    1. Load metadata; build Camera parameter dict
    2. Set image size from synthetic parameters (n_lenslets × newSpacingPx)
    3. Compute geometry (synthetic mode — no white calibration image)
    4. Compute or load LF operators H, Ht (one-time expensive computation)
    5. Generate 3D fluorescent bead volume
    6. Forward project via H → clean 2D light field image
    7. Add Poisson noise → raw observed image
    8. Save data/raw_data.npz: {lf_image, ground_truth}
    9. Save reference outputs: operators_H.pkl, operators_Ht.pkl, ground_truth.npy, lf_image.npy

    Parameters
    ----------
    metadata_path : str
    output_data_path : str
    output_ref_dir : str
    """
    print("=" * 60)
    print("Light Field Microscope — USAF 1951 Dataset Generation")
    print("=" * 60)

    # Load metadata and camera params
    metadata = load_metadata(metadata_path)
    rec = metadata["reconstruction"]
    syn = metadata["synthetic_data"]

    new_spacing_px = rec["newSpacingPx"]
    depth_range = rec["depthRange"]
    depth_step = rec["depthStep"]
    super_res_factor = rec["superResFactor"]
    n_lenslets = syn["n_lenslets"]
    poisson_scale = syn["poisson_scale"]
    seed = syn.get("random_seed", 42)

    Camera = set_camera_params(metadata, new_spacing_px)
    print(f"\nCamera: M={Camera['M']}x, NA={Camera['NA']}, λ={Camera['WaveLength']} μm")

    # Image size: n_lenslets × newSpacingPx (odd if needed)
    img_side = int(n_lenslets * new_spacing_px)
    img_side = img_side + (1 - img_side % 2)
    img_size = np.array([img_side, img_side])
    print(f"Sensor image size: {img_size}")

    # Geometry
    print("\nComputing geometry (synthetic mode)...")
    LensletCenters, Resolution, _, _ = compute_geometry(
        Camera, np.array([]), depth_range, depth_step, super_res_factor, img_size)

    # Volume size
    tex_size = np.ceil(img_size * np.array(Resolution["texScaleFactor"])).astype("int32")
    tex_size = tex_size + (1 - tex_size % 2)
    nd = len(Resolution["depths"])
    print(f"Volume size: {tex_size[0]} × {tex_size[1]} × {nd}")
    print(f"Depth planes: {Resolution['depths']} μm")

    # Store tex_size in Resolution so generate_bead_volume can use it
    Resolution["_tex_size"] = tex_size

    # LF operators: load if cached, else compute
    ops_path_H = os.path.join(output_ref_dir, "operators_H.pkl")
    ops_path_Ht = os.path.join(output_ref_dir, "operators_Ht.pkl")
    os.makedirs(output_ref_dir, exist_ok=True)

    if os.path.exists(ops_path_H) and os.path.exists(ops_path_Ht):
        print("\nLoading precomputed LF operators...")
        with open(ops_path_H, "rb") as f:
            H = pickle.load(f)
        with open(ops_path_Ht, "rb") as f:
            Ht = pickle.load(f)
    else:
        print("\nComputing LF operators (this may take 10-60 minutes)...")
        H, Ht = compute_lf_operators(Camera, Resolution, LensletCenters)
        print("  Saving operators...")
        with open(ops_path_H, "wb") as f:
            pickle.dump(H, f, protocol=4)
        with open(ops_path_Ht, "wb") as f:
            pickle.dump(Ht, f, protocol=4)
        print(f"  Saved to {ops_path_H}")

    # Generate USAF resolution target volume
    print("\nGenerating USAF 1951 resolution target volume...")
    gt_volume = generate_usaf_volume(metadata, Resolution)
    print(f"  Volume range: [{gt_volume.min():.4f}, {gt_volume.max():.4f}]")
    target_depth = metadata.get("usaf_data", {}).get("target_depth", 0)
    print(f"  Pattern placed at depth plane: z ≈ {target_depth:.0f} μm")

    # Forward project → clean LF image
    print("\nForward projecting volume → LF image...")
    lf_clean = forward_project(H, gt_volume, LensletCenters, Resolution, img_size, Camera)
    print(f"  Clean image range: [{lf_clean.min():.4f}, {lf_clean.max():.4f}]")

    # Normalize clean image to [0, 1] before noise; clip small negatives from FFT
    lf_clean = np.clip(lf_clean, 0.0, None)
    lf_max = float(lf_clean.max())
    if lf_max > 0:
        lf_clean_norm = lf_clean / lf_max
    else:
        lf_clean_norm = lf_clean

    # Add Poisson noise
    print(f"\nAdding Poisson noise (scale={poisson_scale})...")
    rng2 = np.random.default_rng(seed + 1)
    lf_image = add_poisson_noise(lf_clean_norm, scale=poisson_scale, rng=rng2)
    print(f"  Noisy image range: [{lf_image.min():.4f}, {lf_image.max():.4f}]")

    # Save raw data
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    np.savez(output_data_path, lf_image=lf_image, ground_truth=gt_volume)
    print(f"\nSaved: {output_data_path}")

    # Save reference outputs
    np.save(os.path.join(output_ref_dir, "ground_truth.npy"), gt_volume)
    np.save(os.path.join(output_ref_dir, "lf_image.npy"), lf_image)
    print(f"Saved reference outputs to {output_ref_dir}/")

    print("\nDataset generation complete.")
    return lf_image, gt_volume, H, Ht, LensletCenters, Resolution, Camera


if __name__ == "__main__":
    generate_synthetic_dataset()
