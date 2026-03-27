"""
Visualization — Light Field Microscope Reconstruction Plots
===========================================================

Plotting utilities for LFM data and reconstruction results.
All functions return matplotlib Figure objects for display in notebooks
or saving to disk in main.py.

Note: never call matplotlib.use() here — the backend is set only in main.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ═══════════════════════════════════════════════════════════════════════════════
# Raw Sensor Image
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lf_image(lf_image: np.ndarray,
                  title: str = "Light Field Image") -> plt.Figure:
    """
    Display the raw 2D sensor image (light field measurement).

    Parameters
    ----------
    lf_image : np.ndarray
        2D float array of shape (imgH, imgW).
    title : str

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(lf_image, cmap="inferno", interpolation="nearest")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Sensor X (px)")
    ax.set_ylabel("Sensor Y (px)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Volume Depth Slices
# ═══════════════════════════════════════════════════════════════════════════════

def plot_volume_slices(volume: np.ndarray, depths: np.ndarray,
                       title: str = "", vmax: float = None) -> plt.Figure:
    """
    Show XY lateral slices of a 3D volume at each depth plane.

    Parameters
    ----------
    volume : np.ndarray
        Shape (texH, texW, nDepths).
    depths : np.ndarray
        Depth values in μm, length nDepths.
    title : str
    vmax : float or None
        Colormap upper limit; defaults to volume.max().

    Returns
    -------
    plt.Figure
    """
    nd = volume.shape[2]
    ncols = min(nd, 5)
    nrows = int(np.ceil(nd / ncols))

    if vmax is None:
        vmax = float(volume.max())

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nd == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for i in range(nd):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.imshow(volume[:, :, i], cmap="hot", vmin=0, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(f"Δz = {depths[i]:.0f} μm", fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for i in range(nd, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Maximum Intensity Projections
# ═══════════════════════════════════════════════════════════════════════════════

def plot_volume_mip(volume: np.ndarray, title: str = "") -> plt.Figure:
    """
    Show maximum-intensity projections: XY (top), XZ (side), YZ (front).

    Parameters
    ----------
    volume : np.ndarray
        Shape (texH, texW, nDepths).
    title : str

    Returns
    -------
    plt.Figure
    """
    mip_xy = volume.max(axis=2)
    mip_xz = volume.max(axis=0)
    mip_yz = volume.max(axis=1)

    vmax = float(volume.max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mip_xy, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
    axes[0].set_title("XY (max-z)")
    axes[0].set_xlabel("X (vox)")
    axes[0].set_ylabel("Y (vox)")

    axes[1].imshow(mip_xz, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")
    axes[1].set_title("XZ (max-y)")
    axes[1].set_xlabel("X (vox)")
    axes[1].set_ylabel("Depth plane")

    axes[2].imshow(mip_yz, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")
    axes[2].set_title("YZ (max-x)")
    axes[2].set_xlabel("Y (vox)")
    axes[2].set_ylabel("Depth plane")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Reconstruction Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_reconstruction_comparison(gt: np.ndarray,
                                    rl_vol: np.ndarray,
                                    ems_vol: np.ndarray,
                                    depths: np.ndarray,
                                    metrics: dict = None) -> plt.Figure:
    """
    Side-by-side comparison: Ground Truth | RL (artifacts) | EMS (clean).

    One row per depth plane, three columns.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth volume, shape (texH, texW, nDepths).
    rl_vol : np.ndarray
        Standard RL reconstruction (with artifacts).
    ems_vol : np.ndarray
        EMS reconstruction (artifact-free).
    depths : np.ndarray
        Depth values in μm.
    metrics : dict or None
        Optional dict with keys 'rl' and 'ems', each having 'nrmse' and 'psnr'.

    Returns
    -------
    plt.Figure
    """
    nd = gt.shape[2]
    vmax = float(gt.max())

    fig = plt.figure(figsize=(9, 2.5 * nd))
    gs = gridspec.GridSpec(nd, 3, figure=fig, hspace=0.4, wspace=0.15)

    col_titles = ["Ground Truth", "RL (artifacts)", "EMS (artifact-free)"]
    cols = [gt, rl_vol, ems_vol]

    for row in range(nd):
        for col_idx, (col_title, vol) in enumerate(zip(col_titles, cols)):
            ax = fig.add_subplot(gs[row, col_idx])
            ax.imshow(vol[:, :, row], cmap="hot", vmin=0, vmax=vmax,
                      interpolation="nearest")
            if row == 0:
                ax.set_title(col_title, fontsize=11, fontweight="bold")
            ax.set_ylabel(f"Δz={depths[row]:.0f}μm", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    # Metrics annotation
    if metrics is not None:
        rl_m = metrics.get("rl", {})
        ems_m = metrics.get("ems", {})
        note = (f"RL:  NRMSE={rl_m.get('nrmse', float('nan')):.4f},  "
                f"PSNR={rl_m.get('psnr', float('nan')):.1f} dB\n"
                f"EMS: NRMSE={ems_m.get('nrmse', float('nan')):.4f},  "
                f"PSNR={ems_m.get('psnr', float('nan')):.1f} dB")
        fig.text(0.5, 0.02, note, ha="center", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("LFM 3D Reconstruction: RL vs EMS", fontsize=13, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# USAF Target Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_usaf_comparison(gt: np.ndarray,
                          rl_vol: np.ndarray,
                          ems_vol: np.ndarray,
                          depth_idx: int,
                          lf_image: np.ndarray = None,
                          voxel_um: float = None,
                          metrics: dict = None) -> plt.Figure:
    """
    Comparison figure for a USAF 1951 target: LF sensor image + reconstruction.

    Four-panel layout:
      [LF sensor image] | [Ground Truth z-slice] | [RL z-slice] | [EMS z-slice]

    RL and EMS are each normalised to their own maximum so that structure is
    visible even when artifacts push the absolute scale far above the GT.
    Colorbars annotate the actual max value for each panel.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth volume, shape (texH, texW, nDepths).
    rl_vol : np.ndarray
        Standard RL reconstruction (with grid artifacts).
    ems_vol : np.ndarray
        EMS reconstruction (artifact-free).
    depth_idx : int
        Depth plane index to display.
    lf_image : np.ndarray or None
        Raw 2D sensor image (imgH, imgW); shown in the leftmost panel if given.
    voxel_um : float or None
        Voxel size in μm; used to compute spatial frequency labels.
    metrics : dict or None
        Optional dict with keys 'rl' and 'ems', each having 'nrmse' and 'psnr'.

    Returns
    -------
    plt.Figure
    """
    gt_sl  = gt[:, :, depth_idx]
    rl_sl  = rl_vol[:, :, depth_idx]
    ems_sl = ems_vol[:, :, depth_idx]

    ncols = 4 if lf_image is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.8))

    col_idx = 0

    # --- Panel 0: LF sensor image ---
    if lf_image is not None:
        ax = axes[col_idx]
        im = ax.imshow(lf_image, cmap="inferno", interpolation="nearest")
        ax.set_title("LF Sensor Image\n(7×7 lenslet mosaic)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Sensor X (px)", fontsize=9)
        ax.set_ylabel("Sensor Y (px)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        col_idx += 1

    # --- Panel 1: Ground truth ---
    ax = axes[col_idx]
    im = ax.imshow(gt_sl, cmap="hot", vmin=0, vmax=1.0, interpolation="nearest")
    ax.set_title("Ground Truth (z = 0 μm)", fontsize=11, fontweight="bold")
    lbl = f"1 px = {voxel_um:.1f} μm" if voxel_um else "voxels"
    ax.set_xlabel(f"X  ({lbl})", fontsize=9)
    ax.set_ylabel(f"Y  ({lbl})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bar-group frequency annotations (recomputed from voxel_um)
    if voxel_um is not None:
        freqs = [f"G1\n{1e3/(2*4*voxel_um):.0f}\nlp/mm",
                 f"G2\n{1e3/(2*2*voxel_um):.0f}\nlp/mm",
                 f"G3\n{1e3/(2*1*voxel_um):.0f}\nlp/mm"]
    else:
        freqs = ["G1", "G2", "G3"]
    for x_pos, label in zip([2, 22, 31], freqs):
        ax.text(x_pos, 0.5, label, color="cyan", fontsize=7, va="top", ha="left")
    col_idx += 1

    # --- Panel 2: RL ---
    ax = axes[col_idx]
    rl_max = float(rl_sl.max()) if rl_sl.max() > 0 else 1.0
    im = ax.imshow(rl_sl / rl_max, cmap="hot", vmin=0, vmax=1.0, interpolation="nearest")
    ax.set_title(f"RL (grid artifacts)\nmax = {rl_max:.2f}×GT", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"X  ({lbl})", fontsize=9)
    ax.set_ylabel(f"Y  ({lbl})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    col_idx += 1

    # --- Panel 3: EMS ---
    ax = axes[col_idx]
    ems_max = float(ems_sl.max()) if ems_sl.max() > 0 else 1.0
    im = ax.imshow(ems_sl / ems_max, cmap="hot", vmin=0, vmax=1.0, interpolation="nearest")
    ax.set_title(f"EMS (artifact-free)\nmax = {ems_max:.2f}×GT", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"X  ({lbl})", fontsize=9)
    ax.set_ylabel(f"Y  ({lbl})", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if metrics is not None:
        rl_m  = metrics.get("rl", {})
        ems_m = metrics.get("ems", {})
        note = (f"RL:   NRMSE={rl_m.get('nrmse', float('nan')):.4f},  "
                f"PSNR={rl_m.get('psnr', float('nan')):.1f} dB  |  "
                f"EMS:  NRMSE={ems_m.get('nrmse', float('nan')):.4f},  "
                f"PSNR={ems_m.get('psnr', float('nan')):.1f} dB")
        fig.text(0.5, -0.02, note, ha="center", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("USAF 1951 Resolution Target — Light Field Microscopy (z = 0 μm)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(estimate: np.ndarray,
                    ground_truth: np.ndarray) -> dict:
    """
    Compute NRMSE and PSNR of a 3D reconstruction against the ground truth.

    Definitions
    -----------
    NRMSE = ||estimate - gt||_F / ||gt||_F
    PSNR  = 20 · log10( max(gt) / RMSE )

    Parameters
    ----------
    estimate : np.ndarray
        Reconstructed volume.
    ground_truth : np.ndarray
        Reference volume (same shape).

    Returns
    -------
    dict
        {'nrmse': float, 'psnr': float}
    """
    diff = estimate.astype(float) - ground_truth.astype(float)
    rmse = float(np.sqrt(np.mean(diff**2)))
    gt_norm = float(np.linalg.norm(ground_truth))
    nrmse = float(np.linalg.norm(diff)) / gt_norm if gt_norm > 0 else float("inf")
    gt_max = float(ground_truth.max())
    psnr = float(20 * np.log10(gt_max / rmse)) if (gt_max > 0 and rmse > 0) else float("inf")
    return {"nrmse": nrmse, "psnr": psnr}


def print_metrics_table(metrics: dict) -> None:
    """
    Print a formatted metrics table to stdout.

    Parameters
    ----------
    metrics : dict
        Keys are method names (e.g. 'rl', 'ems'); values are dicts with
        'nrmse' and 'psnr' keys.
    """
    header = f"{'Method':<12}  {'NRMSE':>8}  {'PSNR (dB)':>10}"
    print(header)
    print("-" * len(header))
    for method, m in metrics.items():
        nrmse = m.get("nrmse", float("nan"))
        psnr = m.get("psnr", float("nan"))
        print(f"{method:<12}  {nrmse:>8.4f}  {psnr:>10.2f}")
