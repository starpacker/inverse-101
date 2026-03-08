"""
Visualization Utilities for EHT Black Hole Imaging
====================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ═══════════════════════════════════════════════════════════════════════════
# Single-panel plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_uv_coverage(uv_coords: np.ndarray, title: str = "EHT uv-Coverage",
                     ax=None, figsize=(6, 6)):
    """
    Plot the (u,v)-plane sampling pattern.

    Each measured baseline (u,v) and its conjugate (−u,−v) are shown.
    The density and spread of points determine the achievable resolution
    and imaging fidelity.

    Parameters
    ----------
    uv_coords : (M, 2) array  [wavelengths]
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    u, v = uv_coords[:, 0] / 1e9, uv_coords[:, 1] / 1e9   # → Gλ

    ax.scatter(u, v, s=4, c="steelblue", alpha=0.7, label="measured")
    ax.scatter(-u, -v, s=4, c="salmon", alpha=0.7, label="conjugate")
    ax.set_xlabel("u  (Gλ)")
    ax.set_ylabel("v  (Gλ)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.axvline(0, color="k", lw=0.4, ls="--")
    ax.legend(fontsize=8, markerscale=2)
    return ax


def plot_image(image: np.ndarray, title: str = "", ax=None,
               cmap: str = "afmhot", pixel_size_uas: float = None,
               vmin=None, vmax=None, figsize=(4.5, 4.5)):
    """
    Display a 2D image with optional physical axis labels.

    Parameters
    ----------
    pixel_size_uas : float or None
        Pixel size in microarcseconds. If given, axes are labelled in μas.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    N = image.shape[0]
    if pixel_size_uas is not None:
        hw = (N / 2) * pixel_size_uas
        extent = [-hw, hw, -hw, hw]
        xlabel = ylabel = "μas"
    else:
        extent = None
        xlabel = ylabel = "pixels"

    im = ax.imshow(image, cmap=cmap, origin="lower",
                   extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_visibilities(vis: np.ndarray, uv_coords: np.ndarray,
                      title: str = "Visibilities", figsize=(11, 4)):
    """
    Plot visibility amplitude and phase vs. baseline length.

    Parameters
    ----------
    vis       : (M,) complex visibilities
    uv_coords : (M, 2) [wavelengths]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    bl = np.sqrt(uv_coords[:, 0] ** 2 + uv_coords[:, 1] ** 2) / 1e9  # Gλ
    amp = np.abs(vis)
    phase = np.angle(vis, deg=True)

    ax1.scatter(bl, amp, s=6, alpha=0.6, c="steelblue")
    ax1.set_xlabel("Baseline length  (Gλ)")
    ax1.set_ylabel("Amplitude  (Jy)")
    ax1.set_title(f"{title} – Amplitude")

    ax2.scatter(bl, phase, s=6, alpha=0.6, c="salmon")
    ax2.set_xlabel("Baseline length  (Gλ)")
    ax2.set_ylabel("Phase  (°)")
    ax2.set_title(f"{title} – Phase")
    ax2.set_ylim(-185, 185)
    ax2.axhline(0, color="k", lw=0.5, ls="--")

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Multi-panel comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison(
    reconstructions: dict,
    ground_truth: np.ndarray = None,
    pixel_size_uas: float = None,
    metrics: dict = None,
    figsize=None,
    cmap: str = "afmhot",
):
    """
    Side-by-side comparison of multiple reconstruction methods.

    Parameters
    ----------
    reconstructions : {'Method Name': image_ndarray, ...}
    ground_truth    : reference image (optional)
    pixel_size_uas  : pixel size in μas for axis labels
    metrics         : {'Method Name': {'nrmse': 0.12, 'ncc': 0.97, ...}, ...}
    """
    panels = []
    titles = []

    if ground_truth is not None:
        panels.append(ground_truth)
        titles.append("Ground Truth")

    for name, img in reconstructions.items():
        panels.append(img)
        if metrics and name in metrics:
            m = metrics[name]
            title = f"{name}\nNRMSE={m.get('nrmse', 0):.3f}  NCC={m.get('ncc', 0):.3f}"
        else:
            title = name
        titles.append(title)

    n = len(panels)
    if figsize is None:
        figsize = (4.2 * n, 4.2)

    vmax = max(p.max() for p in panels)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, panels, titles):
        plot_image(img, title=title, ax=ax, cmap=cmap,
                   pixel_size_uas=pixel_size_uas, vmin=0, vmax=vmax)

    plt.tight_layout()
    return fig


def plot_summary_panel(
    model,
    vis_noisy: np.ndarray,
    reconstructions: dict,
    ground_truth: np.ndarray = None,
    pixel_size_uas: float = None,
    metrics: dict = None,
    figsize=(16, 8),
):
    """
    Full summary: uv-coverage | dirty image | reconstruction methods.

    Parameters
    ----------
    model          : VLBIForwardModel
    vis_noisy      : (M,) complex visibilities
    reconstructions: {'Method': image, ...}
    """
    n_methods = len(reconstructions)
    n_right = n_methods + (1 if ground_truth is not None else 0)
    n_cols = 2 + n_right      # uv + dirty + methods

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.35)

    # ── uv-coverage ───────────────────────────────────────────────────────
    ax_uv = fig.add_subplot(gs[0, 0])
    plot_uv_coverage(model.uv, ax=ax_uv, title="uv-Coverage")

    # ── Dirty image ───────────────────────────────────────────────────────
    dirty = model.dirty_image(vis_noisy)
    ax_dirty = fig.add_subplot(gs[0, 1])
    vmax = max(dirty.max(), *(r.max() for r in reconstructions.values()),
               ground_truth.max() if ground_truth is not None else 0)
    plot_image(dirty, title="Dirty Image", ax=ax_dirty, vmin=0, vmax=vmax,
               pixel_size_uas=pixel_size_uas)

    # ── Ground truth and reconstructions ──────────────────────────────────
    col = 2
    if ground_truth is not None:
        ax = fig.add_subplot(gs[0, col])
        plot_image(ground_truth, title="Ground Truth", ax=ax,
                   vmin=0, vmax=vmax, pixel_size_uas=pixel_size_uas)
        col += 1

    for name, img in reconstructions.items():
        ax = fig.add_subplot(gs[0, col])
        if metrics and name in metrics:
            m = metrics[name]
            title = f"{name}\nNRMSE={m.get('nrmse', 0):.3f}"
        else:
            title = name
        plot_image(img, title=title, ax=ax, vmin=0, vmax=vmax,
                   pixel_size_uas=pixel_size_uas)
        col += 1

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute standard image-quality metrics.

    Metrics are computed after normalising the estimate to match the total
    flux of the ground truth (removes overall scale ambiguity).

    Parameters
    ----------
    estimate     : (N, N) reconstructed image
    ground_truth : (N, N) reference image

    Returns
    -------
    dict with keys: 'nrmse', 'ncc', 'dynamic_range'
    """
    # Flux normalisation
    est_sum = estimate.sum()
    if est_sum > 0:
        est = estimate * (ground_truth.sum() / est_sum)
    else:
        est = estimate.copy()

    # NRMSE  (lower is better)
    nrmse = float(
        np.sqrt(np.mean((est - ground_truth) ** 2))
        / (np.sqrt(np.mean(ground_truth ** 2)) + 1e-30)
    )

    # Normalised cross-correlation  (higher is better, max 1)
    ncc = float(
        np.sum(est * ground_truth)
        / (np.sqrt(np.sum(est ** 2)) * np.sqrt(np.sum(ground_truth ** 2)) + 1e-30)
    )

    # Dynamic range: peak / RMS of off-source background
    threshold = 0.02 * ground_truth.max()
    background = est[ground_truth < threshold]
    rms_bg = float(np.std(background)) if background.size > 0 else 1e-30
    dynamic_range = float(est.max() / (rms_bg + 1e-30))

    return {"nrmse": nrmse, "ncc": ncc, "dynamic_range": dynamic_range}


def print_metrics_table(metrics: dict):
    """Print a formatted table of reconstruction metrics."""
    header = f"{'Method':<20} {'NRMSE':>8} {'NCC':>8} {'Dyn. Range':>12}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(
            f"{name:<20} "
            f"{m['nrmse']:>8.4f} "
            f"{m['ncc']:>8.4f} "
            f"{m['dynamic_range']:>12.1f}"
        )
