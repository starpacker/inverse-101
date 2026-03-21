"""
Visualization Utilities for Closure-Only EHT Imaging
=====================================================

Provides plots for:
- uv-coverage
- Closure phases and amplitudes vs baseline
- Reconstruction comparisons
- Gain robustness demonstration (Figures 4-7 equivalent)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ═══════════════════════════════════════════════════════════════════════════
# Single-panel plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_uv_coverage(uv_coords: np.ndarray, title: str = "EHT uv-Coverage",
                     ax=None, figsize=(6, 6)):
    """
    Plot the (u,v)-plane sampling pattern.

    Parameters
    ----------
    uv_coords : (M, 2) array [wavelengths]
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    u, v = uv_coords[:, 0] / 1e9, uv_coords[:, 1] / 1e9
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


def plot_closure_phases(cphases: np.ndarray, tri_indices: np.ndarray = None,
                        title: str = "Closure Phases", figsize=(8, 4)):
    """
    Plot closure phases vs triangle index.

    Parameters
    ----------
    cphases : (N_tri,) closure phases in radians
    tri_indices : (N_tri,) optional x-axis indices
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = tri_indices if tri_indices is not None else np.arange(len(cphases))
    ax.scatter(x, np.degrees(cphases), s=6, alpha=0.6, c="steelblue")
    ax.set_xlabel("Triangle index")
    ax.set_ylabel("Closure Phase (°)")
    ax.set_title(title)
    ax.set_ylim(-185, 185)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    plt.tight_layout()
    return fig


def plot_closure_amplitudes(log_camps: np.ndarray, quad_indices: np.ndarray = None,
                            title: str = "Log Closure Amplitudes", figsize=(8, 4)):
    """
    Plot log closure amplitudes vs quadrangle index.

    Parameters
    ----------
    log_camps : (N_quad,) log closure amplitudes
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = quad_indices if quad_indices is not None else np.arange(len(log_camps))
    ax.scatter(x, log_camps, s=6, alpha=0.6, c="salmon")
    ax.set_xlabel("Quadrangle index")
    ax.set_ylabel("log CA")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.5, ls="--")
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
    metrics         : {'Method Name': {'nrmse': ..., 'ncc': ...}, ...}
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

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, panels, titles):
        plot_image(img, title=title, ax=ax, cmap=cmap,
                   pixel_size_uas=pixel_size_uas, vmin=0, vmax=img.max())

    plt.tight_layout()
    return fig


def plot_gain_robustness(
    results_by_gain: dict,
    ground_truth: np.ndarray,
    pixel_size_uas: float = None,
    figsize=None,
    cmap: str = "afmhot",
):
    """
    Plot reconstruction quality vs gain error level (Figure 5 equivalent).

    Shows that closure-only imaging is robust to gain errors while
    traditional visibility imaging degrades.

    Parameters
    ----------
    results_by_gain : dict
        {gain_error: {'Closure': image, 'Visibility': image, ...}, ...}
    ground_truth : (N, N) reference image
    """
    gain_levels = sorted(results_by_gain.keys())
    methods = list(results_by_gain[gain_levels[0]].keys())

    n_rows = len(gain_levels)
    n_cols = len(methods) + 1  # +1 for ground truth column

    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, gain in enumerate(gain_levels):
        # Ground truth
        plot_image(ground_truth, title=f"Truth (gain={gain:.0%})",
                   ax=axes[row, 0], cmap=cmap,
                   pixel_size_uas=pixel_size_uas, vmin=0)
        for col, method in enumerate(methods):
            img = results_by_gain[gain][method]
            m = compute_metrics(img, ground_truth)
            plot_image(
                img,
                title=f"{method}\nNRMSE={m['nrmse']:.3f}",
                ax=axes[row, col + 1], cmap=cmap,
                pixel_size_uas=pixel_size_uas, vmin=0,
            )

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute standard image-quality metrics.

    Metrics are computed after normalising the estimate to match the total
    flux of the ground truth.

    Parameters
    ----------
    estimate     : (N, N) reconstructed image
    ground_truth : (N, N) reference image

    Returns
    -------
    dict with keys: 'nrmse', 'ncc', 'dynamic_range'
    """
    est_sum = estimate.sum()
    if est_sum > 0:
        est = estimate * (ground_truth.sum() / est_sum)
    else:
        est = estimate.copy()

    nrmse = float(
        np.sqrt(np.mean((est - ground_truth) ** 2))
        / (np.sqrt(np.mean(ground_truth ** 2)) + 1e-30)
    )

    ncc = float(
        np.sum(est * ground_truth)
        / (np.sqrt(np.sum(est ** 2)) * np.sqrt(np.sum(ground_truth ** 2)) + 1e-30)
    )

    threshold = 0.02 * ground_truth.max()
    background = est[ground_truth < threshold]
    rms_bg = float(np.std(background)) if background.size > 0 else 1e-30
    dynamic_range = float(est.max() / (rms_bg + 1e-30))

    return {"nrmse": nrmse, "ncc": ncc, "dynamic_range": dynamic_range}


def print_metrics_table(metrics: dict):
    """Print a formatted table of reconstruction metrics."""
    header = f"{'Method':<30} {'NRMSE':>8} {'NCC':>8} {'Dyn. Range':>12}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(
            f"{name:<30} "
            f"{m['nrmse']:>8.4f} "
            f"{m['ncc']:>8.4f} "
            f"{m['dynamic_range']:>12.1f}"
        )
