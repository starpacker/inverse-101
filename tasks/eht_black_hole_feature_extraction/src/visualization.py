"""
Visualization for α-DPI — Corner Plots, ELBO, and Feature Metrics
===================================================================

Plotting utilities for posterior parameter distributions (corner plots),
ELBO model comparison, posterior image samples, and parameter recovery.

Reference
---------
Sun et al. (2022), ApJ 932:99 — Figure 7
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Feature Metrics ────────────────────────────────────────────────────────

def compute_feature_metrics(params_physical, ground_truth_params,
                             importance_weights=None,
                             param_names=None) -> dict:
    """
    Compute parameter recovery metrics.

    Parameters
    ----------
    params_physical : (N, n_params) ndarray — posterior parameter samples
    ground_truth_params : (n_params,) ndarray — true parameter values
    importance_weights : (N,) ndarray or None — importance weights
    param_names : list of str or None — parameter names

    Returns
    -------
    dict with keys:
        'bias'      : dict — mean bias per parameter
        'std'       : dict — posterior std per parameter
        'coverage'  : dict — fraction of 1σ coverage per parameter
        'n_params'  : int
    """
    n_params = ground_truth_params.shape[0]
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    if importance_weights is not None:
        w = importance_weights / importance_weights.sum()
        mean = np.sum(w[:, None] * params_physical[:, :n_params], axis=0)
        var = np.sum(w[:, None] * (params_physical[:, :n_params] - mean) ** 2, axis=0)
        std = np.sqrt(var)
    else:
        mean = np.mean(params_physical[:, :n_params], axis=0)
        std = np.std(params_physical[:, :n_params], axis=0)

    bias = {}
    std_dict = {}
    coverage = {}

    for i, name in enumerate(param_names):
        bias[name] = float(mean[i] - ground_truth_params[i])
        std_dict[name] = float(std[i])
        within = np.abs(params_physical[:, i] - ground_truth_params[i]) <= std[i]
        if importance_weights is not None:
            coverage[name] = float(np.sum(w[within]))
        else:
            coverage[name] = float(np.mean(within))

    return {
        'bias': bias,
        'std': std_dict,
        'coverage': coverage,
        'n_params': n_params,
    }


def print_feature_metrics(metrics: dict) -> None:
    """Pretty-print feature extraction metrics."""
    print(f"  {'Parameter':<20s} {'Bias':>10s} {'Std':>10s} {'1σ Cov':>10s}")
    print(f"  {'-' * 52}")
    for name in metrics['bias']:
        print(f"  {name:<20s} {metrics['bias'][name]:>10.3f} "
              f"{metrics['std'][name]:>10.3f} "
              f"{metrics['coverage'][name]:>10.3f}")


# ── Plotting Functions ──────────────────────────────────────────────────────

def plot_corner(params_physical, param_names=None, ground_truth=None,
                importance_weights=None, save_path=None):
    """
    Corner plot of posterior parameter distributions.

    Parameters
    ----------
    params_physical : (N, n_params) ndarray
    param_names : list of str or None
    ground_truth : (n_params,) ndarray or None — true values shown as lines
    importance_weights : (N,) ndarray or None
    save_path : str or None
    """
    if param_names is not None:
        n_params = min(params_physical.shape[1], len(param_names), 6)
        param_names = param_names[:n_params]
    else:
        n_params = min(params_physical.shape[1], 6)
        param_names = [f"param_{i}" for i in range(n_params)]

    fig, axes = plt.subplots(n_params, n_params, figsize=(2.5 * n_params, 2.5 * n_params))
    if n_params == 1:
        axes = np.array([[axes]])

    weights = importance_weights if importance_weights is not None else None

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                # Diagonal: 1D histogram
                ax.hist(params_physical[:, i], bins=50, density=True,
                        alpha=0.7, color='steelblue', weights=weights)
                if ground_truth is not None and i < len(ground_truth):
                    ax.axvline(ground_truth[i], color='red', linewidth=2,
                               linestyle='--')
            else:
                # Off-diagonal: 2D scatter
                if weights is not None:
                    # Subsample by importance weights for visualization
                    idx = np.random.choice(len(params_physical), size=min(5000, len(params_physical)),
                                           p=weights / weights.sum(), replace=True)
                else:
                    idx = np.random.choice(len(params_physical), size=min(5000, len(params_physical)),
                                           replace=False)
                ax.scatter(params_physical[idx, j], params_physical[idx, i],
                          s=1, alpha=0.3, color='steelblue')
                if ground_truth is not None:
                    if j < len(ground_truth) and i < len(ground_truth):
                        ax.axvline(ground_truth[j], color='red', linewidth=1,
                                   linestyle='--', alpha=0.7)
                        ax.axhline(ground_truth[i], color='red', linewidth=1,
                                   linestyle='--', alpha=0.7)

            if i == n_params - 1:
                ax.set_xlabel(param_names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(param_names[i])
            elif j > 0:
                ax.set_yticklabels([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_elbo_comparison(elbos, model_names=None, save_path=None):
    """
    ELBO comparison bar chart for model selection.

    Parameters
    ----------
    elbos : dict or list — {model_name: elbo_value} or list of values
    model_names : list of str or None
    save_path : str or None
    """
    if isinstance(elbos, dict):
        names = list(elbos.keys())
        values = list(elbos.values())
    else:
        values = list(elbos)
        names = model_names or [f"Model {i}" for i in range(len(values))]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    colors = ['steelblue'] * len(values)
    best_idx = np.argmax(values)
    colors[best_idx] = 'darkorange'

    ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("ELBO")
    ax.set_title("Model Selection via ELBO")
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_posterior_images(images, n_show=8, pixel_size_uas=None,
                          importance_weights=None, save_path=None):
    """
    Grid of posterior image samples from geometric model.

    Parameters
    ----------
    images : (K, npix, npix) ndarray
    n_show : int
    pixel_size_uas : float or None
    importance_weights : (K,) ndarray or None
    save_path : str or None
    """
    if importance_weights is not None:
        idx = np.random.choice(len(images), size=n_show,
                                p=importance_weights / importance_weights.sum(),
                                replace=False)
    else:
        idx = np.arange(min(n_show, len(images)))

    n_show = len(idx)
    n_cols = min(4, n_show)
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_show == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    vmax = np.percentile(images[idx], 99)
    for i, ax in enumerate(axes[:n_show]):
        im = images[idx[i]]
        N = im.shape[0]
        if pixel_size_uas is not None:
            extent_uas = N * pixel_size_uas / 2
            extent = [extent_uas, -extent_uas, -extent_uas, extent_uas]
        else:
            extent = None
        ax.imshow(im, origin='lower', cmap='afmhot', extent=extent,
                  vmin=0, vmax=vmax)
        ax.set_title(f"Sample {i + 1}")

    for ax in axes[n_show:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_loss_curves(loss_history, save_path=None):
    """
    Plot training loss components over epochs.

    Parameters
    ----------
    loss_history : dict
        Keys: 'total', 'cphase', 'logca', 'logdet'
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    epochs = np.arange(len(loss_history['total']))

    axes[0].plot(epochs, loss_history['total'])
    axes[0].set_title("Total Loss (ELBO)")
    axes[0].set_yscale('symlog')

    axes[1].plot(epochs, loss_history['cphase'], label='Closure Phase')
    axes[1].plot(epochs, loss_history['logca'], label='Log Closure Amp')
    axes[1].set_title("Data Fidelity")
    axes[1].legend()

    axes[2].plot(epochs, loss_history['logdet'])
    axes[2].set_title("-logdet / nparams")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
