"""
Visualization and metrics for BH-NeRF task.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_metrics(estimate, ground_truth):
    """
    Compute 3D emission recovery metrics.

    Parameters
    ----------
    estimate : np.ndarray
        Estimated emission (arbitrary shape).
    ground_truth : np.ndarray
        Ground truth emission (same shape).

    Returns
    -------
    metrics : dict with keys 'nrmse', 'ncc', 'psnr'
    """
    est = estimate.ravel().astype(np.float64)
    gt = ground_truth.ravel().astype(np.float64)

    mse = np.mean((est - gt) ** 2)
    nrmse = np.sqrt(mse) / (np.max(gt) - np.min(gt) + 1e-30)

    est_centered = est - est.mean()
    gt_centered = gt - gt.mean()
    ncc = float(np.dot(est_centered, gt_centered) /
                (np.linalg.norm(est_centered) * np.linalg.norm(gt_centered) + 1e-30))

    psnr = float(10.0 * np.log10(np.max(gt) ** 2 / (mse + 1e-30)))

    return {
        'nrmse': float(nrmse),
        'ncc': float(ncc),
        'psnr': float(psnr),
    }


def compute_image_metrics(pred_movie, true_movie):
    """
    Compute image-plane metrics averaged over time frames.

    Parameters
    ----------
    pred_movie : np.ndarray, shape (n_frames, H, W)
    true_movie : np.ndarray, shape (n_frames, H, W)

    Returns
    -------
    metrics : dict with keys 'nrmse_image', 'ncc_image', 'lightcurve_mse'
    """
    nrmse_list = []
    ncc_list = []
    for i in range(len(pred_movie)):
        m = compute_metrics(pred_movie[i], true_movie[i])
        nrmse_list.append(m['nrmse'])
        ncc_list.append(m['ncc'])

    pred_lc = pred_movie.sum(axis=(-1, -2))
    true_lc = true_movie.sum(axis=(-1, -2))
    lc_mse = float(np.mean((pred_lc - true_lc) ** 2))

    return {
        'nrmse_image': float(np.mean(nrmse_list)),
        'ncc_image': float(np.mean(ncc_list)),
        'lightcurve_mse': lc_mse,
    }


def plot_emission_slices(emission_3d, fov_M, ground_truth=None,
                         save_path=None):
    """
    Plot x-y, x-z, y-z slices through the 3D emission volume.

    Parameters
    ----------
    emission_3d : np.ndarray, shape (D, H, W)
    fov_M : float
    ground_truth : np.ndarray, optional
    save_path : str, optional
    """
    mid = emission_3d.shape[0] // 2
    extent = [-fov_M / 2, fov_M / 2, -fov_M / 2, fov_M / 2]

    if ground_truth is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        titles = ['x-y (GT)', 'x-z (GT)', 'y-z (GT)',
                  'x-y (Est)', 'x-z (Est)', 'y-z (Est)']
        slices_gt = [ground_truth[:, :, mid], ground_truth[:, mid, :],
                     ground_truth[mid, :, :]]
        slices_est = [emission_3d[:, :, mid], emission_3d[:, mid, :],
                      emission_3d[mid, :, :]]
        for i in range(3):
            axes[0, i].imshow(slices_gt[i].T, origin='lower', extent=extent)
            axes[0, i].set_title(titles[i])
            axes[1, i].imshow(slices_est[i].T, origin='lower', extent=extent)
            axes[1, i].set_title(titles[i + 3])
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['x-y slice', 'x-z slice', 'y-z slice']
        slices = [emission_3d[:, :, mid], emission_3d[:, mid, :],
                  emission_3d[mid, :, :]]
        for i in range(3):
            axes[i].imshow(slices[i].T, origin='lower', extent=extent)
            axes[i].set_title(titles[i])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_movie_comparison(pred_movie, true_movie, t_frames, n_show=6,
                          save_path=None):
    """
    Show selected time frames: true vs predicted images.

    Parameters
    ----------
    pred_movie : np.ndarray, shape (n_frames, H, W)
    true_movie : np.ndarray, shape (n_frames, H, W)
    t_frames : np.ndarray, shape (n_frames,)
    n_show : int
    save_path : str, optional
    """
    indices = np.linspace(0, len(t_frames) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    vmax = max(true_movie.max(), pred_movie.max())

    for j, idx in enumerate(indices):
        axes[0, j].imshow(true_movie[idx].T, origin='lower', vmin=0,
                          vmax=vmax, cmap='hot')
        axes[0, j].set_title(f't={t_frames[idx]:.0f}M (true)')
        axes[0, j].axis('off')

        axes[1, j].imshow(pred_movie[idx].T, origin='lower', vmin=0,
                          vmax=vmax, cmap='hot')
        axes[1, j].set_title(f't={t_frames[idx]:.0f}M (pred)')
        axes[1, j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_lightcurve(pred_images, true_images, t_frames, save_path=None):
    """
    Plot total flux (lightcurve) over time.

    Parameters
    ----------
    pred_images : np.ndarray, shape (n_frames, H, W)
    true_images : np.ndarray, shape (n_frames, H, W)
    t_frames : np.ndarray
    save_path : str, optional
    """
    pred_lc = pred_images.sum(axis=(-1, -2))
    true_lc = true_images.sum(axis=(-1, -2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_frames, true_lc, 'b-', label='True')
    ax.plot(t_frames, pred_lc, 'rx-', label='Predicted')
    ax.set_xlabel('Time [M]')
    ax.set_ylabel('Total flux')
    ax.set_title('Lightcurve')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_loss_curves(loss_history, save_path=None):
    """
    Plot training loss in log scale.

    Parameters
    ----------
    loss_history : list or np.ndarray
    save_path : str, optional
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_metrics_table(metrics):
    """
    Print metrics in a formatted table.

    Parameters
    ----------
    metrics : dict
    """
    print("\n" + "=" * 50)
    print(f"{'Metric':<25} {'Value':>15}")
    print("=" * 50)
    for key, val in metrics.items():
        print(f"{key:<25} {val:>15.6f}")
    print("=" * 50)
