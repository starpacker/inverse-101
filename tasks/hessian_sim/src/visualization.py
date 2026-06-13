"""
Visualization module: evaluation metrics and plotting.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
from scipy.ndimage import zoom as ndizoom


def _upsample_widefield(widefield, target_shape):
    """Upsample widefield to match SIM output size."""
    img_wf_up = ndizoom(widefield, 2, order=3)
    wf_sy, wf_sx = img_wf_up.shape
    sim_sy, sim_sx = target_shape
    if wf_sy != sim_sy or wf_sx != sim_sx:
        cy = (wf_sy - sim_sy) // 2
        cx = (wf_sx - sim_sx) // 2
        if cy >= 0 and cx >= 0:
            img_wf_up = img_wf_up[cy:cy + sim_sy, cx:cx + sim_sx]
        else:
            tmp = np.zeros((sim_sy, sim_sx))
            cy2 = (sim_sy - wf_sy) // 2
            cx2 = (sim_sx - wf_sx) // 2
            tmp[cy2:cy2 + wf_sy, cx2:cx2 + wf_sx] = img_wf_up
            img_wf_up = tmp
    return img_wf_up


def plot_comparison(widefield, sim_result, hessian_result, tv_result,
                    raw_frame=None, t_disp=None, save_path=None):
    """Plot comparison matching Supplementary Figure 8 layout.

    Row 1: Widefield | Raw noisy frame | Wiener-SIM (input y) | Hessian-SIM | TV-SIM
    Row 2: FFT spectra for each

    Parameters
    ----------
    widefield : ndarray (H, W)
    sim_result, hessian_result, tv_result : ndarray (N, 2H, 2W)
    raw_frame : ndarray (H, W) or None – single raw noisy frame for display
    t_disp : int or None
    save_path : str or None
    """
    if t_disp is None:
        t_disp = sim_result.shape[0] // 2

    get_slice = lambda d: np.squeeze(d)

    img_sim = get_slice(sim_result[t_disp])
    img_hess = get_slice(hessian_result[t_disp])
    img_tv = get_slice(tv_result[t_disp])

    img_wf_up = _upsample_widefield(get_slice(widefield), img_sim.shape)
    img_wf_norm = img_wf_up / (img_wf_up.max() + 1e-12) * img_sim.max()

    if raw_frame is not None:
        img_raw_up = _upsample_widefield(get_slice(raw_frame), img_sim.shape)
        img_raw_norm = img_raw_up / (img_raw_up.max() + 1e-12) * img_sim.max()
        images = [img_wf_norm, img_raw_norm, img_sim, img_hess, img_tv]
        titles = ['Widefield', 'Raw noisy frame', 'Wiener-SIM (input y)', 'Hessian-SIM', 'TV-SIM']
        ncols = 5
    else:
        images = [img_wf_norm, img_sim, img_hess, img_tv]
        titles = ['Widefield', 'Wiener-SIM (input y)', 'Hessian-SIM', 'TV-SIM']
        ncols = 4

    vmax_sim = np.percentile(img_sim[img_sim > 0], 99.5) if img_sim.max() > 0 else 1

    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 10))

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[0, i].imshow(img, cmap='hot', vmin=0, vmax=vmax_sim, interpolation='nearest')
        axes[0, i].set_title(title, fontsize=14, fontweight='bold')
        axes[0, i].axis('off')

    fts = [np.log1p(np.abs(fftshift(fft2(img)))) for img in images]
    ft_vmax = max(ft.max() for ft in fts)
    ft_vmin = min(ft.min() for ft in fts)
    for i, (ft, title) in enumerate(zip(fts, titles)):
        axes[1, i].imshow(ft, cmap='viridis', vmin=ft_vmin, vmax=ft_vmax, interpolation='nearest')
        axes[1, i].set_title(f'{title} - FFT', fontsize=12)
        axes[1, i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {save_path}")
    plt.close()
    return fig


def plot_line_profiles(sim_result, hessian_result, tv_result,
                       t_disp=None, save_path=None):
    """Plot horizontal and vertical line profiles through center."""
    if t_disp is None:
        t_disp = sim_result.shape[0] // 2

    get_slice = lambda d: np.squeeze(d)
    img_sim = get_slice(sim_result[t_disp])
    img_hess = get_slice(hessian_result[t_disp])
    img_tv = get_slice(tv_result[t_disp])

    sim_images = [img_sim, img_hess, img_tv]
    sim_titles = ['Wiener-SIM', 'Hessian-SIM', 'TV-SIM']
    sim_colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    row = img_sim.shape[0] // 2
    for img, title, c in zip(sim_images, sim_titles, sim_colors):
        axes[0].plot(img[row, :], label=title, alpha=0.8, color=c)
    axes[0].set_title('Horizontal Line Profile (center row)')
    axes[0].set_xlabel('Pixel')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()

    col = img_sim.shape[1] // 2
    for img, title, c in zip(sim_images, sim_titles, sim_colors):
        axes[1].plot(img[:, col], label=title, alpha=0.8, color=c)
    axes[1].set_title('Vertical Line Profile (center column)')
    axes[1].set_xlabel('Pixel')
    axes[1].set_ylabel('Intensity')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {save_path}")
    plt.close()
    return fig


def plot_hessian_vs_tv(hessian_result, tv_result, t_disp=None, save_path=None):
    """Plot Hessian vs TV zoomed comparison with difference map."""
    if t_disp is None:
        t_disp = hessian_result.shape[0] // 2

    get_slice = lambda d: np.squeeze(d)
    img_hess = get_slice(hessian_result[t_disp])
    img_tv = get_slice(tv_result[t_disp])

    vmax_sim = np.percentile(img_hess[img_hess > 0], 99.5) if img_hess.max() > 0 else 1

    cy, cx = img_hess.shape[0] // 2, img_hess.shape[1] // 2
    qs = min(cy, cx) // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    diff = img_hess - img_tv

    axes[0].imshow(img_hess[cy-qs:cy+qs, cx-qs:cx+qs], cmap='hot', vmin=0, vmax=vmax_sim)
    axes[0].set_title('Hessian-SIM (zoom)', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(img_tv[cy-qs:cy+qs, cx-qs:cx+qs], cmap='hot', vmin=0, vmax=vmax_sim)
    axes[1].set_title('TV-SIM (zoom)', fontweight='bold')
    axes[1].axis('off')

    diff_crop = diff[cy-qs:cy+qs, cx-qs:cx+qs]
    vlim = max(abs(diff_crop.min()), abs(diff_crop.max()))
    if vlim == 0:
        vlim = 1
    axes[2].imshow(diff_crop, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    axes[2].set_title('Hessian - TV (difference)', fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved {save_path}")
    plt.close()
    return fig
