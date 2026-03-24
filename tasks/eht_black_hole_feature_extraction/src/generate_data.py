"""
Synthetic Data Generation for α-DPI Feature Extraction
========================================================

Generates a synthetic black hole crescent + Gaussian image and simulated
EHT observation for testing the α-DPI feature extraction pipeline.

Reference
---------
Sun et al. (2022), ApJ 932:99
Original code: DPI/DPItorch/geometric_model.py (SimpleCrescentNuisance_Param2Img)
"""

import os
import json
import numpy as np


def generate_crescent_gaussian_image(
        npix: int = 64, fov_uas: float = 120.0,
        diameter_uas: float = 44.0, width_uas: float = 11.36,
        asymmetry: float = 0.5, pa_deg: float = -90.5,
        n_gaussian: int = 2, total_flux: float = 1.0) -> tuple:
    """
    Generate a synthetic crescent + elliptical Gaussians image.

    Parameters
    ----------
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.
    diameter_uas : float
        Ring diameter in microarcseconds.
    width_uas : float
        Ring width (Gaussian sigma) in microarcseconds.
    asymmetry : float
        Brightness asymmetry in [0, 1].
    pa_deg : float
        Position angle in degrees.
    n_gaussian : int
        Number of Gaussian components.
    total_flux : float
        Total flux normalization.

    Returns
    -------
    (image, params) : (ndarray, dict)
        image : (npix, npix) ndarray — crescent + Gaussian image
        params : dict — ground truth parameters
    """
    pixel_size = fov_uas / npix
    half_fov = fov_uas / 2

    gap = 1.0 / npix
    xs = np.arange(-1 + gap, 1, 2 * gap)
    grid_y, grid_x = np.meshgrid(-xs, xs, indexing='ij')
    grid_r = np.sqrt(grid_x ** 2 + grid_y ** 2)
    grid_theta = np.arctan2(grid_y, grid_x)

    # Crescent parameters in grid coordinates
    r = (diameter_uas / 2) / half_fov
    sigma = width_uas / half_fov
    eta = pa_deg * np.pi / 180

    # Gaussian ring with asymmetry
    ring = np.exp(-0.5 * (grid_r - r) ** 2 / sigma ** 2)
    S = 1 + asymmetry * np.cos(grid_theta - eta)
    image = S * ring

    # Normalize crescent
    if image.sum() > 0:
        image = image / image.sum()

    # Add Gaussian components
    rng = np.random.RandomState(42)
    gaussian_params = []
    for k in range(n_gaussian):
        x_shift = rng.uniform(-0.5, 0.5)
        y_shift = rng.uniform(-0.5, 0.5)
        sigma_gx = rng.uniform(0.05, 0.3)
        sigma_gy = rng.uniform(0.05, 0.3)
        scale = rng.uniform(0.05, 0.2)

        x_c = grid_x - x_shift
        y_c = grid_y - y_shift
        gaussian = np.exp(-0.5 * (x_c ** 2 / sigma_gx ** 2 + y_c ** 2 / sigma_gy ** 2))
        if gaussian.sum() > 0:
            gaussian = gaussian / gaussian.sum()
        image += scale * gaussian
        gaussian_params.append({
            'x_uas': x_shift * half_fov,
            'y_uas': y_shift * half_fov,
            'sigma_x_uas': sigma_gx * half_fov,
            'sigma_y_uas': sigma_gy * half_fov,
            'scale': scale,
        })

    # Final normalization
    if image.sum() > 0:
        image = image * total_flux / image.sum()

    params = {
        'diameter_uas': diameter_uas,
        'width_uas': width_uas,
        'asymmetry': asymmetry,
        'position_angle_deg': pa_deg,
        'n_gaussian': n_gaussian,
        'gaussians': gaussian_params,
    }

    return image, params


def generate_dataset(npix: int = 64, fov_uas: float = 120.0,
                      data_dir: str = "data") -> None:
    """
    Generate a complete synthetic dataset for α-DPI testing.

    Creates a crescent + Gaussians image and saves metadata.

    Parameters
    ----------
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.
    data_dir : str
        Output directory.
    """
    os.makedirs(data_dir, exist_ok=True)

    image, params = generate_crescent_gaussian_image(npix=npix, fov_uas=fov_uas)

    np.save(os.path.join(data_dir, "synthetic_gt.npy"), image)

    print(f"Synthetic crescent+Gaussian image saved to {data_dir}/synthetic_gt.npy")
    print(f"  Shape: {image.shape}, Total flux: {image.sum():.4f}")
    print(f"  Peak: {image.max():.6f}")
    print(f"  Ground truth params: {params}")
