"""
Data Preprocessing for EHT Black Hole Imaging
==============================================

Handles loading raw observation data and metadata, preparing inputs
for the physics model and solvers.

Pipeline: raw_data (NPZ) + meta_data (JSON) → calibrated visibilities + imaging parameters
"""

import os
import json
import numpy as np


def load_observation(data_dir: str = "data") -> dict:
    """
    Load raw observation data.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing the raw_data file.

    Returns
    -------
    dict with keys:
        'vis_noisy'  : (M,) complex ndarray — noisy complex visibilities
        'uv_coords'  : (M, 2) ndarray — baseline coordinates in wavelengths
    """
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path, allow_pickle=False)
    return {
        "vis_noisy": data["vis_noisy"],
        "uv_coords": data["uv_coords"],
    }


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load imaging metadata.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing the meta_data file.

    Returns
    -------
    dict with keys:
        'N'              : int   — image size (N x N pixels)
        'pixel_size_uas' : float — pixel size in microarcseconds
        'pixel_size_rad' : float — pixel size in radians
        'noise_std'      : float — noise standard deviation
        'freq_ghz'       : float — observing frequency in GHz
        'source_dec_deg' : float — source declination in degrees
        'n_baselines'    : int   — number of measured baselines
    """
    path = os.path.join(data_dir, "meta_data")
    with open(path, "r") as f:
        return json.load(f)


def prepare_data(data_dir: str = "data") -> tuple:
    """
    Load and prepare all data needed for reconstruction.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    vis_noisy : (M,) complex ndarray
        Noisy complex visibilities.
    uv_coords : (M, 2) ndarray
        Baseline coordinates in wavelengths.
    metadata : dict
        Imaging parameters (N, pixel sizes, noise_std, etc.).
    """
    obs = load_observation(data_dir)
    metadata = load_metadata(data_dir)
    return obs["vis_noisy"], obs["uv_coords"], metadata
