"""
Data Preprocessing for Closure-Only EHT Imaging
=================================================

Handles loading raw observation data, computing closure phases and closure
amplitudes from complex visibilities, and estimating their noise statistics.

Pipeline: raw_data (NPZ) + meta_data (JSON) → closure quantities + imaging parameters

Reference
---------
Chael et al. (2018), ApJ 857, 23 — Eqs. 9–12.
"""

import os
import json
import numpy as np
from itertools import combinations


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
        'vis_corrupted' : (M,) complex ndarray — gain-corrupted visibilities
        'vis_true'      : (M,) complex ndarray — true (uncorrupted) visibilities
        'uv_coords'     : (M, 2) ndarray — baseline coordinates in wavelengths
        'station_ids'   : (M, 2) int ndarray — station pair for each baseline
        'noise_std_per_vis' : (M,) ndarray — per-visibility noise std
    """
    path = os.path.join(data_dir, "raw_data.npz")
    data = np.load(path, allow_pickle=False)
    return {
        "vis_corrupted": data["vis_corrupted"],
        "vis_true": data["vis_true"],
        "uv_coords": data["uv_coords"],
        "station_ids": data["station_ids"],
        "noise_std_per_vis": data["noise_std_per_vis"],
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
        'freq_ghz'       : float — observing frequency in GHz
        'source_dec_deg' : float — source declination in degrees
        'obs_duration_hours' : float
        'n_baselines'    : int   — number of measured baselines
        'n_stations'     : int   — number of stations
        'gain_amp_error' : float — fractional amplitude gain error applied
        'gain_phase_error_deg' : float — phase gain error (degrees) applied
    """
    path = os.path.join(data_dir, "meta_data")
    with open(path, "r") as f:
        return json.load(f)


def find_triangles(station_ids: np.ndarray, n_stations: int) -> np.ndarray:
    """
    Find all independent closure phase triangles from baseline station pairs.

    Parameters
    ----------
    station_ids : (M, 2) int ndarray
        Station pair indices for each baseline.
    n_stations : int
        Total number of stations.

    Returns
    -------
    triangles : (N_tri, 3, 2) int ndarray
        Each row is [baseline_idx, baseline_idx, baseline_idx] forming a triangle.
        Actually returns (N_tri, 3) where each entry is the baseline index.
    triangle_stations : (N_tri, 3) int ndarray
        Station indices forming each triangle (i, j, k).
    """
    # Build lookup: (s1, s2) → baseline index
    bl_lookup = {}
    for idx, (s1, s2) in enumerate(station_ids):
        bl_lookup[(int(s1), int(s2))] = idx
        bl_lookup[(int(s2), int(s1))] = -(idx + 1)  # negative = conjugate

    stations = sorted(set(station_ids.ravel()))
    triangles = []
    triangle_stations_list = []

    for i, j, k in combinations(stations, 3):
        # Need baselines (i,j), (j,k), (k,i)
        key_ij = (i, j) if (i, j) in bl_lookup else None
        key_jk = (j, k) if (j, k) in bl_lookup else None
        key_ki = (k, i) if (k, i) in bl_lookup else None

        if key_ij is None:
            key_ij = (j, i) if (j, i) in bl_lookup else None
        if key_jk is None:
            key_jk = (k, j) if (k, j) in bl_lookup else None
        if key_ki is None:
            key_ki = (i, k) if (i, k) in bl_lookup else None

        if key_ij is not None and key_jk is not None and key_ki is not None:
            triangles.append([bl_lookup[key_ij], bl_lookup[key_jk], bl_lookup[key_ki]])
            triangle_stations_list.append([i, j, k])

    return np.array(triangles, dtype=np.int64), np.array(triangle_stations_list, dtype=np.int64)


def find_quadrangles(station_ids: np.ndarray, n_stations: int) -> np.ndarray:
    """
    Find all independent closure amplitude quadrangles.

    For stations (i, j, k, l), the closure amplitude is:
        CA = |V_ij * V_kl| / |V_ik * V_jl|

    Parameters
    ----------
    station_ids : (M, 2) int ndarray
    n_stations : int

    Returns
    -------
    quadrangles : (N_quad, 4) int ndarray
        Each row is [bl_ij, bl_kl, bl_ik, bl_jl] — first two numerator, last two denominator.
    quadrangle_stations : (N_quad, 4) int ndarray
        Station indices (i, j, k, l).
    """
    bl_lookup = {}
    for idx, (s1, s2) in enumerate(station_ids):
        bl_lookup[(int(s1), int(s2))] = idx
        bl_lookup[(int(s2), int(s1))] = idx  # amplitude is symmetric

    stations = sorted(set(station_ids.ravel()))
    quadrangles = []
    quadrangle_stations_list = []

    for i, j, k, l in combinations(stations, 4):
        # Numerator: (i,j) and (k,l); Denominator: (i,k) and (j,l)
        def get_bl(a, b):
            return bl_lookup.get((a, b), bl_lookup.get((b, a), None))

        bl_ij = get_bl(i, j)
        bl_kl = get_bl(k, l)
        bl_ik = get_bl(i, k)
        bl_jl = get_bl(j, l)

        if all(x is not None for x in [bl_ij, bl_kl, bl_ik, bl_jl]):
            quadrangles.append([bl_ij, bl_kl, bl_ik, bl_jl])
            quadrangle_stations_list.append([i, j, k, l])

    return np.array(quadrangles, dtype=np.int64), np.array(quadrangle_stations_list, dtype=np.int64)


def compute_closure_phases(vis: np.ndarray, triangles: np.ndarray,
                           station_ids: np.ndarray) -> np.ndarray:
    """
    Compute closure phases from complex visibilities on triangles.

    The bispectrum for triangle (i,j,k) is:
        B_ijk = V_ij * V_jk * V_ki

    The closure phase is φ_ijk = arg(B_ijk).

    Station-based gains cancel: if V_ij^obs = g_i * g_j^* * V_ij^true,
    then B^obs = |g_i|² |g_j|² |g_k|² * B^true, so
    arg(B^obs) = arg(B^true).

    Parameters
    ----------
    vis : (M,) complex ndarray
    triangles : (N_tri, 3) int ndarray — baseline indices per triangle
    station_ids : (M, 2) int ndarray

    Returns
    -------
    cphases : (N_tri,) ndarray — closure phases in radians
    """
    # Build baseline orientation lookup
    bl_orient = {}
    for idx, (s1, s2) in enumerate(station_ids):
        bl_orient[idx] = (int(s1), int(s2))

    cphases = np.zeros(len(triangles))
    for t, tri_bl in enumerate(triangles):
        # Compute bispectrum V_ij * V_jk * V_ki
        bispec = np.complex128(1.0)
        for bl_code in tri_bl:
            bl_idx = bl_code
            if bl_idx >= 0:
                bispec *= vis[bl_idx]
            else:
                bispec *= np.conj(vis[-(bl_idx + 1)])
        cphases[t] = np.angle(bispec)

    return cphases


def compute_closure_amplitudes(vis: np.ndarray, quadrangles: np.ndarray) -> np.ndarray:
    """
    Compute closure amplitudes from visibility amplitudes on quadrangles.

    For quadrangle (i,j,k,l):
        CA = |V_ij| * |V_kl| / (|V_ik| * |V_jl|)

    Station-based gain amplitudes cancel:
        CA^obs = CA^true (gain amplitudes fully cancel in the ratio).

    Parameters
    ----------
    vis : (M,) complex ndarray
    quadrangles : (N_quad, 4) int ndarray — [bl_ij, bl_kl, bl_ik, bl_jl]

    Returns
    -------
    camps : (N_quad,) ndarray — closure amplitudes (positive real)
    """
    amp = np.abs(vis)
    num = amp[quadrangles[:, 0]] * amp[quadrangles[:, 1]]
    den = amp[quadrangles[:, 2]] * amp[quadrangles[:, 3]]
    camps = num / np.maximum(den, 1e-30)
    return camps


def compute_log_closure_amplitudes(vis: np.ndarray, quadrangles: np.ndarray) -> np.ndarray:
    """
    Compute log closure amplitudes.

    log CA = log|V_ij| + log|V_kl| - log|V_ik| - log|V_jl|

    Parameters
    ----------
    vis : (M,) complex ndarray
    quadrangles : (N_quad, 4) int ndarray

    Returns
    -------
    log_camps : (N_quad,) ndarray
    """
    log_amp = np.log(np.maximum(np.abs(vis), 1e-30))
    log_camps = (log_amp[quadrangles[:, 0]] + log_amp[quadrangles[:, 1]]
                 - log_amp[quadrangles[:, 2]] - log_amp[quadrangles[:, 3]])
    return log_camps


def closure_phase_sigma(vis: np.ndarray, noise_std_per_vis: np.ndarray,
                        triangles: np.ndarray) -> np.ndarray:
    """
    Compute closure phase noise standard deviation.

    For triangle with baselines (1,2,3), the closure phase noise is:
        σ_ψ = sqrt(σ₁²/|V₁|² + σ₂²/|V₂|² + σ₃²/|V₃|²)

    (Eq. 11 of Chael et al. 2018, linearized error propagation.)

    Parameters
    ----------
    vis : (M,) complex ndarray
    noise_std_per_vis : (M,) ndarray
    triangles : (N_tri, 3) int ndarray

    Returns
    -------
    sigma_cp : (N_tri,) ndarray — closure phase noise in radians
    """
    amp = np.maximum(np.abs(vis), 1e-30)
    sigma_cp = np.zeros(len(triangles))
    for t, tri_bl in enumerate(triangles):
        var_sum = 0.0
        for bl_code in tri_bl:
            bl_idx = abs(bl_code) if bl_code >= 0 else -(bl_code + 1)
            var_sum += (noise_std_per_vis[bl_idx] / amp[bl_idx]) ** 2
        sigma_cp[t] = np.sqrt(var_sum)
    return sigma_cp


def closure_amplitude_sigma(vis: np.ndarray, noise_std_per_vis: np.ndarray,
                            quadrangles: np.ndarray) -> np.ndarray:
    """
    Compute log closure amplitude noise standard deviation.

    σ_logCA = sqrt(1/SNR₁² + 1/SNR₂² + 1/SNR₃² + 1/SNR₄²)

    where SNR_k = |V_k| / σ_k.

    (Eq. 12 of Chael et al. 2018.)

    Parameters
    ----------
    vis : (M,) complex ndarray
    noise_std_per_vis : (M,) ndarray
    quadrangles : (N_quad, 4) int ndarray

    Returns
    -------
    sigma_logca : (N_quad,) ndarray
    """
    amp = np.maximum(np.abs(vis), 1e-30)
    snr = amp / np.maximum(noise_std_per_vis, 1e-30)

    var_sum = (1.0 / snr[quadrangles[:, 0]] ** 2 +
               1.0 / snr[quadrangles[:, 1]] ** 2 +
               1.0 / snr[quadrangles[:, 2]] ** 2 +
               1.0 / snr[quadrangles[:, 3]] ** 2)
    return np.sqrt(var_sum)


def prepare_data(data_dir: str = "data") -> tuple:
    """
    Load and prepare all data needed for closure-only reconstruction.

    Parameters
    ----------
    data_dir : str

    Returns
    -------
    obs : dict
        Observation data including visibilities, station_ids, noise.
    closure_data : dict
        Closure phases, closure amplitudes, their noise estimates,
        triangle/quadrangle index arrays.
    metadata : dict
        Imaging parameters.
    """
    obs = load_observation(data_dir)
    metadata = load_metadata(data_dir)
    n_stations = metadata["n_stations"]

    vis = obs["vis_corrupted"]
    station_ids = obs["station_ids"]
    noise_per_vis = obs["noise_std_per_vis"]

    triangles, tri_stations = find_triangles(station_ids, n_stations)
    quadrangles, quad_stations = find_quadrangles(station_ids, n_stations)

    cphases = compute_closure_phases(vis, triangles, station_ids)
    log_camps = compute_log_closure_amplitudes(vis, quadrangles)
    sigma_cp = closure_phase_sigma(vis, noise_per_vis, triangles)
    sigma_logca = closure_amplitude_sigma(vis, noise_per_vis, quadrangles)

    closure_data = {
        "cphases": cphases,
        "log_camps": log_camps,
        "sigma_cp": sigma_cp,
        "sigma_logca": sigma_logca,
        "triangles": triangles,
        "quadrangles": quadrangles,
        "tri_stations": tri_stations,
        "quad_stations": quad_stations,
    }

    return obs, closure_data, metadata
