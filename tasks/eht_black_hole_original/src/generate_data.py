"""
Generate Synthetic EHT Dataset with Gain Corruption
=====================================================

Creates a synthetic M87*-like black hole dataset with realistic
station-based gain errors, so that closure quantities differ from
calibrated visibilities.

Usage
-----
    python -c "from src.generate_data import generate_dataset; generate_dataset()"

Output files (in data/)
-----------------------
    raw_data.npz    all arrays for reconstruction
    meta_data       JSON imaging parameters
"""

import sys
import os
import json
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic image
# ═══════════════════════════════════════════════════════════════════════════

def make_ring_image(
    N: int = 64,
    ring_radius_frac: float = 0.22,
    ring_width_frac: float = 0.055,
    asymmetry: float = 0.5,
    asymmetry_angle_deg: float = 220.0,
) -> np.ndarray:
    """
    Synthetic black hole ring image (M87*-like morphology).

    Parameters
    ----------
    N : int
        Image size (N × N pixels).
    ring_radius_frac : float
        Ring radius as fraction of image half-width.
    ring_width_frac : float
        Ring Gaussian half-width as fraction of image half-width.
    asymmetry : float
        Brightness contrast (0 = uniform, 1 = strong asymmetry).
    asymmetry_angle_deg : float
        Position angle of brightest arc (degrees).

    Returns
    -------
    image : (N, N) ndarray, normalised so image.sum() = 1.
    """
    coords = np.linspace(-1.0, 1.0, N)
    xx, yy = np.meshgrid(coords, coords)

    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    phi0 = np.deg2rad(asymmetry_angle_deg)
    brightness_mod = 1.0 + asymmetry * np.cos(theta - phi0)
    brightness_mod = np.maximum(brightness_mod, 0.0)

    ring = (
        np.exp(-((r - ring_radius_frac) ** 2) / (2.0 * ring_width_frac ** 2))
        * brightness_mod
    )

    ring = np.maximum(ring, 0.0)
    ring /= ring.sum()
    return ring


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic uv-coverage
# ═══════════════════════════════════════════════════════════════════════════

def simulate_eht_uv_coverage(
    source_dec_deg: float = 12.39,
    obs_duration_hours: float = 6.0,
    n_time_steps: int = 15,
    freq_ghz: float = 230.0,
) -> tuple:
    """
    Simulate EHT uv-coverage with station pair information.

    Returns
    -------
    uv_coords : (M, 2) ndarray in wavelengths
    station_ids : (M, 2) int ndarray — station pair for each baseline
    n_stations : int
    """
    telescopes = {
        "ALMA":  (-23.023, -67.755),
        "APEX":  (-23.006, -67.759),
        "JCMT":  (19.822, -155.477),
        "SMA":   (19.824, -155.455),
        "IRAM":  (37.066,  -3.392),
        "LMT":   (18.986, -97.315),
        "SMT":   (32.701, -109.892),
        "SPT":   (-89.991,   0.000),
        "NOEMA": (44.634,   5.908),
    }

    R_earth = 6_371_000.0
    wavelength = 3e8 / (freq_ghz * 1e9)

    def lonlat_to_ecef(lat_deg, lon_deg):
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
        return R_earth * np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])

    positions = {
        name: lonlat_to_ecef(lat, lon)
        for name, (lat, lon) in telescopes.items()
    }
    names = list(positions.keys())
    n_stations = len(names)
    dec = np.deg2rad(source_dec_deg)

    ha_arr = np.deg2rad(
        np.linspace(
            -obs_duration_hours / 2 * 15.0,
             obs_duration_hours / 2 * 15.0,
            n_time_steps,
        )
    )

    uv_list = []
    sid_list = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            B = positions[names[j]] - positions[names[i]]
            for ha in ha_arr:
                u = np.sin(ha) * B[0] + np.cos(ha) * B[1]
                v = (-np.sin(dec) * np.cos(ha) * B[0]
                     + np.sin(dec) * np.sin(ha) * B[1]
                     + np.cos(dec) * B[2])
                uv_list.append([u / wavelength, v / wavelength])
                sid_list.append([i, j])

    return np.array(uv_list), np.array(sid_list, dtype=np.int64), n_stations


def apply_station_gains(
    vis_clean: np.ndarray,
    station_ids: np.ndarray,
    n_stations: int,
    amp_error: float = 0.2,
    phase_error_deg: float = 30.0,
    rng=None,
) -> tuple:
    """
    Apply station-based gain errors to visibilities.

    Each station i gets a complex gain g_i. The corrupted visibility is:
        V_ij^corr = g_i * conj(g_j) * V_ij^true

    Gain amplitudes: |g_i| ~ 1 + U(-amp_error, amp_error)
    Gain phases: arg(g_i) ~ U(-phase_error, phase_error)

    Parameters
    ----------
    vis_clean : (M,) complex — true visibilities
    station_ids : (M, 2) int
    n_stations : int
    amp_error : float — fractional amplitude error (e.g. 0.2 = 20%)
    phase_error_deg : float — max phase error in degrees
    rng : numpy random Generator

    Returns
    -------
    vis_corrupted : (M,) complex
    gains : (n_stations,) complex — applied gains
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate station gains
    amp = 1.0 + rng.uniform(-amp_error, amp_error, n_stations)
    phase = rng.uniform(-np.deg2rad(phase_error_deg),
                        np.deg2rad(phase_error_deg), n_stations)
    gains = amp * np.exp(1j * phase)

    # Apply gains: V_ij^corr = g_i * conj(g_j) * V_ij^true
    g_i = gains[station_ids[:, 0]]
    g_j = gains[station_ids[:, 1]]
    vis_corrupted = g_i * np.conj(g_j) * vis_clean

    return vis_corrupted, gains


# ═══════════════════════════════════════════════════════════════════════════
# Forward model (self-contained for data generation)
# ═══════════════════════════════════════════════════════════════════════════

def _build_measurement_matrix(uv_coords, N, pixel_size_rad):
    """Build DFT measurement matrix A, shape (M, N²)."""
    idx = np.arange(N) - N // 2
    l, m = np.meshgrid(idx * pixel_size_rad, idx * pixel_size_rad)
    l_flat = l.ravel()
    m_flat = m.ravel()
    phase = -2j * np.pi * (
        uv_coords[:, 0:1] * l_flat[np.newaxis, :] +
        uv_coords[:, 1:2] * m_flat[np.newaxis, :]
    )
    return np.exp(phase)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset assembly
# ═══════════════════════════════════════════════════════════════════════════

def generate_dataset(
    N: int = 64,
    pixel_size_uas: float = 2.0,
    snr: float = 20.0,
    gain_amp_error: float = 0.2,
    gain_phase_error_deg: float = 30.0,
    seed: int = 42,
    save_dir: str = "data",
) -> dict:
    """
    Generate and save a complete synthetic EHT dataset with gain corruption.

    Parameters
    ----------
    N : int — image size
    pixel_size_uas : float — pixel size in microarcseconds
    snr : float — per-visibility signal-to-noise ratio
    gain_amp_error : float — fractional amplitude gain error
    gain_phase_error_deg : float — phase gain error in degrees
    seed : int — random seed
    save_dir : str — output directory

    Returns
    -------
    dict with all dataset arrays
    """
    rng = np.random.default_rng(seed)

    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    pixel_size_rad = pixel_size_uas * uas_to_rad

    print(f"Generating {N}×{N} synthetic M87* image ...")
    image = make_ring_image(N=N)

    print("Simulating EHT uv-coverage ...")
    uv_coords, station_ids, n_stations = simulate_eht_uv_coverage()
    M = len(uv_coords)
    print(f"  -> {M} baselines, {n_stations} stations")

    print("Computing visibilities ...")
    A = _build_measurement_matrix(uv_coords, N, pixel_size_rad)
    vis_true = A @ image.ravel()

    print(f"Adding noise (SNR = {snr}) ...")
    signal_rms = np.sqrt(np.mean(np.abs(vis_true) ** 2))
    noise_std = signal_rms / snr
    noise_std_per_vis = np.full(M, noise_std)
    noise = noise_std * (
        rng.standard_normal(M) + 1j * rng.standard_normal(M)
    ) / np.sqrt(2)
    vis_true_noisy = vis_true + noise

    print(f"Applying gain errors (amp={gain_amp_error:.0%}, phase={gain_phase_error_deg}°) ...")
    vis_corrupted, gains = apply_station_gains(
        vis_true_noisy, station_ids, n_stations,
        amp_error=gain_amp_error,
        phase_error_deg=gain_phase_error_deg,
        rng=rng,
    )

    dataset = dict(
        image=image,
        vis_true=vis_true_noisy,
        vis_corrupted=vis_corrupted,
        uv_coords=uv_coords,
        station_ids=station_ids,
        noise_std_per_vis=noise_std_per_vis,
        gains=gains,
    )

    metadata = {
        "N": N,
        "pixel_size_uas": pixel_size_uas,
        "pixel_size_rad": pixel_size_rad,
        "noise_std": float(noise_std),
        "freq_ghz": 230.0,
        "source_dec_deg": 12.39,
        "obs_duration_hours": 6.0,
        "n_baselines": M,
        "n_stations": n_stations,
        "snr": snr,
        "gain_amp_error": gain_amp_error,
        "gain_phase_error_deg": gain_phase_error_deg,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        np.savez(
            os.path.join(save_dir, "raw_data.npz"),
            vis_corrupted=vis_corrupted,
            vis_true=vis_true_noisy,
            uv_coords=uv_coords,
            station_ids=station_ids,
            noise_std_per_vis=noise_std_per_vis,
        )

        with open(os.path.join(save_dir, "meta_data"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save ground truth for evaluation
        ref_dir = os.path.join(os.path.dirname(save_dir), "evaluation", "reference_outputs")
        os.makedirs(ref_dir, exist_ok=True)
        np.save(os.path.join(ref_dir, "ground_truth.npy"), image)

        print(f"\nDataset saved to '{save_dir}/'")
        print(f"  image shape   : {image.shape}")
        print(f"  baselines     : {M}")
        print(f"  stations      : {n_stations}")
        print(f"  pixel size    : {pixel_size_uas} μas")
        print(f"  noise_std     : {noise_std:.4e}")
        print(f"  gain errors   : amp={gain_amp_error:.0%}, phase={gain_phase_error_deg}°")

    dataset["metadata"] = metadata
    return dataset


if __name__ == "__main__":
    generate_dataset()
