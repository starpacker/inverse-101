"""
Generate Synthetic EHT Dataset
================================

Creates a synthetic M87*-like black hole dataset and saves it to data/.

Usage
-----
    python generate_data.py [--N 64] [--snr 20] [--seed 42]

Output files (in data/)
-----------------------
    image.npy         ground truth sky brightness  (N, N)
    uv_coords.npy     (u, v) baseline positions    (M, 2)  [wavelengths]
    vis_clean.npy     noiseless visibilities        (M,)    complex
    vis_noisy.npy     noisy visibilities            (M,)    complex
    dataset.npz       all of the above + metadata
"""

import sys
import os
import argparse
import numpy as np

# Allow running as a script from the task root directory
sys.path.insert(0, os.path.dirname(__file__))
from src.physics_model import VLBIForwardModel


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic image
# ═══════════════════════════════════════════════════════════════════════════

def make_ring_image(
    N: int = 64,
    ring_radius_frac: float = 0.22,
    ring_width_frac: float = 0.055,
    asymmetry: float = 0.5,
    asymmetry_angle_deg: float = 220.0,
    include_jet: bool = False,
) -> np.ndarray:
    """
    Synthetic black hole ring image (M87*-like morphology).

    The brightness distribution mimics the EHT 2019 observations of M87*:
    a bright annular ring (emission from the photon ring / accretion disk)
    surrounding a dark central region (the black hole shadow), with an
    azimuthal brightness asymmetry caused by Doppler boosting from the jet.

    Parameters
    ----------
    N : int
        Image size (N × N pixels).
    ring_radius_frac : float
        Ring radius as a fraction of the image half-width.
    ring_width_frac : float
        Ring Gaussian half-width as fraction of image half-width.
    asymmetry : float
        Brightness contrast ratio between brightest and dimmest arc.
        0 = uniform ring, 1 = strong asymmetry.
    asymmetry_angle_deg : float
        Position angle of the brightest arc (degrees, measured from East).
    include_jet : bool
        If True, add a faint jet structure in the asymmetry direction.

    Returns
    -------
    image : (N, N) ndarray  normalised so that image.sum() = 1.
    """
    coords = np.linspace(-1.0, 1.0, N)
    xx, yy = np.meshgrid(coords, coords)

    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    # Azimuthal brightness modulation (cos-model for Doppler boosting)
    phi0 = np.deg2rad(asymmetry_angle_deg)
    brightness_mod = 1.0 + asymmetry * np.cos(theta - phi0)
    brightness_mod = np.maximum(brightness_mod, 0.0)

    # Gaussian ring profile
    ring = (
        np.exp(-((r - ring_radius_frac) ** 2) / (2.0 * ring_width_frac ** 2))
        * brightness_mod
    )

    if include_jet:
        # Faint jet pointing opposite to the bright arc
        jet_angle = phi0 + np.pi
        jet_perp = np.abs(xx * np.sin(jet_angle) - yy * np.cos(jet_angle))
        jet_para = xx * np.cos(jet_angle) + yy * np.sin(jet_angle)
        jet = (
            0.06
            * np.exp(-(jet_perp ** 2) / 0.015)
            * np.exp(-jet_para / 0.35)
        )
        jet[jet_para < ring_radius_frac] = 0.0
        jet[jet_para < 0.0] = 0.0
        ring += jet

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
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate EHT uv-coverage via Earth-rotation aperture synthesis.

    Each pair of EHT stations traces an elliptical arc in the (u,v)-plane
    as the Earth rotates. The resulting coverage is sparse and anisotropic,
    which is the core challenge of VLBI imaging.

    Parameters
    ----------
    source_dec_deg : float
        Source declination in degrees. M87* ≈ +12.39°, Sgr A* ≈ −29.01°.
    obs_duration_hours : float
        Total observation window.
    n_time_steps : int
        Number of time samples per baseline track.
    freq_ghz : float
        Observing frequency in GHz. EHT observes at 230 GHz (1.3 mm).
    seed : int
        Random seed (unused here, kept for API consistency).

    Returns
    -------
    uv_coords : (M, 2) ndarray  in wavelengths.

    Notes
    -----
    Uses the standard UVW projection:
        u = sin(H) Bx + cos(H) By
        v = −sin(δ) cos(H) Bx + sin(δ) sin(H) By + cos(δ) Bz
    where H = hour angle, δ = declination, (Bx, By, Bz) = baseline in ECEF.

    Reference
    ---------
    Thompson, Moran & Swenson (2017), §4.2.
    """
    # EHT 2017 telescope positions (geodetic latitude, longitude [degrees])
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

    R_earth = 6_371_000.0               # metres
    wavelength = 3e8 / (freq_ghz * 1e9) # metres

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
    dec = np.deg2rad(source_dec_deg)

    # Hour angle range centred on transit
    ha_arr = np.deg2rad(
        np.linspace(
            -obs_duration_hours / 2 * 15.0,
             obs_duration_hours / 2 * 15.0,
            n_time_steps,
        )
    )

    uv_list = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            B = positions[names[j]] - positions[names[i]]   # baseline vector [m]
            for ha in ha_arr:
                u = np.sin(ha) * B[0] + np.cos(ha) * B[1]
                v = (-np.sin(dec) * np.cos(ha) * B[0]
                     + np.sin(dec) * np.sin(ha) * B[1]
                     + np.cos(dec) * B[2])
                uv_list.append([u / wavelength, v / wavelength])

    return np.array(uv_list)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset assembly
# ═══════════════════════════════════════════════════════════════════════════

def generate_dataset(
    N: int = 64,
    pixel_size_uas: float = 2.0,
    snr: float = 20.0,
    seed: int = 42,
    save_dir: str = "data",
) -> dict:
    """
    Generate and save a complete synthetic EHT dataset.

    Parameters
    ----------
    N             : image size (N × N pixels)
    pixel_size_uas: pixel size in microarcseconds (typical EHT: 1–3 μas)
    snr           : per-visibility signal-to-noise ratio
    seed          : random seed for reproducibility
    save_dir      : directory to write output files

    Returns
    -------
    dict with keys: image, uv_coords, vis_clean, vis_noisy,
                    pixel_size_rad, pixel_size_uas, noise_std, N
    """
    rng = np.random.default_rng(seed)

    uas_to_rad = np.pi / (180.0 * 3600.0 * 1e6)
    pixel_size_rad = pixel_size_uas * uas_to_rad

    print(f"Generating {N}×{N} synthetic M87* image …")
    image = make_ring_image(N=N)

    print("Simulating EHT uv-coverage …")
    uv_coords = simulate_eht_uv_coverage(seed=seed)
    print(f"  → {len(uv_coords)} baselines sampled")

    print("Computing visibilities …")
    model = VLBIForwardModel(uv_coords, N, pixel_size_rad)
    vis_clean = model.forward(image)

    print(f"Adding noise (SNR = {snr}) …")
    vis_noisy, noise_std = model.add_noise(vis_clean, snr=snr, rng=rng)

    dataset = dict(
        image=image,
        uv_coords=uv_coords,
        vis_clean=vis_clean,
        vis_noisy=vis_noisy,
        pixel_size_rad=np.array(pixel_size_rad),
        pixel_size_uas=np.array(pixel_size_uas),
        noise_std=np.array(noise_std),
        N=np.array(N),
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Individual .npy files (convenient for quick loading)
        for key, val in dataset.items():
            np.save(os.path.join(save_dir, f"{key}.npy"), val)
        # Single .npz archive
        np.savez(os.path.join(save_dir, "dataset.npz"), **dataset)
        print(f"\nDataset saved to '{save_dir}/'")
        print(f"  image shape   : {image.shape}")
        print(f"  baselines     : {len(uv_coords)}")
        print(f"  pixel size    : {pixel_size_uas} μas")
        print(f"  noise_std     : {noise_std:.4e}")

    return dataset


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic EHT dataset")
    parser.add_argument("--N",   type=int,   default=64,   help="Image size (default 64)")
    parser.add_argument("--snr", type=float, default=20.0, help="Per-visibility SNR (default 20)")
    parser.add_argument("--seed",type=int,   default=42,   help="Random seed (default 42)")
    parser.add_argument("--out", type=str,   default="data", help="Output directory (default: data/)")
    args = parser.parse_args()

    generate_dataset(N=args.N, snr=args.snr, seed=args.seed, save_dir=args.out)
