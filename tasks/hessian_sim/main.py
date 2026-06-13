"""
Hessian-SIM Reconstruction Pipeline — Main Orchestrator

All parameter configuration is in this file.
Calls: preprocess → solve → evaluate.
"""

import os
import time
import numpy as np
import tifffile

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'observation', 'Simulation_twobeam_noise_NA0.9.tif')
OTF_PATH = os.path.join(BASE_DIR, 'data', 'metadata', '488OTF_512.tif')  # measured OTF
OUT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# SIM parameters (from Supplementary Figure 8 description)
NANGLES = 2
NPHASES = 3
WAVELENGTH = 488        # nm (Lifeact-EGFP)
NA_DETECTION = 1.49     # detection objective NA (100x, CFI Apochromat TIRF)
NA_EXCITATION = 0.9     # excitation NA (for pattern generation, from filename)
PIXEL_SIZE = 65         # nm (sCMOS camera pixel size)
WEILAC = 2.0            # Wiener regularization parameter
REGUL = 2 * np.pi       # total phase range
SPJG = [4, 4, 3]        # phase step ratios
BEISHU_AN = 1            # frame averaging for parameter estimation
BEISHU_RE = 1            # frame averaging for reconstruction
FANWEI = 50              # search band half-width for pattern frequency
BACKGROUND = 99.0        # background value added to all pixels (paper: 99 a.u.)

# Denoising parameters
MU = 150.0               # data fidelity weight (Hessian & TV)
SIGMA_Z = 1.0            # z-axis Hessian weight
N_ITER = 100             # Split-Bregman iterations
HESSIAN_LAMBDA = 0.5     # Bregman splitting parameter (from Bregman_Hessian_Denoise.m)

# ============================================================================
# Pipeline
# ============================================================================

from src.physics import generate_otf
from src.preprocessing import (
    estimate_sim_parameters,
    estimate_modulation_and_phase,
    wiener_sim_reconstruct,
    running_average,
)
from src.solver import hessian_denoise, tv_denoise
from src.visualization import plot_comparison, plot_line_profiles, plot_hessian_vs_tv


def main():
    # ---- Load raw data ----
    print("Loading raw data...")
    raw = tifffile.imread(DATA_PATH).astype(np.float64)
    print(f"  Shape: {raw.shape}, dtype: {raw.dtype}")
    nframes_total, sy, sx = raw.shape

    # ---- Background subtraction (paper: 99 a.u. added to all pixels) ----
    print(f"  Subtracting background = {BACKGROUND}")
    raw = raw - BACKGROUND
    raw[raw < 0] = 0

    # ---- Working grid size ----
    n = max(sy, sx, 512)
    print(f"  Working grid size: {n}")

    # ---- Load measured OTF ----
    print(f"Loading measured OTF from {OTF_PATH}...")
    otf = tifffile.imread(OTF_PATH).astype(np.float64)
    otf = otf / otf.max()  # normalize to [0, 1]
    assert otf.shape == (n, n), f"OTF shape {otf.shape} != expected ({n},{n})"
    print(f"  OTF shape: {otf.shape}")

    # Expected pattern frequency (uses excitation NA)
    pg_base = 512 * 2 * NA_EXCITATION * PIXEL_SIZE / WAVELENGTH
    pg = int(np.ceil(pg_base * (n / 512)))
    fanwei_scaled = int(np.ceil(FANWEI * (n / 512)))
    print(f"  pg={pg}, fanwei={fanwei_scaled}")

    # ---- Stage 1: Parameter estimation ----
    print("\n=== Stage 1: Parameter Estimation ===")
    t0 = time.time()
    n_est_frames = NANGLES * NPHASES * BEISHU_AN
    zuobiaox, zuobiaoy = estimate_sim_parameters(
        raw[:n_est_frames], otf, NANGLES, NPHASES, n,
        pg, fanwei_scaled, REGUL, SPJG, BEISHU_AN
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    # ---- Stage 2: Modulation depth & phase estimation ----
    print("\n=== Stage 2: Modulation & Phase Estimation ===")
    t0 = time.time()
    c6, angle6 = estimate_modulation_and_phase(
        raw[:n_est_frames], otf, zuobiaox, zuobiaoy,
        NANGLES, NPHASES, n, REGUL, SPJG,
        WAVELENGTH, beishu_an=BEISHU_AN, bg=None
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    # ---- Stage 3: Wiener SIM reconstruction ----
    print("\n=== Stage 3: Wiener SIM Reconstruction ===")
    t0 = time.time()
    sim_result = wiener_sim_reconstruct(
        raw, otf, zuobiaox, zuobiaoy, c6, angle6,
        NANGLES, NPHASES, n, WEILAC, REGUL, SPJG,
        beishu_re=BEISHU_RE, starframe=0, bg=None
    )
    print(f"  Time: {time.time() - t0:.1f}s")
    print(f"  Output shape: {sim_result.shape}")
    tifffile.imwrite(os.path.join(OUT_DIR, 'wiener_sim.tif'), sim_result)

    # ---- Stage 4: Hessian denoising (lambda=0.5 from Bregman_Hessian_Denoise.m) ----
    print("\n=== Stage 4: Hessian Denoising ===")
    t0 = time.time()
    hessian_result = hessian_denoise(sim_result, mu=MU, sigma_z=SIGMA_Z,
                                     n_iter=N_ITER, lamda=HESSIAN_LAMBDA)
    print(f"  Time: {time.time() - t0:.1f}s")
    tifffile.imwrite(os.path.join(OUT_DIR, 'hessian_sim.tif'), hessian_result.astype(np.float32))

    # ---- Stage 5: TV denoising ----
    print("\n=== Stage 5: TV Denoising ===")
    t0 = time.time()
    tv_result = tv_denoise(sim_result, mu=MU, n_iter=N_ITER)
    print(f"  Time: {time.time() - t0:.1f}s")
    tifffile.imwrite(os.path.join(OUT_DIR, 'tv_sim.tif'), tv_result.astype(np.float32))

    # ---- Stage 6: Running average ----
    print("\n=== Stage 6: Running Average ===")
    ra_hessian = running_average(hessian_result.astype(np.float32), window=3)
    tifffile.imwrite(os.path.join(OUT_DIR, 'running_avg_hessian.tif'), ra_hessian)

    # ---- Widefield reference (from background-subtracted data) ----
    widefield = raw.reshape(-1, NANGLES * NPHASES, sy, sx).mean(axis=1).mean(axis=0)
    raw_frame_0 = raw[0]  # single noisy raw frame for display

    # ---- Visualization ----
    print("\n=== Visualization ===")
    plot_comparison(widefield, sim_result, hessian_result, tv_result,
                    raw_frame=raw_frame_0,
                    save_path=os.path.join(OUT_DIR, 'comparison.png'))
    plot_line_profiles(sim_result, hessian_result, tv_result,
                       save_path=os.path.join(OUT_DIR, 'line_profiles.png'))
    plot_hessian_vs_tv(hessian_result, tv_result,
                       save_path=os.path.join(OUT_DIR, 'hessian_vs_tv.png'))

    print(f"\n=== Done! Results saved to: {OUT_DIR} ===")


if __name__ == '__main__':
    main()
