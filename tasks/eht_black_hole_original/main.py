"""
Closure-Only EHT Black Hole Image Reconstruction
==================================================

Main script demonstrating that imaging with closure quantities (closure
phases and closure amplitudes) is robust to station-based gain errors,
while traditional visibility-based imaging fails.

Reproduces the key result of Chael et al. (2018), ApJ 857, 23:
Closure-only imaging produces correct results even when station gains
are corrupted, because closure quantities are gain-invariant.

Pipeline:
  1. Load observation data (gain-corrupted visibilities)
  2. Compute closure quantities (closure phases, log closure amplitudes)
  3. Reconstruct using closure-only methods and visibility-based method
  4. Compare: closure-only methods recover the source; visibility method fails
  5. Visualize and save results

Usage
-----
    cd tasks/eht_black_hole_original
    python main.py
"""

import os
import numpy as np

from src.preprocessing import prepare_data
from src.physics_model import ClosureForwardModel
from src.solvers import (
    ClosurePhaseOnlySolver,
    ClosurePhasePlusAmpSolver,
    VisibilityRMLSolver,
    TVRegularizer,
    MaxEntropyRegularizer,
)
from src.visualization import (
    plot_uv_coverage,
    plot_comparison,
    plot_closure_phases,
    compute_metrics,
    print_metrics_table,
)


def main():
    # ── Step 1: Load and preprocess observation data ──────────────────────
    print("Step 1: Loading observation data...")
    obs, closure_data, metadata = prepare_data("data")

    N = metadata["N"]
    pixel_size_rad = metadata["pixel_size_rad"]
    pixel_size_uas = metadata["pixel_size_uas"]
    noise_std = metadata["noise_std"]

    print(f"  Image size     : {N}x{N}")
    print(f"  Baselines      : {metadata['n_baselines']}")
    print(f"  Stations       : {metadata['n_stations']}")
    print(f"  Pixel size     : {pixel_size_uas} μas")
    print(f"  Gain errors    : amp={metadata['gain_amp_error']:.0%}, "
          f"phase={metadata['gain_phase_error_deg']}°")
    print(f"  Triangles      : {len(closure_data['cphases'])}")
    print(f"  Quadrangles    : {len(closure_data['log_camps'])}")

    # ── Step 2: Build forward model ───────────────────────────────────────
    print("\nStep 2: Building closure forward model...")
    model = ClosureForwardModel(
        uv_coords=obs["uv_coords"],
        image_size=N,
        pixel_size_rad=pixel_size_rad,
        station_ids=obs["station_ids"],
        triangles=closure_data["triangles"],
        quadrangles=closure_data["quadrangles"],
    )
    print(f"  {model}")

    # ── Step 3: Reconstruct with multiple methods ─────────────────────────
    print("\nStep 3: Reconstructing images...")

    print("  [1/4] Closure Phase Only (RML-TV)...")
    x_cp_tv = ClosurePhaseOnlySolver(
        regularizers=[(1e3, TVRegularizer())],
        alpha_cp=150.0,
        n_iter=500,
    ).reconstruct(model, closure_data)

    print("  [2/4] Closure Phase + Log Closure Amplitude (RML-TV)...")
    x_cpca_tv = ClosurePhasePlusAmpSolver(
        regularizers=[(1e3, TVRegularizer())],
        alpha_cp=150.0,
        alpha_ca=150.0,
        n_iter=500,
    ).reconstruct(model, closure_data)

    print("  [3/4] Closure Phase + Log Closure Amplitude (RML-MEM)...")
    x_cpca_mem = ClosurePhasePlusAmpSolver(
        regularizers=[(5e3, MaxEntropyRegularizer())],
        alpha_cp=150.0,
        alpha_ca=150.0,
        n_iter=500,
    ).reconstruct(model, closure_data)

    print("  [4/4] Visibility RML-TV (using corrupted visibilities)...")
    x_vis_tv = VisibilityRMLSolver(
        regularizers=[(5e3, TVRegularizer())],
        n_iter=500,
    ).reconstruct(model, obs["vis_corrupted"], noise_std)

    reconstructions = {
        "CP-only (TV)": x_cp_tv,
        "CP+CA (TV)": x_cpca_tv,
        "CP+CA (MEM)": x_cpca_mem,
        "Visibility (TV)": x_vis_tv,
    }

    # ── Step 4: Evaluate reconstruction quality ───────────────────────────
    print("\nStep 4: Evaluating reconstruction quality...")
    gt_path = os.path.join("evaluation", "reference_outputs", "ground_truth.npy")
    ground_truth = np.load(gt_path)

    metrics = {}
    for name, recon in reconstructions.items():
        metrics[name] = compute_metrics(recon, ground_truth)

    print()
    print_metrics_table(metrics)

    # ── Step 5: Visualize and save ────────────────────────────────────────
    print("\nStep 5: Generating visualizations...")
    os.makedirs("output", exist_ok=True)

    plot_uv_coverage(obs["uv_coords"])
    plt_fig = plot_comparison(
        reconstructions, ground_truth=ground_truth,
        pixel_size_uas=pixel_size_uas, metrics=metrics,
    )

    plot_closure_phases(closure_data["cphases"])

    # Save primary reconstruction (closure phase + amplitude with TV)
    np.save("output/reconstruction.npy", x_cpca_tv)
    print("\nDone. Primary reconstruction (CP+CA TV) saved to output/reconstruction.npy")

    return reconstructions, metrics


if __name__ == "__main__":
    main()
