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
import json
import numpy as np

from src.preprocessing import (
    prepare_data, compute_closure_phases, compute_log_closure_amplitudes,
    closure_phase_sigma, closure_amplitude_sigma,
)
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
from src.generate_data import apply_station_gains


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

    # ── Step 3: Reconstruct ───────────────────────────────────────────────
    print("\nStep 3: Reconstructing images...")

    gt_path = os.path.join("evaluation", "reference_outputs", "ground_truth.npy")
    ground_truth = np.load(gt_path)

    # --- Reference: Visibility RML with TRUE (uncorrupted) visibilities ---
    print("  [1/5] Reference: Visibility RML-TV (true visibilities)...")
    x_vis_true = VisibilityRMLSolver(
        regularizers=[(5e3, TVRegularizer())],
        n_iter=500,
    ).reconstruct(model, obs["vis_true"], noise_std)

    # --- Visibility RML with CORRUPTED visibilities (baseline) ---
    print("  [2/5] Visibility RML-TV (corrupted visibilities)...")
    x_vis_corrupted = VisibilityRMLSolver(
        regularizers=[(5e3, TVRegularizer())],
        n_iter=500,
    ).reconstruct(model, obs["vis_corrupted"], noise_std)

    # --- Closure-only: use visibility init then refine with closure χ² ---
    # (Following Chael et al. 2018, closure imaging benefits from a
    #  reasonable initialization since the closure χ² landscape is nonlinear.)
    print("  [3/5] Closure Phase Only (RML-TV, 2-step)...")
    x_cp_tv = ClosurePhaseOnlySolver(
        regularizers=[(1e2, TVRegularizer())],
        alpha_cp=100.0,
        n_iter=2000,
    ).reconstruct(model, closure_data, x0=x_vis_corrupted)

    print("  [4/5] CP+CA (RML-TV, 2-step)...")
    x_cpca_tv = ClosurePhasePlusAmpSolver(
        regularizers=[(1e2, TVRegularizer())],
        alpha_cp=100.0,
        alpha_ca=100.0,
        n_iter=2000,
    ).reconstruct(model, closure_data, x0=x_vis_corrupted)

    print("  [5/5] CP+CA (RML-MEM, 2-step)...")
    x_cpca_mem = ClosurePhasePlusAmpSolver(
        regularizers=[(5e3, MaxEntropyRegularizer())],
        alpha_cp=100.0,
        alpha_ca=100.0,
        n_iter=2000,
    ).reconstruct(model, closure_data, x0=x_vis_corrupted)

    reconstructions = {
        "Vis (true)": x_vis_true,
        "Vis (corrupted)": x_vis_corrupted,
        "CP-only (TV)": x_cp_tv,
        "CP+CA (TV)": x_cpca_tv,
        "CP+CA (MEM)": x_cpca_mem,
    }

    # ── Step 4: Evaluate reconstruction quality ───────────────────────────
    print("\nStep 4: Evaluating reconstruction quality...")

    metrics = {}
    for name, recon in reconstructions.items():
        metrics[name] = compute_metrics(recon, ground_truth)

    print()
    print_metrics_table(metrics)

    # ── Step 5: Gain robustness demonstration ─────────────────────────────
    print("\nStep 5: Gain robustness sweep (Figure 5 equivalent)...")
    print("  Demonstrating that closure methods are stable as gain errors grow,")
    print("  while visibility imaging degrades.\n")

    vis_true_noisy = obs["vis_true"]
    station_ids = obs["station_ids"]
    n_stations = metadata["n_stations"]
    tri = closure_data["triangles"]
    quad = closure_data["quadrangles"]

    gain_levels = [0.0, 0.2, 0.5, 1.0]
    sweep_results = {}

    header = f"{'Gain error':>12}  {'Vis NRMSE':>10}  {'Vis NCC':>8}  {'CP+CA NRMSE':>12}  {'CP+CA NCC':>10}"
    print(header)
    print("-" * len(header))

    for amp_err in gain_levels:
        phase_err = amp_err * 150.0  # scale phase error with amp error
        rng = np.random.default_rng(42)

        if amp_err > 0:
            vis_corr, _ = apply_station_gains(
                vis_true_noisy, station_ids, n_stations,
                amp_error=amp_err, phase_error_deg=phase_err, rng=rng,
            )
        else:
            vis_corr = vis_true_noisy.copy()

        # Visibility RML
        x_v = VisibilityRMLSolver(
            regularizers=[(5e3, TVRegularizer())], n_iter=500,
        ).reconstruct(model, vis_corr, noise_std)

        # Closure-only (CP+CA) — closure quantities are gain-invariant
        cp = compute_closure_phases(vis_corr, tri, station_ids)
        lca = compute_log_closure_amplitudes(vis_corr, quad)
        sig_cp = closure_phase_sigma(vis_corr, obs["noise_std_per_vis"], tri)
        sig_lca = closure_amplitude_sigma(vis_corr, obs["noise_std_per_vis"], quad)
        cd = dict(cphases=cp, log_camps=lca, sigma_cp=sig_cp,
                  sigma_logca=sig_lca, triangles=tri, quadrangles=quad)

        x_c = ClosurePhasePlusAmpSolver(
            regularizers=[(1e2, TVRegularizer())],
            alpha_cp=100.0, alpha_ca=100.0, n_iter=2000,
        ).reconstruct(model, cd, x0=x_v)

        m_v = compute_metrics(x_v, ground_truth)
        m_c = compute_metrics(x_c, ground_truth)

        print(f"  {amp_err:>8.0%}+{phase_err:>3.0f}°"
              f"  {m_v['nrmse']:>10.4f}  {m_v['ncc']:>8.4f}"
              f"  {m_c['nrmse']:>12.4f}  {m_c['ncc']:>10.4f}")

        sweep_results[amp_err] = {"vis": m_v, "closure": m_c}

    # ── Step 6: Visualize and save ────────────────────────────────────────
    print("\nStep 6: Generating visualizations...")
    os.makedirs("output", exist_ok=True)

    plot_uv_coverage(obs["uv_coords"])
    plot_comparison(
        reconstructions, ground_truth=ground_truth,
        pixel_size_uas=pixel_size_uas, metrics=metrics,
    )
    plot_closure_phases(closure_data["cphases"])

    # Save primary reconstruction
    np.save("output/reconstruction.npy", x_cpca_tv)

    # Save reference outputs for integration tests
    ref_dir = os.path.join("evaluation", "reference_outputs")
    os.makedirs(ref_dir, exist_ok=True)
    np.save(os.path.join(ref_dir, "cpca_tv.npy"), x_cpca_tv)

    # Save metrics
    metrics_serializable = {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                            for k, v in metrics.items()}
    with open(os.path.join(ref_dir, "metrics.json"), "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    print("\nDone. Primary reconstruction (CP+CA TV) saved to output/reconstruction.npy")
    return reconstructions, metrics


if __name__ == "__main__":
    main()
