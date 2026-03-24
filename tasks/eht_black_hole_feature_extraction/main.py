"""
EHT Black Hole Feature Extraction (α-DPI)
===========================================

Main script orchestrating the α-DPI feature extraction pipeline:
  1. Load and preprocess EHT observation data
  2. Train α-DPI with geometric crescent model
  3. Importance sampling and posterior reweighting
  4. Evaluate parameter recovery
  5. ELBO model selection and visualization

Usage
-----
    cd tasks/eht_black_hole_feature_extraction
    python main.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

from src.preprocessing import prepare_data, load_ground_truth
from src.solvers import AlphaDPISolver
from src.visualization import (
    plot_corner,
    plot_elbo_comparison,
    plot_posterior_images,
    plot_loss_curves,
    compute_feature_metrics,
    print_feature_metrics,
)


def main():
    # ── Step 1: Load and preprocess observation data ──────────────────────
    print("Step 1: Loading and preprocessing observation data...")
    (obs, obs_data, closure_indices, nufft_params,
     flux_const, metadata) = prepare_data("data")

    npix = metadata["npix"]
    fov_uas = metadata["fov_uas"]

    print(f"  Image size      : {npix}x{npix}")
    print(f"  FOV             : {fov_uas} uas")
    print(f"  Visibilities    : {len(obs_data['vis'])}")
    print(f"  Closure phases  : {len(closure_indices['cphase_data']['cphase'])}")
    print(f"  Closure amps    : {len(closure_indices['camp_data']['camp'])}")
    print(f"  Flux estimate   : {flux_const:.4f} Jy")

    # ── Step 2: Train α-DPI with geometric model ─────────────────────────
    print("\nStep 2: Training α-DPI with crescent + Gaussian model...")
    n_gaussian = metadata.get("n_gaussian", 2)

    solver = AlphaDPISolver(
        npix=npix,
        fov_uas=fov_uas,
        n_flow=metadata["n_flow"],
        seqfrac=1.0 / metadata.get("seqfrac_inv", 16),
        n_epoch=metadata["n_epoch"],
        batch_size=metadata["batch_size"],
        lr=metadata["lr"],
        logdet_weight=metadata["logdet_weight"],
        grad_clip=metadata["grad_clip"],
        alpha=metadata.get("alpha_divergence", 1.0),
        beta=metadata.get("beta", 0.0),
        start_order=metadata.get("start_order", 4),
        decay_rate=metadata.get("decay_rate", 2000),
        geometric_model=metadata.get("geometric_model", "simple_crescent_floor_nuisance"),
        n_gaussian=n_gaussian,
        r_range=metadata.get("r_range", [10.0, 40.0]),
        width_range=metadata.get("width_range", [1.0, 40.0]),
        shift_range=metadata.get("shift_range"),
        sigma_range=metadata.get("sigma_range"),
        floor_range=metadata.get("floor_range"),
        asym_range=metadata.get("asym_range"),
        crescent_flux_range=metadata.get("crescent_flux_range"),
        gaussian_scale_range=metadata.get("gaussian_scale_range"),
    )

    result = solver.reconstruct(obs_data, closure_indices, nufft_params, flux_const)

    # ── Step 3: Importance sampling and posterior reweighting ─────────────
    print("\nStep 3: Importance sampling posterior reweighting...")
    posterior = solver.importance_resample(
        obs_data, closure_indices, nufft_params, n_samples=10000
    )
    params_physical = posterior['params_physical']
    importance_weights = posterior['importance_weights']

    # ── Step 4: Evaluate parameter recovery ───────────────────────────────
    print("\nStep 4: Evaluating parameter recovery...")
    gt_params = metadata.get("ground_truth_params", {})
    gt_array = np.array([
        gt_params.get("diameter_uas", 44.0),
        gt_params.get("width_uas", 11.36),
        gt_params.get("asymmetry", 0.5),
        gt_params.get("position_angle_deg", -90.5),
    ])
    param_names = ["diameter (μas)", "width (μas)", "asymmetry", "PA (deg)"]

    metrics = compute_feature_metrics(
        params_physical, gt_array,
        importance_weights=importance_weights,
        param_names=param_names
    )
    print()
    print_feature_metrics(metrics)

    # ── Step 5: ELBO model selection ──────────────────────────────────────
    print("\nStep 5: Computing ELBO for model selection...")
    # Compute ELBO for the trained model
    elbo = solver.compute_elbo(obs_data, closure_indices, nufft_params,
                                n_samples=10000)
    print(f"  ELBO (n_gaussian={n_gaussian}): {elbo:.4f}")

    # ── Step 6: Visualize and save ────────────────────────────────────────
    print("\nStep 6: Generating visualizations and saving outputs...")
    os.makedirs("output", exist_ok=True)

    plot_corner(
        params_physical, param_names=param_names,
        ground_truth=gt_array,
        importance_weights=importance_weights,
        save_path="output/corner_plot.png"
    )

    plot_posterior_images(
        posterior['images'], n_show=8,
        pixel_size_uas=fov_uas / npix,
        importance_weights=importance_weights,
        save_path="output/posterior_images.png"
    )

    plot_loss_curves(result['loss_history'], save_path="output/loss_curves.png")

    # Save outputs
    np.save("output/params_physical.npy", params_physical)
    np.save("output/importance_weights.npy", importance_weights)
    np.save("output/weighted_mean_image.npy", posterior['weighted_mean_image'])
    np.save("output/posterior_images.npy", posterior['images'][:100])

    import torch
    torch.save(solver.params_generator.state_dict(),
               "output/params_generator_state_dict.pt")

    # Save metrics
    all_metrics = {
        'feature_metrics': {k: v for k, v in metrics.items() if k != 'n_params'},
        'elbo': elbo,
        'n_gaussian': n_gaussian,
    }
    with open("output/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save ground truth image
    try:
        gt_image = load_ground_truth("data", npix, fov_uas)
        np.save("output/ground_truth.npy", gt_image)
    except Exception as e:
        print(f"  Warning: Could not load ground truth image: {e}")

    print("\nDone. Outputs saved to output/")
    return posterior, all_metrics


if __name__ == "__main__":
    main()
