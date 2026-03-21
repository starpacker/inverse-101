"""
EHT Black Hole Image Reconstruction
=====================================

Main script orchestrating the full reconstruction pipeline:
  1. Load and preprocess observation data
  2. Build the VLBI forward model
  3. Reconstruct images using multiple methods
  4. Evaluate reconstruction quality
  5. Visualize and save results

Usage
-----
    cd tasks/eht_black_hole
    python main.py
"""

import os
import numpy as np

from src.preprocessing import prepare_data
from src.physics_model import VLBIForwardModel
from src.solvers import (
    DirtyImageReconstructor,
    CLEANReconstructor,
    RMLSolver,
    TVRegularizer,
    MaxEntropyRegularizer,
)
from src.visualization import (
    plot_uv_coverage,
    plot_comparison,
    plot_summary_panel,
    compute_metrics,
    print_metrics_table,
)


def main():
    # ── Step 1: Load and preprocess observation data ──────────────────────
    print("Step 1: Loading observation data...")
    vis_noisy, uv_coords, metadata = prepare_data("data")

    N = metadata["N"]
    pixel_size_rad = metadata["pixel_size_rad"]
    pixel_size_uas = metadata["pixel_size_uas"]
    noise_std = metadata["noise_std"]

    print(f"  Image size     : {N}x{N}")
    print(f"  Baselines      : {len(uv_coords)}")
    print(f"  Pixel size     : {pixel_size_uas} μas")
    print(f"  Noise std      : {noise_std:.4e}")

    # ── Step 2: Build forward model ───────────────────────────────────────
    print("\nStep 2: Building VLBI forward model...")
    model = VLBIForwardModel(uv_coords, N, pixel_size_rad)
    print(f"  {model}")

    # ── Step 3: Reconstruct with multiple methods ─────────────────────────
    print("\nStep 3: Reconstructing images...")

    print("  [1/4] Dirty Image (baseline back-projection)...")
    x_dirty = DirtyImageReconstructor().reconstruct(model, vis_noisy, noise_std)

    print("  [2/4] CLEAN (Högbom, support_radius=15)...")
    x_clean = CLEANReconstructor(
        gain=0.1, n_iter=500, threshold=1e-4, support_radius=15
    ).reconstruct(model, vis_noisy, noise_std)

    print("  [3/4] RML with Total Variation (λ=5e3)...")
    x_tv = RMLSolver(
        regularizers=[(5e3, TVRegularizer())], n_iter=500
    ).reconstruct(model, vis_noisy, noise_std)

    print("  [4/4] RML with Maximum Entropy (λ=1e4)...")
    x_mem = RMLSolver(
        regularizers=[(1e4, MaxEntropyRegularizer())], n_iter=500
    ).reconstruct(model, vis_noisy, noise_std)

    reconstructions = {
        "Dirty Image": x_dirty,
        "CLEAN": x_clean,
        "RML-TV": x_tv,
        "RML-MEM": x_mem,
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
    plot_uv_coverage(uv_coords)
    plot_comparison(
        reconstructions, ground_truth=ground_truth,
        pixel_size_uas=pixel_size_uas, metrics=metrics,
    )

    # Save primary reconstruction output (RML-TV)
    os.makedirs("output", exist_ok=True)
    np.save("output/reconstruction.npy", x_tv)
    print("\nDone. Primary reconstruction (RML-TV) saved to output/reconstruction.npy")

    return reconstructions, metrics


if __name__ == "__main__":
    main()
