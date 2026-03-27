"""
SSNP Intensity Diffraction Tomography
=======================================

Main script orchestrating the full reconstruction pipeline:
  1. Load phantom and metadata
  2. Build the SSNP forward model
  3. Simulate intensity measurements (8 angles, NA=0.65)
  4. Reconstruct 3D RI via gradient descent + TV
  5. Evaluate reconstruction quality
  6. Save reference outputs and visualizations

Usage
-----
    cd tasks/SSNP_ODT
    python main.py
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data
from src.physics_model import SSNPConfig, SSNPForwardModel
from src.solvers import SSNPReconstructor
from src.visualization import (
    plot_ri_slices,
    plot_xz_cross_section,
    plot_comparison,
    plot_loss_history,
    plot_measurements,
    compute_metrics,
    print_metrics_table,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("Step 1: Loading phantom and metadata...")
    phantom_dn, metadata = prepare_data("data")
    config = SSNPConfig.from_metadata(metadata)

    print(f"  Phantom shape  : {phantom_dn.shape}")
    print(f"  Δn range       : [{phantom_dn.min():.6f}, {phantom_dn.max():.6f}]")
    print(f"  Norm. res      : {config.res}")
    print(f"  Voxel size     : {config.res_um} μm")

    # ── Step 2: Build forward model ───────────────────────────────────────
    print("\nStep 2: Building SSNP forward model...")
    model = SSNPForwardModel(config, device=device)
    print(f"  {model}")

    # ── Step 3: Simulate measurements ─────────────────────────────────────
    print("\nStep 3: Simulating IDT measurements...")
    dn_tensor = torch.tensor(phantom_dn, dtype=torch.float64, device=device)

    with torch.no_grad():
        measurements = model.forward(dn_tensor)

    meas_np = measurements.cpu().numpy()
    print(f"  {config.n_angles} angles, intensity range: "
          f"[{meas_np.min():.4f}, {meas_np.max():.4f}]")

    # ── Step 4: Reconstruct ───────────────────────────────────────────────
    print("\nStep 4: Reconstructing 3D RI...")
    reconstructor = SSNPReconstructor(
        n_iter=10,
        lr=50.0,
        tv_weight=0.0,
        positivity=True,
        device=device,
    )

    # Pass amplitudes (|field|) to the reconstructor, matching original code
    meas_amp = torch.sqrt(measurements)

    dn_recon, loss_history = reconstructor.reconstruct(meas_amp, model)
    print(f"  Final loss: {loss_history[-1]:.6f}")

    # ── Step 5: Evaluate ──────────────────────────────────────────────────
    print("\nStep 5: Evaluating reconstruction quality...")
    metrics = compute_metrics(dn_recon, phantom_dn)
    print()
    print_metrics_table(metrics)

    # ── Step 6: Save outputs ──────────────────────────────────────────────
    print("\nStep 6: Saving outputs...")
    os.makedirs("output", exist_ok=True)
    ref_dir = os.path.join("evaluation", "reference_outputs")
    os.makedirs(ref_dir, exist_ok=True)

    np.save(os.path.join(ref_dir, "ground_truth.npy"), phantom_dn)
    np.save(os.path.join(ref_dir, "measurements.npy"), meas_np)
    np.save(os.path.join(ref_dir, "reconstruction.npy"), dn_recon)
    np.save(os.path.join(ref_dir, "loss_history.npy"), np.array(loss_history))

    with open(os.path.join(ref_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Visualizations
    fig = plot_ri_slices(phantom_dn, title="Ground Truth Δn")
    fig.savefig(os.path.join("output", "ground_truth_slices.png"), dpi=150)

    fig = plot_ri_slices(dn_recon, title="Reconstructed Δn")
    fig.savefig(os.path.join("output", "reconstruction_slices.png"), dpi=150)

    fig = plot_measurements(meas_np)
    fig.savefig(os.path.join("output", "measurements.png"), dpi=150)

    fig = plot_comparison(phantom_dn, dn_recon)
    fig.savefig(os.path.join("output", "comparison.png"), dpi=150)

    fig = plot_loss_history(loss_history)
    fig.savefig(os.path.join("output", "loss_history.png"), dpi=150)

    print("\nDone. Outputs saved to output/ and evaluation/reference_outputs/")
    return dn_recon, metrics


if __name__ == "__main__":
    main()
