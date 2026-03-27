"""
Reflection-Mode Optical Diffraction Tomography (rMS-FPT)
=========================================================

Main script orchestrating the full reconstruction pipeline:
  1. Load metadata and generate synthetic phantom
  2. Build the reflection BPM forward model
  3. Simulate intensity measurements (16 angles, NA=0.28)
  4. Reconstruct 3D RI via FISTA + TV
  5. Evaluate reconstruction quality
  6. Save reference outputs and visualizations

Usage
-----
    cd tasks/reflection_ODT
    python main.py
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data
from src.physics_model import ReflectionBPMConfig, ReflectionBPMForwardModel
from src.solvers import ReflectionBPMReconstructor
from src.visualization import (
    plot_ri_slices,
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
    print("Step 1: Loading metadata and generating phantom...")
    phantom_dn, metadata = prepare_data("data")
    config = ReflectionBPMConfig.from_metadata(metadata)

    print(f"  Volume shape   : {config.volume_shape}")
    print(f"  Norm. res      : ({config.res[0]:.4f}, {config.res[1]:.4f}, {config.res[2]:.4f})")
    print(f"  Voxel size     : {config.res_um} μm")
    print(f"  n0 = {config.n0}, NA_obj = {config.NA_obj}, NA_illu = {config.NA_illu}")

    # ── Step 2: Build forward model ───────────────────────────────────────
    print("\nStep 2: Building reflection BPM forward model...")
    model = ReflectionBPMForwardModel(config, device=device)
    print(f"  {model}")

    # ── Step 3: Simulate measurements ─────────────────────────────────────
    print("\nStep 3: Simulating reflection-mode ODT measurements...")
    dn_tensor = torch.tensor(phantom_dn, dtype=torch.float64, device=device)

    with torch.no_grad():
        measurements = model.forward(dn_tensor)

    meas_np = measurements.cpu().numpy()
    print(f"  {config.n_angles} angles, intensity range: "
          f"[{meas_np.min():.6f}, {meas_np.max():.6f}]")

    # ── Step 4: Reconstruct ───────────────────────────────────────────────
    print("\nStep 4: Reconstructing 3D RI via FISTA + TV...")
    reconstructor = ReflectionBPMReconstructor(
        n_iter=50,
        lr=5.0,
        tv_weight=8e-7,
        positivity=False,
        device=device,
    )

    # Pass amplitudes (|field|) to the reconstructor
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
