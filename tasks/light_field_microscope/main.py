"""
main.py — Light Field Microscope Reconstruction Pipeline
=========================================================

Runs the complete LFM 3D reconstruction pipeline:
  1. Load raw LF image and ground truth from data/raw_data.npz
  2. Compute geometry (synthetic mode)
  3. Load precomputed LF operators H, Ht
  4. Run standard RL deconvolution (filter_flag=False) → aliasing artifacts
  5. Run EMS deconvolution (filter_flag=True) → artifact-free
  6. Evaluate NRMSE and PSNR against ground truth
  7. Save reconstructions to output/

Usage
-----
  cd tasks/light_field_microscope
  python main.py

Requires data/raw_data.npz and evaluation/reference_outputs/operators_H.pkl.
Run src/generate_data.py first if they do not exist.
"""

import os
import pickle
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.preprocessing import prepare_data, compute_geometry
from src.physics_model import forward_project, backward_project, compute_lf_operators
from src.solvers import ems_deconvolve
from src.visualization import compute_metrics, print_metrics_table, plot_usaf_comparison


def load_operators(ops_path_H: str, ops_path_Ht: str) -> tuple:
    """Load precomputed H, Ht operators from pickle files."""
    with open(ops_path_H, "rb") as f:
        H = pickle.load(f)
    with open(ops_path_Ht, "rb") as f:
        Ht = pickle.load(f)
    return H, Ht


def main():
    # ── Paths ──────────────────────────────────────────────────────────────
    metadata_path  = "data/meta_data"
    data_path      = "data/raw_data.npz"
    ops_path_H     = "evaluation/reference_outputs/operators_H.pkl"
    ops_path_Ht    = "evaluation/reference_outputs/operators_Ht.pkl"
    output_dir     = "output"
    ref_dir        = "evaluation/reference_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load data ──────────────────────────────────────────────────
    print("Step 1: Loading data...")
    Camera, metadata = prepare_data(metadata_path)
    data = np.load(data_path)
    lf_image     = data["lf_image"]
    ground_truth = data["ground_truth"]
    print(f"  LF image shape : {lf_image.shape}")
    print(f"  Ground truth   : {ground_truth.shape}")

    rec = metadata["reconstruction"]
    depth_range      = rec["depthRange"]
    depth_step       = rec["depthStep"]
    super_res_factor = rec["superResFactor"]
    n_iter           = rec["niter"]
    lanczos_n        = rec["lanczosWindowSize"]

    # ── Step 2: Compute geometry ───────────────────────────────────────────
    print("\nStep 2: Computing geometry (synthetic mode)...")
    img_size = np.array(lf_image.shape)
    LensletCenters, Resolution, _, _ = compute_geometry(
        Camera, np.array([]), depth_range, depth_step, super_res_factor, img_size)

    tex_size = np.ceil(img_size * np.array(Resolution["texScaleFactor"])).astype("int32")
    tex_size = tex_size + (1 - tex_size % 2)
    print(f"  Volume size: {tex_size[0]} × {tex_size[1]} × {len(Resolution['depths'])}")

    # ── Step 3: Load operators ─────────────────────────────────────────────
    print("\nStep 3: Loading LF operators...")
    if os.path.exists(ops_path_H) and os.path.exists(ops_path_Ht):
        H, Ht = load_operators(ops_path_H, ops_path_Ht)
        print("  Loaded from cache.")
    else:
        print("  Operators not found — computing (this will take 10-60 min)...")
        H, Ht = compute_lf_operators(Camera, Resolution, LensletCenters)
        os.makedirs(ref_dir, exist_ok=True)
        with open(ops_path_H, "wb") as f:
            pickle.dump(H, f, protocol=4)
        with open(ops_path_Ht, "wb") as f:
            pickle.dump(Ht, f, protocol=4)

    # ── Step 4: Standard RL deconvolution (no anti-aliasing) ──────────────
    print("\nStep 4: Running standard Richardson-Lucy (filter_flag=False)...")
    vol_rl = ems_deconvolve(
        H, Ht, lf_image, LensletCenters, Resolution, Camera,
        n_iter=n_iter, filter_flag=False, lanczos_n=lanczos_n)

    # ── Step 5: EMS deconvolution (depth-adaptive anti-aliasing) ──────────
    print("\nStep 5: Running EMS deconvolution (filter_flag=True)...")
    vol_ems = ems_deconvolve(
        H, Ht, lf_image, LensletCenters, Resolution, Camera,
        n_iter=n_iter, filter_flag=True, lanczos_n=lanczos_n)

    # ── Step 6: Evaluate ───────────────────────────────────────────────────
    print("\nStep 6: Evaluating reconstructions...")
    metrics = {
        "rl":  compute_metrics(vol_rl,  ground_truth),
        "ems": compute_metrics(vol_ems, ground_truth),
    }
    print_metrics_table(metrics)

    # ── Step 7: Save results ───────────────────────────────────────────────
    print("\nStep 7: Saving results...")
    np.save(os.path.join(output_dir, "reconstruction_rl.npy"),  vol_rl)
    np.save(os.path.join(output_dir, "reconstruction_ems.npy"), vol_ems)

    # Save reference outputs
    os.makedirs(ref_dir, exist_ok=True)
    np.save(os.path.join(ref_dir, "reconstruction_rl.npy"),  vol_rl)
    np.save(os.path.join(ref_dir, "reconstruction_ems.npy"), vol_ems)
    with open(os.path.join(ref_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Comparison figure (USAF target at z=0 depth plane)
    depths = Resolution["depths"]
    usaf_depth = metadata.get("usaf_data", {}).get("target_depth", 0.0)
    depth_idx = int(np.argmin(np.abs(np.array(depths) - usaf_depth)))
    voxel_um = float(Resolution["texRes"][0])
    fig = plot_usaf_comparison(ground_truth, vol_rl, vol_ems, depth_idx,
                                lf_image=lf_image,
                                voxel_um=voxel_um, metrics=metrics)
    fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=100, bbox_inches="tight")
    print(f"  Saved comparison plot to {output_dir}/comparison.png")

    print("\nDone.")
    return vol_rl, vol_ems, metrics


if __name__ == "__main__":
    main()
