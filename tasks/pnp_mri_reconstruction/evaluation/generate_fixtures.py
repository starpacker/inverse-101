"""Generate test fixtures for unit tests.

Creates .npz fixture files for each module's tests.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import load_observation, load_metadata, normalize_image
from src.physics_model import MRIForwardModel
from src.visualization import compute_snr, compute_metrics


def main():
    task_dir = os.path.dirname(os.path.dirname(__file__))
    orig_dir = os.getcwd()
    os.chdir(task_dir)

    fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")

    # ---- Preprocessing fixtures ----
    pp_dir = os.path.join(fixture_dir, "preprocessing")
    os.makedirs(pp_dir, exist_ok=True)

    obs = load_observation("data")
    metadata = load_metadata("data")
    img_raw = obs["img"]
    img_norm = normalize_image(img_raw)

    np.savez(
        os.path.join(pp_dir, "load_observation.npz"),
        output_img_shape=np.array(img_raw.shape),
        output_img_dtype=str(img_raw.dtype),
        output_img_min=img_raw.min(),
        output_img_max=img_raw.max(),
    )
    # Use a self-contained test: normalizing the patch alone
    patch_raw = img_raw[:8, :8].copy()
    patch_norm = normalize_image(patch_raw)
    np.savez(
        os.path.join(pp_dir, "normalize_image.npz"),
        input_img=patch_raw,
        output_img=patch_norm,
    )
    with open(os.path.join(pp_dir, "load_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ---- Physics model fixtures ----
    pm_dir = os.path.join(fixture_dir, "physics_model")
    os.makedirs(pm_dir, exist_ok=True)

    image_size = np.array([320, 320])
    mask = MRIForwardModel.generate_mask(image_size, 36)
    model = MRIForwardModel(mask)

    np.savez(
        os.path.join(pm_dir, "generate_mask.npz"),
        input_image_size=image_size,
        input_num_lines=36,
        output_mask_shape=np.array(mask.shape),
        output_mask_sum=mask.sum(),
        output_mask_dtype=str(mask.dtype),
    )

    # Forward/adjoint on a small test image
    test_img = img_norm
    y = model.forward(test_img)
    x_adj = model.adjoint(y)
    g, cost = model.grad(test_img, y)

    np.savez(
        os.path.join(pm_dir, "forward.npz"),
        input_image=test_img[:16, :16],
        output_y_shape=np.array(y.shape),
        output_y_nonzero=int(np.count_nonzero(y)),
    )

    np.savez(
        os.path.join(pm_dir, "grad.npz"),
        output_grad_shape=np.array(g.shape),
        output_cost_at_true=float(cost),
    )

    # IFFT reconstruction
    ifft_recon = model.ifft_recon(y)
    np.savez(
        os.path.join(pm_dir, "ifft_recon.npz"),
        output_shape=np.array(ifft_recon.shape),
        output_snr=compute_snr(img_norm, ifft_recon),
    )

    # ---- Solver fixtures (metrics only, no TF) ----
    sol_dir = os.path.join(fixture_dir, "solvers")
    os.makedirs(sol_dir, exist_ok=True)

    # Use existing reference outputs if available
    ref_dir = os.path.join(os.path.dirname(__file__), "reference_outputs")
    if os.path.exists(os.path.join(ref_dir, "pnp_mssn_recon.npy")):
        pnp_recon = np.load(os.path.join(ref_dir, "pnp_mssn_recon.npy"))
        pnp_metrics = compute_metrics(img_norm, pnp_recon)
        np.savez(
            os.path.join(sol_dir, "pnp_pgm.npz"),
            output_recon_shape=np.array(pnp_recon.shape),
            output_snr=pnp_metrics["snr_db"],
            output_nrmse=pnp_metrics["nrmse"],
            output_ncc=pnp_metrics["ncc"],
        )

    print(f"Fixtures saved to {fixture_dir}")
    os.chdir(orig_dir)


if __name__ == "__main__":
    main()
