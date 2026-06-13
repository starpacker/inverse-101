"""Generate reference outputs from a completed PnP-MSSN run.

Run this after main.py has completed to populate evaluation/reference_outputs/.
"""

import os
import sys
import json
import numpy as np
import scipy.io as spio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing import prepare_data
from src.physics_model import MRIForwardModel
from src.visualization import compute_metrics, compute_snr


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "reference_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    task_dir = os.path.dirname(os.path.dirname(__file__))
    orig_dir = os.getcwd()
    os.chdir(task_dir)

    img, mask, y, metadata = prepare_data("data")

    # Ground truth
    np.save(os.path.join(out_dir, "ground_truth.npy"), img)

    # Sampling mask
    np.save(os.path.join(out_dir, "sampling_mask.npy"), mask)

    # IFFT reconstruction
    model = MRIForwardModel(mask)
    ifft_recon = model.ifft_recon(y)
    np.save(os.path.join(out_dir, "ifft_recon.npy"), ifft_recon)

    # Load PnP-MSSN results from completed run
    iter_dir = "output/iterations"
    if os.path.exists(iter_dir):
        # Final reconstruction
        final_mat = spio.loadmat(
            os.path.join(iter_dir, f"iter_{metadata['num_iter']}_mat.mat"),
            squeeze_me=True,
        )
        pnp_recon = final_mat["img"]
        np.save(os.path.join(out_dir, "pnp_mssn_recon.npy"), pnp_recon)

        # SNR history from all iterations
        snr_history = []
        for it in range(1, metadata["num_iter"] + 1):
            mat = spio.loadmat(
                os.path.join(iter_dir, f"iter_{it}_mat.mat"), squeeze_me=True
            )
            snr_history.append(compute_snr(img, mat["img"]))
        np.save(os.path.join(out_dir, "snr_history.npy"), np.array(snr_history))

        # Selected iteration reconstructions for notebook
        selected_iters = [1, 5, 10, 25, 50, 100, 150, 200]
        selected_recons = {}
        for it in selected_iters:
            mat_path = os.path.join(iter_dir, f"iter_{it}_mat.mat")
            if os.path.exists(mat_path):
                mat = spio.loadmat(mat_path, squeeze_me=True)
                selected_recons[str(it)] = mat["img"]
        np.savez(os.path.join(out_dir, "selected_iterations.npz"), **selected_recons)
    elif os.path.exists("output/reconstruction.npy"):
        pnp_recon = np.load("output/reconstruction.npy")
        np.save(os.path.join(out_dir, "pnp_mssn_recon.npy"), pnp_recon)
    else:
        print("Warning: No reconstruction found. Run main.py first.")
        os.chdir(orig_dir)
        return

    # Metrics
    ifft_metrics = compute_metrics(img, ifft_recon)
    pnp_metrics = compute_metrics(img, pnp_recon)
    metrics = {
        "IFFT": {k: float(v) for k, v in ifft_metrics.items()},
        "PnP-MSSN": {k: float(v) for k, v in pnp_metrics.items()},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Reference outputs saved to {out_dir}")
    print(f"IFFT SNR: {ifft_metrics['snr_db']:.2f} dB")
    print(f"PnP-MSSN SNR: {pnp_metrics['snr_db']:.2f} dB")

    os.chdir(orig_dir)


if __name__ == "__main__":
    main()
