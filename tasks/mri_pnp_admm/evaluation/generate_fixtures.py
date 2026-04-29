"""Generate test fixtures for PnP-ADMM CS-MRI task."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")

    # Physics model fixtures
    physics_dir = os.path.join(fixtures_dir, "physics_model")
    os.makedirs(physics_dir, exist_ok=True)

    from src.physics_model import forward_model, data_fidelity_proximal, zero_filled_recon

    np.random.seed(42)
    test_image = np.random.rand(16, 16)
    test_mask = np.zeros((16, 16))
    test_mask[::2, :] = 1  # 50% sampling

    kspace = forward_model(test_image, test_mask)
    zf = zero_filled_recon(kspace)

    vtilde = np.random.rand(16, 16)
    y = kspace + 0.01 * (np.random.randn(16, 16) + 1j * np.random.randn(16, 16))
    v = data_fidelity_proximal(vtilde, y, test_mask, alpha=2.0)

    np.savez_compressed(
        os.path.join(physics_dir, "forward_model.npz"),
        input_image=test_image, input_mask=test_mask,
        output_kspace_real=kspace.real, output_kspace_imag=kspace.imag,
        output_zerofill=zf,
    )
    np.savez_compressed(
        os.path.join(physics_dir, "proximal.npz"),
        input_vtilde=vtilde, input_y_real=y.real, input_y_imag=y.imag,
        input_mask=test_mask, input_alpha=np.array(2.0),
        output_v=v,
    )

    # Visualization fixtures
    viz_dir = os.path.join(fixtures_dir, "visualization")
    os.makedirs(viz_dir, exist_ok=True)

    from src.visualization import compute_metrics, compute_psnr

    a = np.random.RandomState(42).rand(16, 16)
    b = np.random.RandomState(43).rand(16, 16)
    metrics = compute_metrics(a, b)
    psnr_val = compute_psnr(a, b)

    np.savez_compressed(
        os.path.join(viz_dir, "metrics_test.npz"),
        input_a=a, input_b=b,
        output_ncc=np.array(metrics["ncc"]),
        output_nrmse=np.array(metrics["nrmse"]),
        output_psnr=np.array(psnr_val),
    )

    print("Fixtures generated.")


if __name__ == "__main__":
    main()
