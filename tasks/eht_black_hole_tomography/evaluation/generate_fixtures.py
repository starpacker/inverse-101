"""
Generate test fixtures for BH-NeRF task.

Run from the task root directory:
    python evaluation/generate_fixtures.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.physics_model import (
    rotation_matrix, keplerian_omega, velocity_warp_coords,
    fill_unsupervised, trilinear_interpolate, volume_render,
    dft_matrix, rotation_matrix_torch,
)
from src.solvers import positional_encoding, MLP, BHNeRFModel, loss_fn_image
from src.preprocessing import load_metadata, load_observation, load_ground_truth
from src.visualization import compute_metrics, compute_image_metrics


def save_fixture(path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **kwargs)
    print(f"  Saved: {path}")


def generate_physics_model_fixtures():
    print("Generating physics_model fixtures...")
    base = "evaluation/fixtures/physics_model"

    # --- rotation_matrix ---
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 4
    R = rotation_matrix(axis, angle)
    save_fixture(f"{base}/rotation_matrix.npz",
                 input_axis=axis, input_angle=np.array(angle),
                 output_matrix=R)

    # rotation_matrix batched
    angles = np.array([0.0, np.pi / 2, np.pi])
    R_batch = rotation_matrix(axis, angles)
    save_fixture(f"{base}/rotation_matrix_batched.npz",
                 input_axis=axis, input_angles=angles,
                 output_matrix=R_batch)

    # --- keplerian_omega ---
    r_vals = np.array([6.0, 8.0, 10.0, 12.0])
    omega_vals = keplerian_omega(r_vals, spin=0.0, M=1.0)
    save_fixture(f"{base}/keplerian_omega.npz",
                 input_r=r_vals, param_spin=np.array(0.0),
                 output_Omega=omega_vals)

    # --- velocity_warp_coords ---
    np.random.seed(42)
    coords_np = np.random.randn(3, 4, 4, 5).astype(np.float32)
    Omega_np = np.abs(np.random.randn(4, 4, 5).astype(np.float32)) * 0.01
    t_geo_np = np.random.randn(4, 4, 5).astype(np.float32) * 0.1
    rot_axis_t = torch.tensor([0.0, 0.0, 1.0])

    coords_t = torch.tensor(coords_np)
    Omega_t = torch.tensor(Omega_np)
    t_geo_t = torch.tensor(t_geo_np)

    warped = velocity_warp_coords(
        coords_t, Omega_t, t_frame=100.0, t_start_obs=0.0,
        t_geo=t_geo_t, t_injection=0.0, rot_axis=rot_axis_t, GM_c3=1.0
    )
    save_fixture(f"{base}/velocity_warp.npz",
                 input_coords=coords_np, input_Omega=Omega_np,
                 input_t_geo=t_geo_np, param_t_frame=np.float32(100.0),
                 param_t_start_obs=np.float32(0.0),
                 param_t_injection=np.float32(0.0),
                 output_warped=warped.numpy())

    # --- fill_unsupervised ---
    emission_in = torch.ones(4, 4, 5)
    coords_fill = torch.tensor(coords_np)
    emission_out = fill_unsupervised(emission_in, coords_fill,
                                     rmin=1.0, rmax=5.0, z_width=2.0)
    save_fixture(f"{base}/fill_unsupervised.npz",
                 input_emission=emission_in.numpy(),
                 input_coords=coords_np,
                 param_rmin=np.float32(1.0), param_rmax=np.float32(5.0),
                 param_z_width=np.float32(2.0),
                 output_emission=emission_out.numpy())

    # --- trilinear_interpolate ---
    torch.manual_seed(42)
    volume = torch.rand(8, 8, 8)
    query_coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [-2.0, -2.0, -2.0],
        [2.0, 2.0, 2.0],
        [1.0, 0.5, -0.5],
    ])
    interp_vals = trilinear_interpolate(volume, query_coords, -4.0, 4.0)
    save_fixture(f"{base}/trilinear_interpolate.npz",
                 input_volume=volume.numpy(),
                 input_coords=query_coords.numpy(),
                 param_fov_min=np.float32(-4.0),
                 param_fov_max=np.float32(4.0),
                 output_values=interp_vals.numpy())

    # --- volume_render ---
    torch.manual_seed(42)
    emission_vr = torch.rand(4, 4, 10)
    g_vr = torch.ones(4, 4, 10) * 1.2
    dtau_vr = torch.ones(4, 4, 10) * 0.1
    Sigma_vr = torch.ones(4, 4, 10) * 100.0
    image_vr = volume_render(emission_vr, g_vr, dtau_vr, Sigma_vr)
    save_fixture(f"{base}/volume_render.npz",
                 input_emission=emission_vr.numpy(),
                 input_g=g_vr.numpy(),
                 input_dtau=dtau_vr.numpy(),
                 input_Sigma=Sigma_vr.numpy(),
                 output_image=image_vr.numpy())

    # --- dft_matrix ---
    uv = np.array([[1e9, 0.5e9], [-0.5e9, 1e9], [0.3e9, -0.7e9]])
    A = dft_matrix(uv, fov_rad=1e-10, npix=4)
    save_fixture(f"{base}/dft_matrix.npz",
                 input_uv_coords=uv,
                 param_fov_rad=np.float64(1e-10),
                 param_npix=np.int32(4),
                 output_A=A)


def generate_preprocessing_fixtures():
    print("Generating preprocessing fixtures...")
    base = "evaluation/fixtures/preprocessing"

    metadata = load_metadata("data")
    obs_data = load_observation("data")
    gt = load_ground_truth("data")

    save_fixture(f"{base}/load_observation.npz",
                 ray_coords_shape=np.array(obs_data['ray_coords'].shape),
                 Omega_shape=np.array(obs_data['Omega'].shape),
                 t_frames=obs_data['t_frames'],
                 n_frames=np.int32(len(obs_data['t_frames'])))

    save_fixture(f"{base}/load_ground_truth.npz",
                 emission_shape=np.array(gt['emission_3d'].shape),
                 images_shape=np.array(gt['images'].shape),
                 rot_axis=gt['rot_axis'])


def generate_solvers_fixtures():
    print("Generating solvers fixtures...")
    base = "evaluation/fixtures/solvers"

    # --- positional_encoding ---
    torch.manual_seed(42)
    x = torch.randn(5, 3)
    encoded = positional_encoding(x, deg=3)
    save_fixture(f"{base}/positional_encoding.npz",
                 input_x=x.numpy(), param_deg=np.int32(3),
                 output_encoded=encoded.numpy())

    # --- MLP ---
    torch.manual_seed(42)
    in_features = 3 + 2 * 3 * 3  # 3D + posenc(3)
    mlp = MLP(in_features, net_depth=4, net_width=128, out_channel=1)
    mlp.eval()
    x_mlp = torch.randn(10, in_features)
    with torch.no_grad():
        out_mlp = mlp(x_mlp)
    # Save state dict as numpy arrays
    state_items = {f"state_{k}": v.numpy() for k, v in mlp.state_dict().items()}
    save_fixture(f"{base}/mlp_forward.npz",
                 input_x=x_mlp.numpy(),
                 output_y=out_mlp.numpy(),
                 param_in_features=np.int32(in_features),
                 **state_items)

    # --- loss_fn_image ---
    torch.manual_seed(42)
    pred = torch.randn(8, 8)
    target = torch.randn(8, 8)
    loss = loss_fn_image(pred, target, sigma=1.0)
    save_fixture(f"{base}/loss_fn_image.npz",
                 input_pred=pred.numpy(),
                 input_target=target.numpy(),
                 param_sigma=np.float32(1.0),
                 output_loss=np.float32(loss.item()))


def generate_visualization_fixtures():
    print("Generating visualization fixtures...")
    base = "evaluation/fixtures/visualization"

    np.random.seed(42)
    estimate = np.random.rand(8, 8, 8).astype(np.float32)
    ground_truth = np.random.rand(8, 8, 8).astype(np.float32)
    metrics = compute_metrics(estimate, ground_truth)

    save_fixture(f"{base}/compute_metrics.npz",
                 input_estimate=estimate,
                 input_ground_truth=ground_truth,
                 output_nrmse=np.float64(metrics['nrmse']),
                 output_ncc=np.float64(metrics['ncc']),
                 output_psnr=np.float64(metrics['psnr']))

    # image metrics
    pred_movie = np.random.rand(5, 8, 8).astype(np.float32)
    true_movie = np.random.rand(5, 8, 8).astype(np.float32)
    img_metrics = compute_image_metrics(pred_movie, true_movie)
    save_fixture(f"{base}/compute_image_metrics.npz",
                 input_pred_movie=pred_movie,
                 input_true_movie=true_movie,
                 output_nrmse_image=np.float64(img_metrics['nrmse_image']),
                 output_ncc_image=np.float64(img_metrics['ncc_image']),
                 output_lightcurve_mse=np.float64(img_metrics['lightcurve_mse']))


if __name__ == '__main__':
    generate_physics_model_fixtures()
    generate_preprocessing_fixtures()
    generate_solvers_fixtures()
    generate_visualization_fixtures()
    print("\nAll fixtures generated successfully.")
