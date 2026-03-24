"""
Generate test fixtures for all src/ modules.

Run:
    cd tasks/eht_black_hole_dynamic
    python evaluation/generate_fixtures.py
"""

import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)

FIXTURE_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures')


def generate_preprocessing_fixtures():
    """Generate fixtures for preprocessing module."""
    out_dir = os.path.join(FIXTURE_DIR, 'preprocessing')
    os.makedirs(out_dir, exist_ok=True)

    from src.preprocessing import load_observation, load_metadata

    data_dir = os.path.join(TASK_DIR, 'data')
    obs = load_observation(data_dir)
    meta = load_metadata(data_dir)

    np.savez(os.path.join(out_dir, 'load_observation.npz'),
             n_frames=obs['n_frames'],
             frame_times=obs['frame_times'],
             vis_0_shape=np.array(obs['vis'][0].shape),
             sigma_0_shape=np.array(obs['sigma'][0].shape),
             uv_0_shape=np.array(obs['uv'][0].shape))

    print(f'  Preprocessing fixtures saved to {out_dir}/')


def generate_physics_model_fixtures():
    """Generate fixtures for physics_model module."""
    out_dir = os.path.join(FIXTURE_DIR, 'physics_model')
    os.makedirs(out_dir, exist_ok=True)

    from src.physics_model import (
        DFTForwardModel, compute_visibilities, grad_vis,
        gauss_image_covariance, affine_motion_basis,
        gen_freq_comp, gen_phase_shift_matrix, calc_warp_matrix,
        product_gaussians_lem1, product_gaussians_lem2,
        apply_motion_basis,
    )

    # Small test case
    N = 8
    psize = 1e-11
    rng = np.random.default_rng(123)
    uv = rng.standard_normal((10, 2)) * 1e9
    imvec = np.abs(rng.standard_normal(N * N)) + 0.01

    model = DFTForwardModel(uv, N, psize)
    vis = compute_visibilities(imvec, model.matrix)
    F = grad_vis(imvec, model.matrix)

    np.savez(os.path.join(out_dir, 'dft_forward.npz'),
             input_uv=uv, param_N=N, param_psize=psize,
             input_imvec=imvec,
             output_vis=vis,
             output_grad=F)

    # Covariance
    cov = gauss_image_covariance(N, psize, imvec, power_dropoff=2.0, frac=0.5)
    np.savez(os.path.join(out_dir, 'gauss_covariance.npz'),
             input_imvec=imvec, param_N=N, param_psize=psize,
             output_cov=cov)

    # Motion basis
    init_x, init_y, fb_x, fb_y, initTheta = affine_motion_basis(N, psize)
    np.savez(os.path.join(out_dir, 'affine_motion_basis.npz'),
             param_N=N, param_psize=psize,
             output_init_x=init_x, output_init_y=init_y,
             output_flowbasis_x=fb_x, output_flowbasis_y=fb_y,
             output_initTheta=initTheta)

    # Warp matrix
    theta = initTheta.copy()
    warp = calc_warp_matrix(N, psize, theta, init_x, init_y, fb_x, fb_y, initTheta)
    np.savez(os.path.join(out_dir, 'warp_matrix.npz'),
             param_N=N, param_psize=psize,
             input_theta=theta,
             output_warp=warp)

    # Gaussian product lemmas
    m1 = rng.standard_normal(4)
    m2 = rng.standard_normal(4)
    S1 = np.eye(4) * 2.0 + rng.standard_normal((4, 4)) * 0.1
    S1 = S1 @ S1.T
    S2 = np.eye(4) * 3.0 + rng.standard_normal((4, 4)) * 0.1
    S2 = S2 @ S2.T

    mean1, cov1 = product_gaussians_lem1(m1, S1, m2, S2)
    np.savez(os.path.join(out_dir, 'prod_gaussians_lem1.npz'),
             input_m1=m1, input_S1=S1, input_m2=m2, input_S2=S2,
             output_mean=mean1, output_cov=cov1)

    A_mat = rng.standard_normal((3, 4))
    Sigma = np.eye(3) * 1.5
    y = rng.standard_normal(3)
    mu = rng.standard_normal(4)
    Q_mat = np.eye(4) * 2.0
    mean2, cov2 = product_gaussians_lem2(A_mat, Sigma, y, mu, Q_mat)
    np.savez(os.path.join(out_dir, 'prod_gaussians_lem2.npz'),
             input_A=A_mat, input_Sigma=Sigma, input_y=y,
             input_mu=mu, input_Q=Q_mat,
             output_mean=mean2, output_cov=cov2)

    print(f'  Physics model fixtures saved to {out_dir}/')


def generate_solvers_fixtures():
    """Generate fixtures for solvers module."""
    out_dir = os.path.join(FIXTURE_DIR, 'solvers')
    os.makedirs(out_dir, exist_ok=True)

    from src.physics_model import (
        DFTForwardModel, gauss_image_covariance, affine_motion_basis,
    )
    from src.solvers import solve_single_image

    N = 8
    psize = 1e-11
    rng = np.random.default_rng(456)

    uv = rng.standard_normal((10, 2)) * 1e9
    imvec_true = np.abs(rng.standard_normal(N * N)) + 0.01
    imvec_true = imvec_true / imvec_true.sum() * 2.0

    model = DFTForwardModel(uv, N, psize)
    vis = model.forward(imvec_true) + 0.01 * (rng.standard_normal(10) + 1j * rng.standard_normal(10))
    sigma = np.ones(10) * 0.01

    prior_mean = np.ones(N * N) * 2.0 / (N * N)
    prior_cov = gauss_image_covariance(N, psize, prior_mean, power_dropoff=2.0, frac=0.5)

    z_vec, P, z_lin = solve_single_image(prior_mean, prior_cov, model, vis, sigma)

    np.savez(os.path.join(out_dir, 'solve_single_image.npz'),
             input_prior_mean=prior_mean,
             input_vis=vis, input_sigma=sigma,
             input_uv=uv, param_N=N, param_psize=psize,
             output_z_vec=z_vec,
             output_z_vec_shape=np.array(z_vec.shape))

    print(f'  Solvers fixtures saved to {out_dir}/')


def generate_visualization_fixtures():
    """Generate fixtures for visualization module."""
    out_dir = os.path.join(FIXTURE_DIR, 'visualization')
    os.makedirs(out_dir, exist_ok=True)

    from src.visualization import compute_metrics, compute_video_metrics

    rng = np.random.default_rng(789)
    est = rng.standard_normal((8, 8))
    ref = rng.standard_normal((8, 8))

    m = compute_metrics(est, ref)
    np.savez(os.path.join(out_dir, 'compute_metrics.npz'),
             input_est=est, input_ref=ref,
             output_nrmse=m['nrmse'], output_ncc=m['ncc'])

    est_frames = [rng.standard_normal((8, 8)) for _ in range(3)]
    ref_frames = [rng.standard_normal((8, 8)) for _ in range(3)]
    vm = compute_video_metrics(est_frames, ref_frames)

    np.savez(os.path.join(out_dir, 'compute_video_metrics.npz'),
             output_avg_nrmse=vm['average']['nrmse'],
             output_avg_ncc=vm['average']['ncc'],
             output_n_frames=len(vm['per_frame']))

    print(f'  Visualization fixtures saved to {out_dir}/')


if __name__ == '__main__':
    print('Generating test fixtures ...')
    generate_preprocessing_fixtures()
    generate_physics_model_fixtures()
    generate_solvers_fixtures()
    generate_visualization_fixtures()
    print('Done!')
