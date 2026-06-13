"""
Generate test fixtures for all evaluation tests.

Run from the task directory:
    cd tasks/eht_black_hole_UQ
    python evaluation/generate_fixtures.py
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

os.makedirs(os.path.join(FIXTURE_BASE, "preprocessing"), exist_ok=True)
os.makedirs(os.path.join(FIXTURE_BASE, "physics_model"), exist_ok=True)
os.makedirs(os.path.join(FIXTURE_BASE, "solvers"), exist_ok=True)
os.makedirs(os.path.join(FIXTURE_BASE, "visualization"), exist_ok=True)


def generate_preprocessing_fixtures():
    print("Generating preprocessing fixtures...")
    from src.preprocessing import (
        load_observation, load_ground_truth, load_metadata,
        extract_closure_indices, compute_nufft_params, build_prior_image
    )

    # load_observation
    obs_data = load_observation("data")
    np.savez(os.path.join(FIXTURE_BASE, "preprocessing", "load_observation.npz"),
             output_vis_shape=np.array(obs_data['vis'].shape),
             output_uv_shape=np.array(obs_data['uv_coords'].shape),
             output_vis_first5=obs_data['vis'][:5])

    # extract_closure_indices
    obs = obs_data['obs']
    closure = extract_closure_indices(obs)
    closure_fixture = {
        'output_n_cphase': np.array(len(closure['cphase_ind_list'][0])),
        'output_n_camp': np.array(len(closure['camp_ind_list'][0])),
    }
    for i in range(3):
        closure_fixture[f'output_cphase_ind{i}'] = closure['cphase_ind_list'][i]
        closure_fixture[f'output_cphase_sign{i}'] = closure['cphase_sign_list'][i]
    for i in range(4):
        closure_fixture[f'output_camp_ind{i}'] = closure['camp_ind_list'][i]
    np.savez(os.path.join(FIXTURE_BASE, "preprocessing", "extract_closure_indices.npz"),
             **closure_fixture)

    # compute_nufft_params
    nufft = compute_nufft_params(obs, 32, 160.0)
    np.savez(os.path.join(FIXTURE_BASE, "preprocessing", "compute_nufft_params.npz"),
             output_ktraj_shape=np.array(nufft['ktraj_vis'].shape),
             output_pulsefac_shape=np.array(nufft['pulsefac_vis'].shape),
             output_ktraj_vis=nufft['ktraj_vis'].numpy(),
             output_pulsefac_vis=nufft['pulsefac_vis'].numpy())

    # build_prior_image
    prior, flux_const = build_prior_image(obs, 32, 160.0)
    np.savez(os.path.join(FIXTURE_BASE, "preprocessing", "build_prior_image.npz"),
             output_prior=prior,
             output_flux_const=np.array(flux_const))

    # load_ground_truth
    gt = load_ground_truth("data", npix=32, fov_uas=160.0)
    np.savez(os.path.join(FIXTURE_BASE, "preprocessing", "load_ground_truth.npz"),
             output_image=gt)

    print("  Done.")
    return obs_data, closure, nufft, prior, flux_const


def generate_physics_model_fixtures(obs_data, closure, nufft):
    print("Generating physics_model fixtures...")
    from src.physics_model import (
        NUFFTForwardModel,
        Loss_angle_diff, Loss_logca_diff2, Loss_vis_diff, Loss_logamp_diff,
        Loss_visamp_diff, Loss_l1, Loss_TSV, Loss_TV, Loss_flux, Loss_center,
        Loss_cross_entropy,
    )

    device = torch.device("cpu")

    # Build model
    model = NUFFTForwardModel(
        32, nufft['ktraj_vis'], nufft['pulsefac_vis'],
        [torch.tensor(a, dtype=torch.long) for a in closure['cphase_ind_list']],
        [torch.tensor(a, dtype=torch.float32) for a in closure['cphase_sign_list']],
        [torch.tensor(a, dtype=torch.long) for a in closure['camp_ind_list']],
        device
    )

    # Test image: simple Gaussian
    np.random.seed(42)
    test_img = np.abs(np.random.randn(2, 32, 32)).astype(np.float32)
    images = torch.tensor(test_img)

    vis, visamp, cphase, logcamp = model(images)

    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "nufft_forward.npz"),
             input_images=test_img,
             output_vis=vis.detach().numpy(),
             output_vis_shape=np.array(vis.shape),
             output_visamp_shape=np.array(visamp.shape),
             output_cphase=cphase.detach().numpy(),
             output_cphase_shape=np.array(cphase.shape),
             output_logcamp=logcamp.detach().numpy(),
             output_logcamp_shape=np.array(logcamp.shape))

    # Loss function fixtures
    obs = obs_data['obs']

    # Loss_angle_diff
    sigma_cp = np.array(closure['cphase_data']['sigmacp'][:10], dtype=np.float32)
    true_cp = np.random.randn(2, 10).astype(np.float32) * 30
    pred_cp = true_cp + np.random.randn(2, 10).astype(np.float32) * 5
    loss_fn = Loss_angle_diff(sigma_cp, device)
    loss_val = loss_fn(torch.tensor(true_cp), torch.tensor(pred_cp))
    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "loss_angle_diff.npz"),
             input_sigma=sigma_cp, input_true=true_cp, input_pred=pred_cp,
             output_loss=loss_val.detach().numpy())

    # Loss_logca_diff2
    sigma_ca = np.abs(np.random.randn(10).astype(np.float32)) + 0.1
    true_ca = np.random.randn(2, 10).astype(np.float32)
    pred_ca = true_ca + np.random.randn(2, 10).astype(np.float32) * 0.1
    loss_fn2 = Loss_logca_diff2(sigma_ca, device)
    loss_val2 = loss_fn2(torch.tensor(true_ca), torch.tensor(pred_ca))
    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "loss_logca_diff2.npz"),
             input_sigma=sigma_ca, input_true=true_ca, input_pred=pred_ca,
             output_loss=loss_val2.detach().numpy())

    # Image prior losses
    test_img_batch = torch.tensor(test_img)  # (2, 32, 32)
    l1_val = Loss_l1(test_img_batch)
    tsv_val = Loss_TSV(test_img_batch)
    flux_val = Loss_flux(1.0)(test_img_batch)
    center_val = Loss_center(device, center=15.5, dim=32)(test_img_batch)

    tv_val = Loss_TV(test_img_batch)
    prior_im = torch.abs(torch.randn(32, 32)) + 0.01
    ce_val = Loss_cross_entropy(prior_im, test_img_batch)

    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "loss_priors.npz"),
             input_image=test_img,
             output_l1=l1_val.detach().numpy(),
             output_tsv=tsv_val.detach().numpy(),
             output_tv=tv_val.detach().numpy(),
             config_flux=np.array(1.0),
             output_flux=flux_val.detach().numpy(),
             output_center=center_val.detach().numpy(),
             input_prior_im=prior_im.numpy(),
             output_cross_entropy=ce_val.detach().numpy())

    # Loss_vis_diff
    sigma_vis = np.abs(np.random.randn(20).astype(np.float32)) + 0.1
    vis_true_2d = np.random.randn(2, 20).astype(np.float32)
    vis_pred_batch = np.random.randn(4, 2, 20).astype(np.float32)
    loss_vis_fn = Loss_vis_diff(sigma_vis, device)
    loss_vis_val = loss_vis_fn(torch.tensor(vis_true_2d), torch.tensor(vis_pred_batch))
    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "loss_vis_diff.npz"),
             input_sigma=sigma_vis, input_true=vis_true_2d,
             input_pred=vis_pred_batch,
             output_loss=loss_vis_val.detach().numpy())

    # Loss_logamp_diff
    sigma_amp = np.abs(np.random.randn(20).astype(np.float32)) + 0.1
    amp_true = np.abs(np.random.randn(4, 20).astype(np.float32)) + 0.1
    amp_pred = np.abs(amp_true + np.random.randn(4, 20).astype(np.float32) * 0.01) + 0.01
    loss_logamp_fn = Loss_logamp_diff(sigma_amp, device)
    loss_logamp_val = loss_logamp_fn(torch.tensor(amp_true), torch.tensor(amp_pred))
    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "loss_logamp_diff.npz"),
             input_sigma=sigma_amp, input_true=amp_true, input_pred=amp_pred,
             output_loss=loss_logamp_val.detach().numpy())

    # Loss_visamp_diff
    sigma_va = np.abs(np.random.randn(20).astype(np.float32)) + 0.1
    va_true = np.abs(np.random.randn(4, 20).astype(np.float32)) + 0.1
    va_pred = va_true + np.random.randn(4, 20).astype(np.float32) * 0.1
    loss_va_fn = Loss_visamp_diff(sigma_va, device)
    loss_va_val = loss_va_fn(torch.tensor(va_true), torch.tensor(va_pred))
    np.savez(os.path.join(FIXTURE_BASE, "physics_model", "loss_visamp_diff.npz"),
             input_sigma=sigma_va, input_true=va_true, input_pred=va_pred,
             output_loss=loss_va_val.detach().numpy())

    print("  Done.")


def generate_visualization_fixtures():
    print("Generating visualization fixtures...")
    from src.visualization import compute_metrics, compute_uq_metrics

    np.random.seed(42)
    gt = np.abs(np.random.randn(32, 32)).astype(np.float64) + 0.01
    estimate = gt + np.random.randn(32, 32) * 0.1

    metrics = compute_metrics(estimate, gt)
    np.savez(os.path.join(FIXTURE_BASE, "visualization", "compute_metrics.npz"),
             input_estimate=estimate,
             input_ground_truth=gt,
             output_nrmse=np.array(metrics['nrmse']),
             output_ncc=np.array(metrics['ncc']),
             output_dynamic_range=np.array(metrics['dynamic_range']))

    # UQ metrics
    std = np.abs(np.random.randn(32, 32)) * 0.2 + 0.05
    uq_metrics = compute_uq_metrics(estimate, std, gt)
    np.savez(os.path.join(FIXTURE_BASE, "visualization", "compute_uq_metrics.npz"),
             input_mean=estimate,
             input_std=std,
             input_gt=gt,
             output_calibration=np.array(uq_metrics['calibration']),
             output_mean_uncertainty=np.array(uq_metrics['mean_uncertainty']))

    print("  Done.")


def generate_solvers_fixtures():
    print("Generating solvers fixtures...")
    from src.solvers import RealNVP, Img_logscale, ActNorm, AffineCoupling, Flow

    # ── RealNVP: fixed weights + fixed input → deterministic output ──
    # Use small model (ndim=64, n_flow=4) for fixture speed
    torch.manual_seed(123)
    model = RealNVP(64, n_flow=4, affine=True, seqfrac=4)
    model.eval()

    # Save the state_dict as numpy arrays for portability
    state_dict = {k: v.numpy() for k, v in model.state_dict().items()}

    torch.manual_seed(456)
    z_input = torch.randn(4, 64)

    with torch.no_grad():
        x_rev, logdet_rev = model.reverse(z_input)
        z_fwd, logdet_fwd = model.forward(x_rev)

    np.savez(os.path.join(FIXTURE_BASE, "solvers", "realnvp.npz"),
             input_z=z_input.numpy(),
             output_reverse_x=x_rev.numpy(),
             output_reverse_logdet=logdet_rev.numpy(),
             output_forward_z=z_fwd.numpy(),
             output_forward_logdet=logdet_fwd.numpy())
    # Save state_dict separately (too many keys for savez naming)
    torch.save(model.state_dict(),
               os.path.join(FIXTURE_BASE, "solvers", "realnvp_state_dict.pt"))

    # ── AffineCoupling: fixed weights → deterministic output ──
    torch.manual_seed(789)
    coupling = AffineCoupling(64, seqfrac=4, affine=True, batch_norm=True)
    coupling.eval()
    x_input = torch.randn(4, 64)

    with torch.no_grad():
        z_coup, logdet_coup = coupling(x_input)

    np.savez(os.path.join(FIXTURE_BASE, "solvers", "affine_coupling.npz"),
             input_x=x_input.numpy(),
             output_z=z_coup.numpy(),
             output_logdet=logdet_coup.numpy())
    torch.save(coupling.state_dict(),
               os.path.join(FIXTURE_BASE, "solvers", "affine_coupling_state_dict.pt"))

    # ── Img_logscale: deterministic output ──
    logscale = Img_logscale(scale=0.5)
    output_val = torch.exp(logscale.forward()).item()
    np.savez(os.path.join(FIXTURE_BASE, "solvers", "img_logscale.npz"),
             config_scale=np.array(0.5),
             output_exp_logscale=np.array(output_val))

    print("  Done.")


if __name__ == "__main__":
    obs_data, closure, nufft, prior, flux_const = generate_preprocessing_fixtures()
    generate_physics_model_fixtures(obs_data, closure, nufft)
    generate_solvers_fixtures()
    generate_visualization_fixtures()
    print("\nAll fixtures generated successfully!")
