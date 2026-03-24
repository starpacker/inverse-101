"""
Generate test fixtures for eht_black_hole_feature_extraction.

Usage:
    cd tasks/eht_black_hole_feature_extraction
    python evaluation/generate_fixtures.py
"""

import os
import sys
import numpy as np
import torch

TASK_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from src.preprocessing import (
    load_observation, load_metadata, extract_closure_indices,
    compute_nufft_params, estimate_flux,
)
from src.physics_model import (
    SimpleCrescentParam2Img, SimpleCrescentNuisanceParam2Img,
    NUFFTForwardModel, Loss_angle_diff, Loss_logca_diff2,
)
from src.solvers import RealNVP, AlphaDPISolver
from src.visualization import compute_feature_metrics


def generate_preprocessing_fixtures(fix_dir):
    """Generate fixtures for preprocessing functions."""
    os.makedirs(fix_dir, exist_ok=True)

    obs_data = load_observation("data")
    metadata = load_metadata("data")
    obs = obs_data["obs"]

    # load_observation fixture
    np.savez(os.path.join(fix_dir, "load_observation.npz"),
             vis_shape=np.array(obs_data['vis'].shape),
             vis_dtype=str(obs_data['vis'].dtype),
             uv_shape=np.array(obs_data['uv_coords'].shape),
             vis_sample=obs_data['vis'][:5],
             sigma_sample=obs_data['vis_sigma'][:5])

    # extract_closure_indices fixture
    closure_indices = extract_closure_indices(obs)
    np.savez(os.path.join(fix_dir, "extract_closure_indices.npz"),
             n_cphase=len(closure_indices['cphase_data']['cphase']),
             n_camp=len(closure_indices['camp_data']['camp']),
             cphase_ind0_sample=closure_indices['cphase_ind_list'][0][:10],
             cphase_sign0_sample=closure_indices['cphase_sign_list'][0][:10],
             camp_ind0_sample=closure_indices['camp_ind_list'][0][:10])

    # compute_nufft_params fixture
    nufft_params = compute_nufft_params(obs, metadata['npix'], metadata['fov_uas'])
    np.savez(os.path.join(fix_dir, "compute_nufft_params.npz"),
             ktraj_shape=np.array(nufft_params['ktraj_vis'].shape),
             pulsefac_shape=np.array(nufft_params['pulsefac_vis'].shape),
             ktraj_sample=nufft_params['ktraj_vis'][0, :, :5].numpy(),
             pulsefac_sample=nufft_params['pulsefac_vis'][:, :5].numpy())

    # estimate_flux fixture
    flux = estimate_flux(obs)
    np.savez(os.path.join(fix_dir, "estimate_flux.npz"),
             flux_const=flux)

    print(f"  Preprocessing fixtures saved to {fix_dir}")


def generate_physics_model_fixtures(fix_dir):
    """Generate fixtures for physics model functions."""
    os.makedirs(fix_dir, exist_ok=True)

    torch.manual_seed(42)

    # SimpleCrescentParam2Img fixture
    model = SimpleCrescentParam2Img(npix=64, fov=120.0)
    params = torch.rand(4, 4)
    img = model.forward(params)
    np.savez(os.path.join(fix_dir, "simple_crescent.npz"),
             input_params=params.numpy(),
             output_img=img.detach().numpy(),
             output_shape=np.array(img.shape))

    # SimpleCrescentNuisanceParam2Img fixture
    model2 = SimpleCrescentNuisanceParam2Img(npix=64, n_gaussian=2, fov=120.0)
    params2 = torch.rand(4, model2.nparams)
    img2 = model2.forward(params2)
    np.savez(os.path.join(fix_dir, "simple_crescent_nuisance.npz"),
             input_params=params2.numpy(),
             output_img=img2.detach().numpy(),
             output_shape=np.array(img2.shape),
             nparams=model2.nparams)

    # NUFFTForwardModel fixture (requires data)
    obs_data = load_observation("data")
    obs = obs_data["obs"]
    metadata = load_metadata("data")
    closure_indices = extract_closure_indices(obs)
    nufft_params = compute_nufft_params(obs, metadata['npix'], metadata['fov_uas'])

    cphase_ind_torch = [torch.tensor(a, dtype=torch.long)
                        for a in closure_indices["cphase_ind_list"]]
    cphase_sign_torch = [torch.tensor(a, dtype=torch.float32)
                         for a in closure_indices["cphase_sign_list"]]
    camp_ind_torch = [torch.tensor(a, dtype=torch.long)
                      for a in closure_indices["camp_ind_list"]]

    device = torch.device('cpu')
    forward_model = NUFFTForwardModel(
        64, nufft_params["ktraj_vis"], nufft_params["pulsefac_vis"],
        cphase_ind_torch, cphase_sign_torch, camp_ind_torch, device
    )

    torch.manual_seed(42)
    test_img = model2.forward(torch.rand(2, model2.nparams))
    vis, visamp, cphase, logcamp = forward_model(test_img)
    np.savez(os.path.join(fix_dir, "nufft_forward.npz"),
             input_img=test_img.detach().numpy(),
             output_vis=vis.detach().numpy(),
             output_visamp=visamp.detach().numpy(),
             output_cphase=cphase.detach().numpy(),
             output_logcamp=logcamp.detach().numpy())

    # Loss functions fixture
    sigma_cp = torch.Tensor(closure_indices['cphase_data']['sigmacp']).to(device)
    sigma_ca = torch.Tensor(closure_indices['logcamp_data']['sigmaca']).to(device)
    cphase_true = torch.Tensor(np.array(closure_indices['cphase_data']['cphase'])).to(device)
    logcamp_true = torch.Tensor(np.array(closure_indices['logcamp_data']['camp'])).to(device)

    Loss_cp = Loss_angle_diff(closure_indices['cphase_data']['sigmacp'], device)
    Loss_lca = Loss_logca_diff2(closure_indices['logcamp_data']['sigmaca'], device)

    loss_cp = Loss_cp(cphase_true, cphase)
    loss_lca = Loss_lca(logcamp_true, logcamp)
    np.savez(os.path.join(fix_dir, "loss_functions.npz"),
             output_loss_cphase=loss_cp.detach().numpy(),
             output_loss_logca=loss_lca.detach().numpy())

    print(f"  Physics model fixtures saved to {fix_dir}")


def generate_solver_fixtures(fix_dir):
    """Generate fixtures for solver components."""
    os.makedirs(fix_dir, exist_ok=True)

    torch.manual_seed(42)

    # RealNVP fixture
    nparams = 16
    flow = RealNVP(nparams, n_flow=4, affine=True, seqfrac=1 / 16,
                    permute='random', batch_norm=True)
    flow.eval()

    z = torch.randn(8, nparams)
    with torch.no_grad():
        out, logdet = flow.forward(z)
        z_back, logdet_back = flow.reverse(out)

    np.savez(os.path.join(fix_dir, "realnvp.npz"),
             input_z=z.numpy(),
             output_forward=out.numpy(),
             output_logdet=logdet.numpy(),
             output_reverse=z_back.numpy(),
             output_logdet_reverse=logdet_back.numpy())
    torch.save(flow.state_dict(), os.path.join(fix_dir, "realnvp_state_dict.pt"))

    print(f"  Solver fixtures saved to {fix_dir}")


def generate_visualization_fixtures(fix_dir):
    """Generate fixtures for visualization functions."""
    os.makedirs(fix_dir, exist_ok=True)

    np.random.seed(42)
    n_samples = 1000
    n_params = 4

    params = np.random.randn(n_samples, n_params)
    params[:, 0] = params[:, 0] * 2 + 44  # diameter
    params[:, 1] = params[:, 1] * 1 + 11  # width
    params[:, 2] = np.clip(params[:, 2] * 0.1 + 0.5, 0, 1)  # asymmetry
    params[:, 3] = params[:, 3] * 5 - 90  # PA
    gt = np.array([44.0, 11.36, 0.5, -90.5])

    weights = np.random.dirichlet(np.ones(n_samples))

    metrics = compute_feature_metrics(params, gt, weights,
                                       ['diameter', 'width', 'asymmetry', 'PA'])

    np.savez(os.path.join(fix_dir, "feature_metrics.npz"),
             input_params=params,
             input_gt=gt,
             input_weights=weights,
             output_n_params=metrics['n_params'])

    print(f"  Visualization fixtures saved to {fix_dir}")


if __name__ == "__main__":
    os.chdir(TASK_DIR)
    print("Generating fixtures for eht_black_hole_feature_extraction...")

    generate_preprocessing_fixtures("evaluation/fixtures/preprocessing")
    generate_physics_model_fixtures("evaluation/fixtures/physics_model")
    generate_solver_fixtures("evaluation/fixtures/solvers")
    generate_visualization_fixtures("evaluation/fixtures/visualization")

    print("Done.")
