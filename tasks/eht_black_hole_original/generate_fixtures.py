"""Generate all test fixtures for eht_black_hole_original task."""
import os
import sys
import json
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import (
    load_observation, load_metadata, find_triangles, find_quadrangles,
    compute_closure_phases, compute_log_closure_amplitudes,
    closure_phase_sigma, closure_amplitude_sigma, prepare_data,
)
from src.physics_model import ClosureForwardModel
from src.solvers import (
    ClosurePhaseOnlySolver, ClosurePhasePlusAmpSolver, VisibilityRMLSolver,
    TVRegularizer, MaxEntropyRegularizer, L1SparsityRegularizer,
)
from src.visualization import compute_metrics

FIXTURE_BASE = os.path.join(TASK_DIR, "evaluation", "fixtures")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def gen_preprocessing_fixtures():
    d = os.path.join(FIXTURE_BASE, "preprocessing")
    ensure_dir(d)

    # load_observation
    obs = load_observation("data")
    np.savez(os.path.join(d, "load_observation.npz"),
             output_vis_corrupted=obs["vis_corrupted"],
             output_vis_true=obs["vis_true"],
             output_uv_coords=obs["uv_coords"],
             output_station_ids=obs["station_ids"],
             output_noise_std_per_vis=obs["noise_std_per_vis"])

    # load_metadata
    meta = load_metadata("data")
    with open(os.path.join(d, "load_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # find_triangles — small synthetic case (4 stations, 6 baselines)
    small_sids = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                          dtype=np.int64)
    tri, tri_st = find_triangles(small_sids, 4)
    np.savez(os.path.join(d, "find_triangles.npz"),
             input_station_ids=small_sids, input_n_stations=np.array(4),
             output_triangles=tri, output_triangle_stations=tri_st)

    # find_quadrangles
    quad, quad_st = find_quadrangles(small_sids, 4)
    np.savez(os.path.join(d, "find_quadrangles.npz"),
             input_station_ids=small_sids, input_n_stations=np.array(4),
             output_quadrangles=quad, output_quadrangle_stations=quad_st)

    # compute_closure_phases — deterministic with known vis
    rng = np.random.default_rng(123)
    vis_small = rng.standard_normal(6) + 1j * rng.standard_normal(6)
    cphases = compute_closure_phases(vis_small, tri, small_sids)
    np.savez(os.path.join(d, "compute_closure_phases.npz"),
             input_vis=vis_small, input_triangles=tri,
             input_station_ids=small_sids, output_cphases=cphases)

    # compute_log_closure_amplitudes
    log_camps = compute_log_closure_amplitudes(vis_small, quad)
    np.savez(os.path.join(d, "compute_log_closure_amplitudes.npz"),
             input_vis=vis_small, input_quadrangles=quad,
             output_log_camps=log_camps)

    # closure_phase_sigma
    noise_small = np.abs(rng.standard_normal(6)) * 0.1 + 0.01
    sigma_cp = closure_phase_sigma(vis_small, noise_small, tri)
    np.savez(os.path.join(d, "closure_phase_sigma.npz"),
             input_vis=vis_small, input_noise_std_per_vis=noise_small,
             input_triangles=tri, output_sigma_cp=sigma_cp)

    # closure_amplitude_sigma
    sigma_logca = closure_amplitude_sigma(vis_small, noise_small, quad)
    np.savez(os.path.join(d, "closure_amplitude_sigma.npz"),
             input_vis=vis_small, input_noise_std_per_vis=noise_small,
             input_quadrangles=quad, output_sigma_logca=sigma_logca)

    # prepare_data — full dataset
    obs_full, cdata, meta_full = prepare_data("data")
    np.savez(os.path.join(d, "prepare_data.npz"),
             output_cphases=cdata["cphases"],
             output_log_camps=cdata["log_camps"],
             output_sigma_cp=cdata["sigma_cp"],
             output_sigma_logca=cdata["sigma_logca"],
             output_triangles=cdata["triangles"],
             output_quadrangles=cdata["quadrangles"])
    with open(os.path.join(d, "prepare_data_meta.json"), "w") as f:
        json.dump(meta_full, f, indent=2)

    print("  preprocessing fixtures generated")


def _make_small_model(rng):
    """Build a small ClosureForwardModel for testing (N=8, 5 stations)."""
    N = 8
    pixel_size_rad = 1e-11
    station_ids = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)],
                           dtype=np.int64)
    M = len(station_ids)  # 10 baselines
    uv_coords = rng.standard_normal((M, 2)) * 1e9

    tri, _ = find_triangles(station_ids, 5)
    quad, _ = find_quadrangles(station_ids, 5)

    model = ClosureForwardModel(uv_coords, N, pixel_size_rad,
                                station_ids, tri, quad)
    params = dict(
        param_uv_coords=uv_coords,
        param_image_size=np.array(N),
        param_pixel_size_rad=np.array(pixel_size_rad),
        param_station_ids=station_ids,
        param_triangles=tri,
        param_quadrangles=quad,
    )
    return model, params, N, M


def gen_physics_model_fixtures():
    d = os.path.join(FIXTURE_BASE, "physics_model")
    ensure_dir(d)

    rng = np.random.default_rng(42)
    model, params, N, M = _make_small_model(rng)

    # init
    np.savez(os.path.join(d, "init.npz"), **params,
             output_A_shape=np.array(model.A.shape),
             output_A_row0=model.A[0, :],
             output_A_col0=model.A[:, 0])

    # forward
    image = rng.uniform(0, 1, (N, N))
    image /= image.sum()
    vis = model.forward(image)
    np.savez(os.path.join(d, "forward.npz"),
             **params, input_image=image, output_vis=vis)

    # adjoint
    vis_test = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    adj = model.adjoint(vis_test)
    np.savez(os.path.join(d, "adjoint.npz"),
             **params, input_vis=vis_test, output_image=adj)

    # dirty_image
    dirty = model.dirty_image(vis_test)
    np.savez(os.path.join(d, "dirty_image.npz"),
             **params, input_vis=vis_test, output_image=dirty)

    # psf
    psf = model.psf()
    np.savez(os.path.join(d, "psf.npz"), **params, output_psf=psf)

    # model_closure_phases
    cp = model.model_closure_phases(image)
    np.savez(os.path.join(d, "model_closure_phases.npz"),
             **params, input_image=image, output_cphases=cp)

    # closure_phase_chisq
    cphases_obs = cp + rng.standard_normal(len(cp)) * 0.1
    sigma_cp = np.full(len(cp), 0.1)
    chisq_cp = model.closure_phase_chisq(image, cphases_obs, sigma_cp)
    np.savez(os.path.join(d, "closure_phase_chisq.npz"),
             **params, input_image=image,
             input_cphases_obs=cphases_obs, input_sigma_cp=sigma_cp,
             output_chisq=np.array(chisq_cp))

    # closure_phase_chisq_grad
    grad_cp = model.closure_phase_chisq_grad(image, cphases_obs, sigma_cp)
    np.savez(os.path.join(d, "closure_phase_chisq_grad.npz"),
             **params, input_image=image,
             input_cphases_obs=cphases_obs, input_sigma_cp=sigma_cp,
             output_grad=grad_cp)

    # model_log_closure_amplitudes
    lca = model.model_log_closure_amplitudes(image)
    np.savez(os.path.join(d, "model_log_closure_amplitudes.npz"),
             **params, input_image=image, output_log_camps=lca)

    # log_closure_amp_chisq
    lca_obs = lca + rng.standard_normal(len(lca)) * 0.1
    sigma_logca = np.full(len(lca), 0.1)
    chisq_lca = model.log_closure_amp_chisq(image, lca_obs, sigma_logca)
    np.savez(os.path.join(d, "log_closure_amp_chisq.npz"),
             **params, input_image=image,
             input_log_camps_obs=lca_obs, input_sigma_logca=sigma_logca,
             output_chisq=np.array(chisq_lca))

    # log_closure_amp_chisq_grad
    grad_lca = model.log_closure_amp_chisq_grad(image, lca_obs, sigma_logca)
    np.savez(os.path.join(d, "log_closure_amp_chisq_grad.npz"),
             **params, input_image=image,
             input_log_camps_obs=lca_obs, input_sigma_logca=sigma_logca,
             output_grad=grad_lca)

    # visibility_chisq
    vis_obs = vis + (rng.standard_normal(M) + 1j * rng.standard_normal(M)) * 0.01
    noise_std = 0.01
    chisq_vis = model.visibility_chisq(image, vis_obs, noise_std)
    np.savez(os.path.join(d, "visibility_chisq.npz"),
             **params, input_image=image,
             input_vis_obs=vis_obs, input_noise_std=np.array(noise_std),
             output_chisq=np.array(chisq_vis))

    # visibility_chisq_grad
    grad_vis = model.visibility_chisq_grad(image, vis_obs, noise_std)
    np.savez(os.path.join(d, "visibility_chisq_grad.npz"),
             **params, input_image=image,
             input_vis_obs=vis_obs, input_noise_std=np.array(noise_std),
             output_grad=grad_vis)

    print("  physics_model fixtures generated")


def gen_solvers_fixtures():
    d = os.path.join(FIXTURE_BASE, "solvers")
    ensure_dir(d)

    rng = np.random.default_rng(99)
    N = 8

    # ── Regularizer fixtures (deterministic) ──────────────────────────────
    image = rng.uniform(0.01, 1.0, (N, N))

    tv = TVRegularizer(epsilon=1e-6)
    tv_val, tv_grad = tv.value_and_grad(image)
    np.savez(os.path.join(d, "tv_regularizer.npz"),
             input_image=image, config_epsilon=np.array(1e-6),
             output_val=np.array(tv_val), output_grad=tv_grad)

    mem = MaxEntropyRegularizer(epsilon=1e-12)
    mem_val, mem_grad = mem.value_and_grad(image)
    np.savez(os.path.join(d, "max_entropy_regularizer.npz"),
             input_image=image, config_epsilon=np.array(1e-12),
             output_val=np.array(mem_val), output_grad=mem_grad)

    l1 = L1SparsityRegularizer(epsilon=1e-8)
    l1_val, l1_grad = l1.value_and_grad(image)
    np.savez(os.path.join(d, "l1_sparsity_regularizer.npz"),
             input_image=image, config_epsilon=np.array(1e-8),
             output_val=np.array(l1_val), output_grad=l1_grad)

    # ── Solver fixtures (small problem) ───────────────────────────────────
    pixel_size_rad = 1e-11
    station_ids = np.array([[i, j] for i in range(5) for j in range(i + 1, 5)],
                           dtype=np.int64)
    M = len(station_ids)
    uv_coords = rng.standard_normal((M, 2)) * 1e9

    tri, _ = find_triangles(station_ids, 5)
    quad, _ = find_quadrangles(station_ids, 5)
    model = ClosureForwardModel(uv_coords, N, pixel_size_rad,
                                station_ids, tri, quad)

    gt = rng.uniform(0, 1, (N, N))
    gt /= gt.sum()
    vis_true = model.forward(gt)
    noise_std = 0.01
    vis_noisy = vis_true + noise_std * (
        rng.standard_normal(M) + 1j * rng.standard_normal(M)
    ) / np.sqrt(2)

    from src.preprocessing import compute_closure_phases, compute_log_closure_amplitudes
    cphases = compute_closure_phases(vis_noisy, tri, station_ids)
    log_camps = compute_log_closure_amplitudes(vis_noisy, quad)
    sigma_cp = np.full(len(cphases), 0.1)
    sigma_logca = np.full(len(log_camps), 0.1)

    closure_data = dict(cphases=cphases, log_camps=log_camps,
                        sigma_cp=sigma_cp, sigma_logca=sigma_logca,
                        triangles=tri, quadrangles=quad)

    solver_params = dict(
        param_uv_coords=uv_coords,
        param_image_size=np.array(N),
        param_pixel_size_rad=np.array(pixel_size_rad),
        param_station_ids=station_ids,
        param_triangles=tri,
        param_quadrangles=quad,
    )

    # ClosurePhaseOnlySolver
    x_cp = ClosurePhaseOnlySolver(
        regularizers=[(100.0, TVRegularizer())],
        alpha_cp=50.0, n_iter=100,
    ).reconstruct(model, closure_data)
    np.savez(os.path.join(d, "closure_phase_only_solver.npz"),
             **solver_params,
             input_cphases=cphases, input_sigma_cp=sigma_cp,
             config_alpha_cp=np.array(50.0),
             config_lambda_tv=np.array(100.0),
             config_n_iter=np.array(100),
             output_image=x_cp)

    # ClosurePhasePlusAmpSolver
    x_cpca = ClosurePhasePlusAmpSolver(
        regularizers=[(100.0, TVRegularizer())],
        alpha_cp=50.0, alpha_ca=50.0, n_iter=100,
    ).reconstruct(model, closure_data)
    np.savez(os.path.join(d, "closure_phase_plus_amp_solver.npz"),
             **solver_params,
             input_cphases=cphases, input_log_camps=log_camps,
             input_sigma_cp=sigma_cp, input_sigma_logca=sigma_logca,
             config_alpha_cp=np.array(50.0), config_alpha_ca=np.array(50.0),
             config_lambda_tv=np.array(100.0),
             config_n_iter=np.array(100),
             output_image=x_cpca)

    # VisibilityRMLSolver
    x_vis = VisibilityRMLSolver(
        regularizers=[(100.0, TVRegularizer())],
        n_iter=100,
    ).reconstruct(model, vis_noisy, noise_std)
    np.savez(os.path.join(d, "visibility_rml_solver.npz"),
             **solver_params,
             input_vis=vis_noisy,
             input_noise_std=np.array(noise_std),
             config_lambda_tv=np.array(100.0),
             config_n_iter=np.array(100),
             output_image=x_vis)

    print("  solvers fixtures generated")


def gen_visualization_fixtures():
    d = os.path.join(FIXTURE_BASE, "visualization")
    ensure_dir(d)

    rng = np.random.default_rng(77)
    N = 16
    gt = rng.uniform(0, 1, (N, N))
    gt /= gt.sum()
    est = gt + rng.standard_normal((N, N)) * 0.01
    est = np.maximum(est, 0)

    m = compute_metrics(est, gt)
    np.savez(os.path.join(d, "compute_metrics.npz"),
             input_estimate=est, input_ground_truth=gt,
             output_nrmse=np.array(m["nrmse"]),
             output_ncc=np.array(m["ncc"]),
             output_dynamic_range=np.array(m["dynamic_range"]))

    print("  visualization fixtures generated")


if __name__ == "__main__":
    print("Generating fixtures...")
    gen_preprocessing_fixtures()
    gen_physics_model_fixtures()
    gen_solvers_fixtures()
    gen_visualization_fixtures()
    print("Done!")
