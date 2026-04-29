from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
import yaml

from src.generate_data import (
    add_poisson_noise,
    compute_shift_perpendicular_um_from_sampling,
    make_linepair_object,
    place_object_at_depth,
    resolve_usaf_configuration,
)
from src.physics_model import (
    build_volume_system,
    compute_conventional_image,
    write_wave_model_config,
)
from src.preprocessing import load_metadata
from src.solvers import run_richardson_lucy
from src.visualization import (
    compute_center_line_profile,
    compute_image_metrics,
    compute_volume_slice_energy,
    extract_line_profile,
    normalize_profile,
    normalized_cross_correlation,
    normalized_root_mean_square_error,
    plot_light_field_usaf_comparison,
    plot_volume_reconstruction_demo,
)


TASK_DIR = Path(__file__).resolve().parents[2]
FIXTURE_DIR = TASK_DIR / "evaluation" / "fixtures"


def test_load_metadata_reads_json_fixture():
    metadata = load_metadata(FIXTURE_DIR / "config_metadata.json")
    assert metadata["microscope"]["M"] == 20
    assert metadata["reconstruction"]["newSpacingPx"] == 15


def test_physics_model_exports_expected_api():
    import src.physics_model as pm
    assert hasattr(pm, "build_volume_system")
    assert hasattr(pm, "compute_conventional_image")
    assert hasattr(pm, "write_wave_model_config")
    assert hasattr(pm, "build_lfm_system")
    assert hasattr(pm, "compute_native_plane_psf")


def test_write_wave_model_config_writes_expected_keys(tmp_path):
    metadata = load_metadata(FIXTURE_DIR / "config_metadata.json")
    config_path = tmp_path / "lfm_config.yaml"
    write_wave_model_config(metadata, config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["M"] == 20.0
    assert payload["lensPitch"] == 150.0
    assert payload["gridType"] == "reg"


def test_build_volume_system_raises_for_missing_config():
    with pytest.raises(FileNotFoundError):
        build_volume_system(
            config_path="/nonexistent/path/lfm_config.yaml",
            n_lenslets=15,
            new_spacing_px=15,
            depth_range_um=(0.0, 0.0),
            depth_step_um=10.0,
            theta_samples=16,
            kernel_tol=0.01,
        )


def test_compute_conventional_image_output_shape_and_normalization(tmp_path):
    from src.physics_model import load_camera_params, build_resolution
    metadata = load_metadata(FIXTURE_DIR / "config_metadata.json")
    config_path = tmp_path / "lfm_config.yaml"
    write_wave_model_config(metadata, config_path)
    camera = load_camera_params(config_path, new_spacing_px=15)
    resolution = build_resolution(camera, depth_range=(0.0, 0.0), depth_step=10.0)

    class MinimalSystem:
        pass
    system = MinimalSystem()
    system.camera = camera
    system.resolution = resolution

    object_2d = np.zeros((15, 15), dtype=np.float64)
    object_2d[7, 7] = 1.0
    image = compute_conventional_image(system, object_2d, 0.0, 16)
    assert image.shape == object_2d.shape
    assert np.all(image >= 0.0)
    assert image.sum() > 0.0


def test_run_richardson_lucy_uses_default_init():
    class StubSystem:
        tex_shape = (4, 4)
        depths = [0.0]

        def forward_project(self, volume):
            return np.ones((4, 4), dtype=np.float64)

        def backward_project(self, projection):
            return np.ones((4, 4, 1), dtype=np.float64)

    system = StubSystem()
    observation = np.ones((4, 4), dtype=np.float64)
    reconstruction = run_richardson_lucy(system, observation, iterations=3)
    assert reconstruction.shape == (4, 4, 1)
    assert np.all(reconstruction >= 0.0)


def test_make_linepair_object_and_place_object_at_depth():
    phantom = make_linepair_object((9, 9), (1.0, 1.0), lp_per_mm=100.0, window_size_um=4.0)
    assert phantom.shape == (9, 9)
    assert set(np.unique(phantom)).issubset({0.0, 1.0})

    volume = place_object_at_depth(phantom, (9, 9), np.array([0.0, 10.0]), target_depth_um=10.0)
    assert volume.shape == (9, 9, 2)
    assert np.allclose(volume[:, :, 1], phantom)


def test_shift_helpers_and_usaf_configuration_resolution():
    assert np.isclose(compute_shift_perpendicular_um_from_sampling(15, 0.5), 3.75)

    metadata = load_metadata(FIXTURE_DIR / "config_metadata.json")
    usaf = resolve_usaf_configuration(metadata)
    assert usaf["target_depths_um"].tolist() == [0.0, -30.0]
    assert usaf["profile_margin_vox"] == 5


def test_add_poisson_noise_preserves_shape_and_nonnegativity():
    rng = np.random.default_rng(0)
    image = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float64)
    noisy = add_poisson_noise(image, scale=100.0, rng=rng)
    assert noisy.shape == image.shape
    assert np.all(noisy >= 0.0)


def test_visualization_metrics_and_profiles():
    fixture = np.load(FIXTURE_DIR / "input_case_arrays.npz")
    target = fixture["input_target"]
    rl = fixture["input_rl"]

    ncc = normalized_cross_correlation(rl, target)
    nrmse = normalized_root_mean_square_error(rl, target)
    image_metrics = compute_image_metrics(rl, target)
    profile_spec = compute_center_line_profile(target, tex_res_x_um=1.0, margin_vox=1)
    profile = extract_line_profile(target, profile_spec)
    profile_norm = normalize_profile(profile)

    assert 0.0 <= ncc <= 1.0
    assert nrmse >= 0.0
    assert set(image_metrics.keys()) == {"mse", "nrmse", "ssim", "psnr"}
    assert profile.ndim == 1
    assert np.isclose(profile_norm.max(), 1.0)


def test_plot_light_field_usaf_comparison_returns_figure():
    fixture = np.load(FIXTURE_DIR / "input_case_arrays.npz")
    case = {
        "target_depth_um": -70.0,
        "gt_image": fixture["input_target"],
        "observation": fixture["input_observation"],
        "conventional_image": fixture["input_conventional"],
        "rl_image": fixture["input_rl"],
        "conventional": {"mse": 0.1, "nrmse": 0.3, "ssim": 0.7, "psnr": 18.0},
        "rl": {"mse": 0.05, "nrmse": 0.2, "ssim": 0.9, "psnr": 22.0},
        "profile": {
            "row": 4,
            "x_um": fixture["input_profile_x_um"],
            "gt": fixture["input_profile_gt"],
            "conventional": fixture["input_profile_conventional"],
            "rl": fixture["input_profile_rl"],
        },
    }
    fig = plot_light_field_usaf_comparison([case], rl_iterations=10, title="Demo")
    assert len(fig.axes) >= 5
    fig.clf()


def test_volume_slice_energy_and_plot_volume_reconstruction_demo():
    depths_um = np.array([-100.0, -75.0, -50.0, -25.0, 0.0], dtype=np.float64)
    gt_volume = np.zeros((9, 9, 5), dtype=np.float64)
    reconstruction_volume = np.zeros((9, 9, 5), dtype=np.float64)
    gt_volume[2:7, 2:7, 2] = 1.0
    reconstruction_volume[2:7, 2:7, 2] = 0.8
    reconstruction_volume[2:7, 2:7, 1] = 0.1
    observation = np.ones((9, 9), dtype=np.float64)

    energy = compute_volume_slice_energy(reconstruction_volume)
    assert energy.shape == (5,)
    assert int(np.argmax(energy)) == 2

    fig = plot_volume_reconstruction_demo(
        observation=observation,
        gt_volume=gt_volume,
        reconstruction_volume=reconstruction_volume,
        depths_um=depths_um,
        target_depth_um=-50.0,
        title="Volume Demo",
    )
    assert len(fig.axes) == 12
    fig.clf()
