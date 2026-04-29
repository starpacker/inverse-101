"""End-to-end integration tests for fan-beam CT reconstruction."""

import os
import json
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_sinogram_data, load_ground_truth, load_metadata
from src.physics_model import fan_beam_geometry, fan_beam_fbp
from src.visualization import compute_ncc, compute_nrmse, centre_crop_normalize


@pytest.fixture
def data_dir():
    return TASK_DIR


@pytest.fixture
def ref_dir():
    return os.path.join(TASK_DIR, "evaluation", "reference_outputs")


DATA_DIR = os.path.join(TASK_DIR, "data")


def test_data_files_exist():
    assert os.path.exists(os.path.join(DATA_DIR, "ground_truth.npz"))
    assert os.path.exists(os.path.join(DATA_DIR, "raw_data.npz"))
    assert os.path.exists(os.path.join(DATA_DIR, "meta_data.json"))


def test_reference_outputs_exist(ref_dir):
    for name in ['recon_fbp_full.npz', 'recon_fbp_short.npz', 'recon_tv_short.npz']:
        path = os.path.join(ref_dir, name)
        assert os.path.exists(path), f"Missing {name}"


def test_metrics_json_schema():
    path = os.path.join(TASK_DIR, "evaluation", "metrics.json")
    assert os.path.exists(path)
    with open(path) as f:
        metrics = json.load(f)
    assert "baseline" in metrics
    assert "ncc_boundary" in metrics
    assert "nrmse_boundary" in metrics
    assert len(metrics["baseline"]) >= 3


def test_reference_output_shapes(ref_dir):
    for name in ['recon_fbp_full.npz', 'recon_fbp_short.npz', 'recon_tv_short.npz']:
        data = np.load(os.path.join(ref_dir, name))
        assert 'reconstruction' in data
        assert data['reconstruction'].shape[0] == 1  # batch dim


def test_reconstruction_non_negative(ref_dir):
    """TV reconstruction should be non-negative."""
    data = np.load(os.path.join(ref_dir, "recon_tv_short.npz"))
    assert np.all(data["reconstruction"] >= -0.01)


def test_fbp_full_quality(data_dir):
    """Full-scan FBP should have reasonable NCC."""
    phantom = load_ground_truth(TASK_DIR)[0]
    sino_full, _, angles_full, _, det_pos = load_sinogram_data(TASK_DIR)
    meta = load_metadata(TASK_DIR)

    N = meta['image_size']
    geo = fan_beam_geometry(N, meta['n_det'], len(angles_full.astype(np.float64)),
                             meta['source_to_isocenter_pixels'],
                             meta['isocenter_to_detector_pixels'],
                             angle_range=2 * np.pi)
    fbp = fan_beam_fbp(sino_full[0].astype(np.float64), geo,
                        filter_type='hann', cutoff=0.3)
    fbp = np.maximum(fbp, 0)

    gt_crop = centre_crop_normalize(phantom)
    fbp_crop = centre_crop_normalize(fbp)
    ncc = compute_ncc(fbp_crop, gt_crop)
    assert ncc > 0.4, f"Full-scan FBP NCC too low: {ncc}"


def test_tv_quality(data_dir):
    """TV reconstruction should achieve NCC > 0.9."""
    ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
    ref = np.load(os.path.join(ref_dir, "recon_tv_short.npz"))
    recon = ref["reconstruction"][0]
    phantom = load_ground_truth(TASK_DIR)[0]

    gt_crop = centre_crop_normalize(phantom)
    tv_crop = centre_crop_normalize(recon)
    ncc = compute_ncc(tv_crop, gt_crop)
    assert ncc > 0.9, f"TV NCC too low: {ncc}"


def test_tv_beats_fbp_short(data_dir):
    """TV should outperform short-scan FBP."""
    ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
    phantom = load_ground_truth(TASK_DIR)[0]

    fbp_short = np.load(os.path.join(ref_dir, "recon_fbp_short.npz"))["reconstruction"][0]
    tv_short = np.load(os.path.join(ref_dir, "recon_tv_short.npz"))["reconstruction"][0]

    gt_crop = centre_crop_normalize(phantom)
    fbp_ncc = compute_ncc(centre_crop_normalize(fbp_short), gt_crop)
    tv_ncc = compute_ncc(centre_crop_normalize(tv_short), gt_crop)

    assert tv_ncc > fbp_ncc, f"TV NCC ({tv_ncc}) should beat FBP NCC ({fbp_ncc})"
