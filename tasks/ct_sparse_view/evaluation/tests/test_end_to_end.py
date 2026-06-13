"""End-to-end integration tests for sparse-view CT reconstruction."""

import os
import json
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_ground_truth, load_raw_data, load_metadata
from src.physics_model import radon_transform, filtered_back_projection
from src.solvers import tv_reconstruction
from src.visualization import compute_ncc, compute_nrmse, centre_crop


@pytest.fixture
def data_dir():
    return os.path.join(TASK_DIR, "data")


@pytest.fixture
def ref_dir():
    return os.path.join(TASK_DIR, "evaluation", "reference_outputs")


def test_data_files_exist(data_dir):
    assert os.path.exists(os.path.join(data_dir, "ground_truth.npz"))
    assert os.path.exists(os.path.join(data_dir, "raw_data.npz"))
    assert os.path.exists(os.path.join(data_dir, "meta_data.json"))


def test_reference_outputs_exist(ref_dir):
    assert os.path.exists(os.path.join(ref_dir, "reconstructions.npz"))


def test_metrics_json_exists():
    path = os.path.join(TASK_DIR, "evaluation", "metrics.json")
    assert os.path.exists(path)
    with open(path) as f:
        metrics = json.load(f)
    assert "baseline" in metrics
    assert "ncc_boundary" in metrics
    assert "nrmse_boundary" in metrics


def test_fbp_sparse_quality(data_dir):
    """FBP on sparse views should have NCC > 0.7 (reasonable but not great)."""
    phantom = load_ground_truth(data_dir)
    raw = load_raw_data(data_dir)
    meta = load_metadata(data_dir)

    fbp = filtered_back_projection(raw["sinogram_sparse"], raw["angles_sparse"],
                                    output_size=meta["image_size"])

    gt_crop = centre_crop(phantom, 0.8)
    fbp_crop = centre_crop(fbp, 0.8)
    ncc = compute_ncc(fbp_crop, gt_crop)
    assert ncc > 0.7, f"FBP NCC too low: {ncc}"


def test_tv_reconstruction_quality(data_dir):
    """TV reconstruction should achieve NCC > 0.9 and NRMSE < 0.15."""
    phantom = load_ground_truth(data_dir)
    raw = load_raw_data(data_dir)
    meta = load_metadata(data_dir)

    tv_recon, _ = tv_reconstruction(
        raw["sinogram_sparse"], raw["angles_sparse"],
        meta["image_size"], lam=0.01, n_iter=300, positivity=True,
    )

    gt_crop = centre_crop(phantom, 0.8)
    tv_crop = centre_crop(tv_recon, 0.8)

    gt_min, gt_max = gt_crop.min(), gt_crop.max()
    gt_norm = (gt_crop - gt_min) / (gt_max - gt_min)
    tv_norm = (tv_crop - gt_min) / (gt_max - gt_min)

    ncc = compute_ncc(tv_norm, gt_norm)
    nrmse = compute_nrmse(tv_norm, gt_norm)

    assert ncc > 0.9, f"TV NCC too low: {ncc}"
    assert nrmse < 0.15, f"TV NRMSE too high: {nrmse}"


def test_tv_beats_fbp(data_dir):
    """TV reconstruction should outperform FBP on sparse views."""
    phantom = load_ground_truth(data_dir)
    raw = load_raw_data(data_dir)
    meta = load_metadata(data_dir)

    fbp = filtered_back_projection(raw["sinogram_sparse"], raw["angles_sparse"],
                                    output_size=meta["image_size"])
    tv_recon, _ = tv_reconstruction(
        raw["sinogram_sparse"], raw["angles_sparse"],
        meta["image_size"], lam=0.01, n_iter=300, positivity=True,
    )

    gt_crop = centre_crop(phantom, 0.8)
    fbp_ncc = compute_ncc(centre_crop(fbp, 0.8), gt_crop)
    tv_ncc = compute_ncc(centre_crop(tv_recon, 0.8), gt_crop)

    assert tv_ncc > fbp_ncc, f"TV NCC ({tv_ncc}) should beat FBP NCC ({fbp_ncc})"


def test_reference_parity(ref_dir, data_dir):
    """Verify reference outputs match current code output."""
    ref = np.load(os.path.join(ref_dir, "reconstructions.npz"))
    raw = load_raw_data(data_dir)
    meta = load_metadata(data_dir)

    # FBP parity
    fbp = filtered_back_projection(raw["sinogram_sparse"], raw["angles_sparse"],
                                    output_size=meta["image_size"])
    np.testing.assert_allclose(fbp, ref["fbp_sparse"].squeeze(0), rtol=1e-10)
