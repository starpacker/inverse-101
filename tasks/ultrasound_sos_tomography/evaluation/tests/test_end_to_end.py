"""End-to-end test: run main.py and check outputs exist with expected quality."""

import os
import json
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)


def test_reference_outputs_exist():
    """Check that reference outputs were generated."""
    ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
    assert os.path.isfile(os.path.join(ref_dir, "reconstructions.npz"))


def test_metrics_json_exists():
    metrics_path = os.path.join(TASK_DIR, "evaluation", "metrics.json")
    assert os.path.isfile(metrics_path)


def test_metrics_json_schema():
    """metrics.json should have the required structure."""
    metrics_path = os.path.join(TASK_DIR, "evaluation", "metrics.json")
    with open(metrics_path) as f:
        m = json.load(f)
    assert "baseline" in m
    assert "ncc_boundary" in m
    assert "nrmse_boundary" in m
    assert len(m["baseline"]) >= 2
    for entry in m["baseline"]:
        assert "method" in entry
        assert "ncc_vs_ref" in entry
        assert "nrmse_vs_ref" in entry


def test_reference_reconstruction_quality():
    """The TV-PDHG reconstruction should meet quality thresholds."""
    from src.preprocessing import load_ground_truth
    from src.visualization import compute_ncc, compute_nrmse, centre_crop

    data_dir = os.path.join(TASK_DIR, "data")
    ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")

    gt = load_ground_truth(data_dir)
    delta_s_gt = gt["slowness_perturbation"]
    recon = np.load(os.path.join(ref_dir, "reconstructions.npz"))
    delta_s_tv = recon["delta_s_tv"].squeeze()

    gt_crop = centre_crop(delta_s_gt, 0.8)
    tv_crop = centre_crop(delta_s_tv, 0.8)

    ncc = compute_ncc(tv_crop, gt_crop)
    nrmse = compute_nrmse(tv_crop, gt_crop)

    assert ncc > 0.95, f"TV-PDHG NCC too low: {ncc}"
    assert nrmse < 0.05, f"TV-PDHG NRMSE too high: {nrmse}"


def test_sos_range_physical():
    """Reconstructed SoS should be in a physically reasonable range."""
    ref_dir = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
    recon = np.load(os.path.join(ref_dir, "reconstructions.npz"))
    sos_tv = recon["sos_tv"].squeeze()

    # SoS should be between 1000 and 5000 m/s for biological tissue
    assert sos_tv.min() > 1000, f"SoS too low: {sos_tv.min()}"
    assert sos_tv.max() < 5000, f"SoS too high: {sos_tv.max()}"
