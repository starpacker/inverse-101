"""Tests for the preprocessing module."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_ground_truth, load_raw_data, load_metadata


@pytest.fixture
def data_dir():
    return os.path.join(TASK_DIR, "data")


def test_load_ground_truth_keys(data_dir):
    gt = load_ground_truth(data_dir)
    assert "sos_phantom" in gt
    assert "slowness_perturbation" in gt


def test_load_ground_truth_shapes(data_dir):
    gt = load_ground_truth(data_dir)
    assert gt["sos_phantom"].shape == (128, 128)
    assert gt["slowness_perturbation"].shape == (128, 128)


def test_load_ground_truth_sos_range(data_dir):
    gt = load_ground_truth(data_dir)
    sos = gt["sos_phantom"]
    assert sos.min() >= 1400
    assert sos.max() <= 3000


def test_load_ground_truth_perturbation_consistency(data_dir):
    """Slowness perturbation should be consistent with SoS phantom."""
    gt = load_ground_truth(data_dir)
    sos = gt["sos_phantom"]
    delta_s = gt["slowness_perturbation"]
    expected = 1.0 / sos - 1.0 / 1500.0
    np.testing.assert_allclose(delta_s, expected, rtol=1e-10)


def test_load_raw_data_keys(data_dir):
    raw = load_raw_data(data_dir)
    expected_keys = {"sinogram", "sinogram_clean", "sinogram_full", "angles", "angles_full"}
    assert expected_keys == set(raw.keys())


def test_load_raw_data_shapes(data_dir):
    raw = load_raw_data(data_dir)
    assert raw["sinogram"].shape == (128, 60)
    assert raw["sinogram_clean"].shape == (128, 60)
    assert raw["sinogram_full"].shape == (128, 180)
    assert raw["angles"].shape == (60,)
    assert raw["angles_full"].shape == (180,)


def test_load_metadata(data_dir):
    meta = load_metadata(data_dir)
    assert meta["image_size"] == 128
    assert meta["n_angles"] == 60
    assert meta["background_sos_m_per_s"] == 1500.0
