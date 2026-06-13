"""Tests for preprocessing module."""

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


def test_load_ground_truth_shape(data_dir):
    phantom = load_ground_truth(data_dir)
    assert phantom.shape == (256, 256)
    assert phantom.dtype == np.float64


def test_load_ground_truth_range(data_dir):
    phantom = load_ground_truth(data_dir)
    assert phantom.min() >= 0.0
    assert phantom.max() <= 1.1  # Shepp-Logan values are in [0, ~1]


def test_load_raw_data_keys(data_dir):
    raw = load_raw_data(data_dir)
    expected_keys = {"sinogram_sparse", "sinogram_full", "angles_sparse", "angles_full"}
    assert set(raw.keys()) == expected_keys


def test_load_raw_data_shapes(data_dir):
    raw = load_raw_data(data_dir)
    assert raw["sinogram_sparse"].shape == (256, 30)
    assert raw["sinogram_full"].shape == (256, 180)
    assert raw["angles_sparse"].shape == (30,)
    assert raw["angles_full"].shape == (180,)


def test_load_metadata(data_dir):
    meta = load_metadata(data_dir)
    assert meta["image_size"] == 256
    assert meta["n_angles_full"] == 180
    assert meta["n_angles_sparse"] == 30
    assert meta["noise_std"] == 0.02
