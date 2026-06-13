"""Tests for preprocessing.py."""

import os
import sys
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_raw_data, load_ground_truth, load_metadata

DATA_DIR = os.path.join(TASK_DIR, "data")


@pytest.fixture(scope="module")
def ensure_data():
    """Ensure data files exist."""
    if not os.path.exists(os.path.join(DATA_DIR, "raw_data.npz")):
        pytest.skip("Data files not generated yet")


class TestLoadRawData:
    def test_shapes(self, ensure_data):
        sinograms, spectra, mus, energies, theta = load_raw_data(DATA_DIR)
        assert sinograms.shape[0] == 2  # nMeas
        assert spectra.shape[0] == 2
        assert mus.shape[0] == 2
        assert len(energies.shape) == 1
        assert len(theta.shape) == 1

    def test_positive_sinograms(self, ensure_data):
        sinograms, _, _, _, _ = load_raw_data(DATA_DIR)
        assert np.all(sinograms >= 0)


class TestLoadGroundTruth:
    def test_shapes(self, ensure_data):
        tissue, bone, t_sino, b_sino = load_ground_truth(DATA_DIR)
        assert tissue.ndim == 2
        assert bone.ndim == 2
        assert t_sino.ndim == 2
        assert b_sino.ndim == 2

    def test_non_negative(self, ensure_data):
        tissue, bone, _, _ = load_ground_truth(DATA_DIR)
        assert np.all(tissue >= 0)
        assert np.all(bone >= 0)


class TestLoadMetadata:
    def test_keys(self, ensure_data):
        meta = load_metadata(DATA_DIR)
        assert "image_size" in meta
        assert "n_angles" in meta
        assert "material_names" in meta
        assert meta["image_size"] == 128
