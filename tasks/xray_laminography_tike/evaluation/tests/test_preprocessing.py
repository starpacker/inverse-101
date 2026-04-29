"""Tests for src/preprocessing.py."""

import os
import sys
import json
import tempfile

import numpy as np
import pytest

try:
    import cupy
    cupy.array([1])
    HAS_GPU = True
except Exception:
    HAS_GPU = False

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_raw_data, load_ground_truth, load_metadata
from src.preprocessing import create_initial_guess


class TestLoadRawData:
    """Tests for load_raw_data."""

    def test_loads_from_file(self):
        """Test loading raw_data.npz returns correct keys and dtypes."""
        path = os.path.join(TASK_DIR, 'data', 'raw_data.npz')
        if not os.path.exists(path):
            pytest.skip("raw_data.npz not found")

        result = load_raw_data(path)
        assert 'projections' in result
        assert 'theta' in result
        assert result['projections'].dtype == np.complex64
        assert result['theta'].dtype == np.float32

    def test_shapes(self):
        """Test that loaded arrays have batch-first shapes."""
        path = os.path.join(TASK_DIR, 'data', 'raw_data.npz')
        if not os.path.exists(path):
            pytest.skip("raw_data.npz not found")

        result = load_raw_data(path)
        assert result['projections'].ndim == 4
        assert result['projections'].shape[0] == 1
        assert result['theta'].ndim == 2
        assert result['theta'].shape[0] == 1

    def test_synthetic_npz(self, tmp_path):
        """Test loading a synthetic npz file."""
        proj = np.zeros((1, 4, 8, 8), dtype=np.complex64)
        theta = np.zeros((1, 4), dtype=np.float32)
        path = str(tmp_path / 'test_raw.npz')
        np.savez(path, projections=proj, theta=theta)

        result = load_raw_data(path)
        assert result['projections'].shape == (1, 4, 8, 8)
        assert result['theta'].shape == (1, 4)


class TestLoadGroundTruth:
    """Tests for load_ground_truth."""

    def test_loads_from_file(self):
        """Test loading ground_truth.npz returns correct key and dtype."""
        path = os.path.join(TASK_DIR, 'data', 'ground_truth.npz')
        if not os.path.exists(path):
            pytest.skip("ground_truth.npz not found")

        result = load_ground_truth(path)
        assert 'volume' in result
        assert result['volume'].dtype == np.complex64

    def test_shape(self):
        """Test that volume has batch-first 4D shape."""
        path = os.path.join(TASK_DIR, 'data', 'ground_truth.npz')
        if not os.path.exists(path):
            pytest.skip("ground_truth.npz not found")

        result = load_ground_truth(path)
        assert result['volume'].ndim == 4
        assert result['volume'].shape[0] == 1


class TestLoadMetadata:
    """Tests for load_metadata."""

    def test_loads_from_file(self):
        """Test loading meta_data.json returns expected keys."""
        path = os.path.join(TASK_DIR, 'data', 'meta_data.json')
        result = load_metadata(path)
        assert 'volume_shape' in result
        assert 'n_angles' in result
        assert 'tilt_rad' in result

    def test_values(self):
        """Test that metadata has expected values."""
        path = os.path.join(TASK_DIR, 'data', 'meta_data.json')
        result = load_metadata(path)
        assert result['volume_shape'] == [128, 128, 128]
        assert result['n_angles'] == 128
        assert abs(result['tilt_rad'] - np.pi / 2) < 1e-4


class TestCreateInitialGuess:
    """Tests for create_initial_guess."""

    def test_shape_and_dtype(self):
        """Test that initial guess has correct shape and dtype."""
        shape = (64, 64, 64)
        guess = create_initial_guess(shape)
        assert guess.shape == shape
        assert guess.dtype == np.complex64

    def test_all_zeros(self):
        """Test that initial guess is all zeros."""
        guess = create_initial_guess((16, 16, 16))
        assert np.all(guess == 0)

    def test_custom_dtype(self):
        """Test with a custom dtype."""
        guess = create_initial_guess((8, 8, 8), dtype=np.complex128)
        assert guess.dtype == np.complex128
