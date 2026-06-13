"""Tests for data loading and preprocessing."""

import os
import numpy as np
import pytest

from src.preprocessing import (
    load_multi_echo_data,
    load_ground_truth,
    load_metadata,
    preprocess_signal,
)


class TestLoadData:
    """Tests for data loading functions."""

    def test_multi_echo_shape(self, task_dir):
        """Multi-echo data should be (1, 256, 256, 10)."""
        signal = load_multi_echo_data(task_dir)
        assert signal.shape == (1, 256, 256, 10)
        assert signal.dtype == np.float64

    def test_ground_truth_shapes(self, task_dir):
        """Ground truth maps should be (1, 256, 256)."""
        T2, M0, mask = load_ground_truth(task_dir)
        assert T2.shape == (1, 256, 256)
        assert M0.shape == (1, 256, 256)
        assert mask.shape == (1, 256, 256)
        assert mask.dtype == bool

    def test_metadata_keys(self, task_dir):
        """Metadata should contain required keys."""
        meta = load_metadata(task_dir)
        assert 'echo_times_ms' in meta
        assert 'noise_sigma' in meta
        assert 'image_size' in meta
        assert 'n_echoes' in meta
        assert isinstance(meta['echo_times_ms'], np.ndarray)
        assert len(meta['echo_times_ms']) == 10

    def test_ground_truth_non_negative(self, task_dir):
        """T2 and M0 should be non-negative."""
        T2, M0, mask = load_ground_truth(task_dir)
        assert np.all(T2 >= 0)
        assert np.all(M0 >= 0)


class TestPreprocess:
    """Tests for signal preprocessing."""

    def test_removes_batch_dim(self):
        """Should remove batch dimension."""
        signal = np.random.rand(1, 32, 32, 10)
        result = preprocess_signal(signal)
        assert result.shape == (32, 32, 10)

    def test_non_negative(self):
        """Should clamp negative values to zero."""
        signal = np.array([[[[-1.0, 0.5, 1.0]]]])
        result = preprocess_signal(signal)
        assert np.all(result >= 0)
