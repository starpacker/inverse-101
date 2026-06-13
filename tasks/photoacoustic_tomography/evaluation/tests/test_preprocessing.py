"""Tests for src/preprocessing.py."""

import numpy as np
import os
import pytest


class TestLoadRawData:
    """Tests for loading raw PA data."""

    def test_loads_without_error(self):
        from src.preprocessing import load_raw_data
        signals, xd, yd, t = load_raw_data("data")
        assert signals is not None

    def test_shapes(self):
        from src.preprocessing import load_raw_data
        signals, xd, yd, t = load_raw_data("data")
        assert signals.ndim == 3  # (n_time, n_det_x, n_det_y)
        assert xd.ndim == 1
        assert yd.ndim == 1
        assert t.ndim == 1
        assert signals.shape[1] == len(xd)
        assert signals.shape[2] == len(yd)
        assert signals.shape[0] == len(t)

    def test_dtypes(self):
        from src.preprocessing import load_raw_data
        signals, xd, yd, t = load_raw_data("data")
        assert signals.dtype == np.float64
        assert xd.dtype == np.float64


class TestLoadGroundTruth:
    """Tests for loading ground truth."""

    def test_loads_without_error(self):
        from src.preprocessing import load_ground_truth
        gt, xf, yf = load_ground_truth("data")
        assert gt is not None

    def test_shapes(self):
        from src.preprocessing import load_ground_truth
        gt, xf, yf = load_ground_truth("data")
        assert gt.ndim == 2
        assert gt.shape[0] == len(xf)
        assert gt.shape[1] == len(yf)


class TestLoadMetadata:
    """Tests for loading metadata."""

    def test_loads_without_error(self):
        from src.preprocessing import load_metadata
        meta = load_metadata("data")
        assert isinstance(meta, dict)

    def test_required_keys(self):
        from src.preprocessing import load_metadata
        meta = load_metadata("data")
        required = ["sound_speed_m_per_s", "sampling_frequency_hz",
                     "target_plane_z_m", "detector_size_m"]
        for key in required:
            assert key in meta, f"Missing key: {key}"
