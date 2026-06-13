"""Tests for preprocessing module."""

import os
import numpy as np
import pytest
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "preprocessing")

from src.preprocessing import load_observation, load_metadata, normalize, minus_log


class TestLoadObservation:
    def test_keys(self):
        obs = load_observation(os.path.join(TASK_DIR, "data"))
        assert set(obs.keys()) == {"projections", "flat_field", "dark_field", "theta"}

    def test_shapes(self):
        obs = load_observation(os.path.join(TASK_DIR, "data"))
        assert obs["projections"].shape == (181, 2, 640)
        assert obs["flat_field"].shape == (10, 2, 640)
        assert obs["dark_field"].shape == (10, 2, 640)
        assert obs["theta"].shape == (181,)

    def test_theta_range(self):
        obs = load_observation(os.path.join(TASK_DIR, "data"))
        assert obs["theta"][0] == pytest.approx(0.0, abs=1e-6)
        assert obs["theta"][-1] == pytest.approx(np.pi, abs=0.05)


class TestLoadMetadata:
    def test_keys(self):
        meta = load_metadata(os.path.join(TASK_DIR, "data"))
        assert "n_projections" in meta
        assert "n_detector_pixels" in meta
        assert meta["n_projections"] == 181
        assert meta["n_detector_pixels"] == 640


class TestNormalize:
    def test_output_values(self):
        f = np.load(os.path.join(FIXTURE_DIR, "normalize.npz"))
        result = normalize(f["input_proj"], f["input_flat"], f["input_dark"])
        np.testing.assert_allclose(result, f["output_normalized"], rtol=1e-10)

    def test_dtype(self):
        f = np.load(os.path.join(FIXTURE_DIR, "normalize.npz"))
        result = normalize(f["input_proj"], f["input_flat"], f["input_dark"])
        assert result.dtype == np.float64


class TestMinusLog:
    def test_output_values(self):
        f = np.load(os.path.join(FIXTURE_DIR, "minus_log.npz"))
        result = minus_log(f["input_normalized"])
        np.testing.assert_allclose(result, f["output_sinogram"], rtol=1e-10)

    def test_non_negative_for_valid_input(self):
        # For transmission in (0, 1], -log should be >= 0
        transmission = np.array([0.5, 0.1, 0.9, 1.0])
        result = minus_log(transmission)
        assert np.all(result >= 0)

    def test_handles_zero(self):
        transmission = np.array([0.0, 0.5])
        result = minus_log(transmission)
        assert np.isfinite(result).all()
