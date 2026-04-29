"""Tests for generate_data module."""

import os
import sys
import numpy as np
import pytest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")


class TestGenerateObservations:
    """Test synthetic observation generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "generate_data.npz"))
        self.gt = fix["input_ground_truth"]
        self.expected_obs = fix["output_observations"]
        self.expected_mask = fix["output_mask"]

    def test_output_shape(self):
        from src.generate_data import generate_observations
        obs, mask = generate_observations(self.gt, mask_ratio=0.1, noise_sigma=0.001, seed=42)
        assert obs.shape == self.gt.shape
        assert mask.shape == (1, 128, 128)

    def test_deterministic_with_seed(self):
        from src.generate_data import generate_observations
        obs1, mask1 = generate_observations(self.gt, seed=42)
        obs2, mask2 = generate_observations(self.gt, seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(mask1, mask2)

    def test_matches_fixture(self):
        from src.generate_data import generate_observations
        obs, mask = generate_observations(self.gt, mask_ratio=0.1, noise_sigma=0.001, seed=42)
        np.testing.assert_allclose(obs, self.expected_obs, rtol=1e-5)
        np.testing.assert_array_equal(mask, self.expected_mask)

    def test_mask_is_binary(self):
        from src.generate_data import generate_observations
        _, mask = generate_observations(self.gt, seed=42)
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_mask_coverage_approximate(self):
        from src.generate_data import generate_observations
        _, mask = generate_observations(self.gt, mask_ratio=0.1, seed=42)
        coverage = mask.mean()
        assert 0.05 < coverage < 0.20

    def test_observations_clipped(self):
        from src.generate_data import generate_observations
        obs, _ = generate_observations(self.gt, seed=42)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0
