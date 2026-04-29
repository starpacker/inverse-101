"""Tests for physics_model module."""

import os
import sys
import numpy as np
import torch
import pytest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")


class TestObservationOperator:
    """Test the masking observation operator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "physics_model_mask.npz"))
        self.input_x = fix["input_x"]
        self.mask = fix["param_mask"]
        self.expected_masked = fix["output_masked"]

    def test_numpy_masking(self):
        from src.physics_model import make_observation_operator
        op = make_observation_operator(self.mask)
        result = op(self.input_x)
        np.testing.assert_allclose(result, self.expected_masked, rtol=1e-10)

    def test_torch_masking(self):
        from src.physics_model import make_observation_operator
        mask_t = torch.from_numpy(self.mask)
        x_t = torch.from_numpy(self.input_x)
        op = make_observation_operator(mask_t)
        result = op(x_t).numpy()
        np.testing.assert_allclose(result, self.expected_masked, rtol=1e-6)

    def test_mask_zeros_out_unobserved(self):
        from src.physics_model import make_observation_operator
        op = make_observation_operator(self.mask)
        result = op(self.input_x)
        unobserved = self.mask[0] == 0
        for i in range(result.shape[0]):
            np.testing.assert_array_equal(result[i][unobserved], 0.0)

    def test_mask_preserves_observed(self):
        from src.physics_model import make_observation_operator
        op = make_observation_operator(self.mask)
        result = op(self.input_x)
        observed = self.mask[0] == 1
        for i in range(result.shape[0]):
            np.testing.assert_allclose(result[i][observed], self.input_x[i][observed], rtol=1e-10)


class TestNoiser:
    """Test the Gaussian noiser (statistical properties)."""

    def test_noiser_mean(self):
        from src.physics_model import make_noiser
        noiser = make_noiser(0.001)
        x = np.ones((100, 128, 128), dtype=np.float32)
        results = np.stack([noiser(x) for _ in range(10)])
        mean_noise = (results - 1.0).mean()
        assert abs(mean_noise) < 0.01, f"Mean noise {mean_noise} too far from 0"

    def test_noiser_std(self):
        from src.physics_model import make_noiser
        sigma = 0.01
        noiser = make_noiser(sigma)
        x = np.zeros((1000, 128, 128), dtype=np.float32)
        result = noiser(x)
        measured_std = result.std()
        assert abs(measured_std - sigma) < 0.002, f"Measured std {measured_std} vs expected {sigma}"

    def test_noiser_torch(self):
        from src.physics_model import make_noiser
        noiser = make_noiser(0.001)
        x = torch.ones(10, 128, 128)
        result = noiser(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_forward_model_shape(self):
        from src.physics_model import forward_model
        x = np.random.rand(3, 128, 128).astype(np.float32)
        mask = (np.random.rand(1, 128, 128) < 0.1).astype(np.float32)
        y = forward_model(x, mask, 0.001)
        assert y.shape == x.shape
        assert y.dtype == np.float32
