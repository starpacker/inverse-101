"""Tests for physics_model module."""

import os
import numpy as np
import pytest
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "physics_model")

from src.physics_model import ParallelBeamProjector, find_rotation_center


class TestForward:
    @pytest.fixture
    def fixture(self):
        return np.load(os.path.join(FIXTURE_DIR, "projector.npz"))

    def test_output_shape(self, fixture):
        projector = ParallelBeamProjector(
            int(fixture["param_n_pixels"]),
            int(fixture["param_n_detector"]),
            fixture["param_theta"],
        )
        result = projector.forward(fixture["input_image"])
        assert result.shape == fixture["output_sinogram"].shape

    def test_output_values(self, fixture):
        projector = ParallelBeamProjector(
            int(fixture["param_n_pixels"]),
            int(fixture["param_n_detector"]),
            fixture["param_theta"],
        )
        result = projector.forward(fixture["input_image"])
        np.testing.assert_allclose(result, fixture["output_sinogram"], rtol=1e-10)

    def test_zero_image(self):
        projector = ParallelBeamProjector(8, 8, np.array([0.0, np.pi / 2]))
        result = projector.forward(np.zeros((8, 8)))
        np.testing.assert_allclose(result, 0.0, atol=1e-14)


class TestAdjoint:
    @pytest.fixture
    def fixture(self):
        return np.load(os.path.join(FIXTURE_DIR, "projector.npz"))

    def test_output_shape(self, fixture):
        projector = ParallelBeamProjector(
            int(fixture["param_n_pixels"]),
            int(fixture["param_n_detector"]),
            fixture["param_theta"],
        )
        result = projector.adjoint(fixture["output_sinogram"])
        assert result.shape == fixture["output_adjoint"].shape

    def test_output_values(self, fixture):
        projector = ParallelBeamProjector(
            int(fixture["param_n_pixels"]),
            int(fixture["param_n_detector"]),
            fixture["param_theta"],
        )
        result = projector.adjoint(fixture["output_sinogram"])
        np.testing.assert_allclose(result, fixture["output_adjoint"], rtol=1e-10)


class TestFindRotationCenter:
    @pytest.fixture
    def fixture(self):
        return np.load(os.path.join(FIXTURE_DIR, "rotation_center.npz"))

    def test_center_close_to_expected(self, fixture):
        center = find_rotation_center(
            fixture["input_sinogram"],
            fixture["input_theta"],
            init=float(fixture["config_init"]),
            tol=float(fixture["config_tol"]),
        )
        # Should be within 2 pixels of expected
        assert abs(center - float(fixture["output_center"])) < 2.0

    def test_center_in_valid_range(self, fixture):
        center = find_rotation_center(
            fixture["input_sinogram"],
            fixture["input_theta"],
            init=float(fixture["config_init"]),
            tol=float(fixture["config_tol"]),
        )
        n_det = fixture["input_sinogram"].shape[1]
        assert 0 < center < n_det
