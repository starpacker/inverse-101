"""Tests for src/physics_model.py."""

import numpy as np
import pytest
import os
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


class TestStepFunction:
    """Tests for the Heaviside step function."""

    def test_values(self):
        from src.physics_model import step_function
        fix = np.load(os.path.join(FIXTURES_DIR, "physics_model_step_function.npz"))
        result = step_function(fix["input_x"])
        np.testing.assert_array_equal(result, fix["output_y"])

    def test_negative(self):
        from src.physics_model import step_function
        assert step_function(np.array([-5.0]))[0] == 0.0

    def test_positive(self):
        from src.physics_model import step_function
        assert step_function(np.array([5.0]))[0] == 1.0

    def test_zero(self):
        from src.physics_model import step_function
        assert step_function(np.array([0.0]))[0] == 0.5


class TestPASignalSingleTarget:
    """Tests for single-target PA signal simulation."""

    @pytest.fixture
    def fixture(self):
        return np.load(
            os.path.join(FIXTURES_DIR, "physics_model_single_target.npz"))

    def test_shape(self, fixture):
        from src.physics_model import pa_signal_single_target
        sig = pa_signal_single_target(
            fixture["param_tar_info"], fixture["param_xd"],
            fixture["param_yd"], fixture["param_t"], float(fixture["param_c"]))
        assert sig.shape == (len(fixture["param_t"]),
                             len(fixture["param_xd"]),
                             len(fixture["param_yd"]))

    def test_exact_values(self, fixture):
        from src.physics_model import pa_signal_single_target
        sig = pa_signal_single_target(
            fixture["param_tar_info"], fixture["param_xd"],
            fixture["param_yd"], fixture["param_t"], float(fixture["param_c"]))
        np.testing.assert_allclose(sig, fixture["output_signal"], rtol=1e-10)

    def test_causality(self, fixture):
        """No signal before the earliest possible arrival time."""
        from src.physics_model import pa_signal_single_target
        tar = fixture["param_tar_info"]
        t = fixture["param_t"]
        c = float(fixture["param_c"])

        sig = pa_signal_single_target(
            tar, fixture["param_xd"], fixture["param_yd"], t, c)

        # Min distance from any detector to target
        min_dist = tar[2]  # z-coordinate (target directly above centre)
        radius = tar[3]
        earliest_time = (min_dist - radius) / c
        earliest_idx = int(np.floor(earliest_time / (t[1] - t[0])))

        # All signals should be zero well before arrival
        if earliest_idx > 5:
            assert np.max(np.abs(sig[:earliest_idx - 2, :, :])) == 0.0

    def test_symmetry(self, fixture):
        """Centre detector should have symmetric response for on-axis target."""
        from src.physics_model import pa_signal_single_target
        sig = pa_signal_single_target(
            fixture["param_tar_info"], fixture["param_xd"],
            fixture["param_yd"], fixture["param_t"], float(fixture["param_c"]))
        # Target at (0,0,z): signal at det (0,0) should be same as det(-2,-2) and (2,2) by symmetry
        np.testing.assert_allclose(sig[:, 0, 0], sig[:, 2, 2], rtol=1e-10)
        np.testing.assert_allclose(sig[:, 0, 2], sig[:, 2, 0], rtol=1e-10)


class TestGroundTruthImage:
    """Tests for ground truth image generation."""

    def test_exact_values(self):
        from src.physics_model import generate_ground_truth_image
        fix = np.load(
            os.path.join(FIXTURES_DIR, "physics_model_ground_truth.npz"))
        gt = generate_ground_truth_image(
            fix["param_tar_info"], fix["param_xf"], fix["param_yf"])
        np.testing.assert_array_equal(gt, fix["output_gt"])

    def test_binary(self):
        from src.physics_model import generate_ground_truth_image
        fix = np.load(
            os.path.join(FIXTURES_DIR, "physics_model_ground_truth.npz"))
        gt = generate_ground_truth_image(
            fix["param_tar_info"], fix["param_xf"], fix["param_yf"])
        unique = np.unique(gt)
        assert set(unique).issubset({0.0, 1.0})

    def test_has_nonzero(self):
        from src.physics_model import generate_ground_truth_image
        fix = np.load(
            os.path.join(FIXTURES_DIR, "physics_model_ground_truth.npz"))
        gt = generate_ground_truth_image(
            fix["param_tar_info"], fix["param_xf"], fix["param_yf"])
        assert np.count_nonzero(gt) > 0


class TestSimulatePASignals:
    """Tests for multi-target PA signal simulation."""

    @pytest.fixture
    def fixture(self):
        return np.load(
            os.path.join(FIXTURES_DIR, "physics_model_simulate.npz"))

    def test_fixture_match(self, fixture):
        from src.physics_model import simulate_pa_signals
        signals = simulate_pa_signals(
            fixture["input_tar_info"], fixture["input_xd"],
            fixture["input_yd"], fixture["input_t"])
        np.testing.assert_allclose(signals, fixture["output_signals"], rtol=1e-10)

    def test_shape(self, fixture):
        from src.physics_model import simulate_pa_signals
        signals = simulate_pa_signals(
            fixture["input_tar_info"], fixture["input_xd"],
            fixture["input_yd"], fixture["input_t"])
        n_time = len(fixture["input_t"])
        n_det_x = len(fixture["input_xd"])
        n_det_y = len(fixture["input_yd"])
        assert signals.shape == (n_time, n_det_x, n_det_y)

    def test_deterministic(self, fixture):
        from src.physics_model import simulate_pa_signals
        s1 = simulate_pa_signals(
            fixture["input_tar_info"], fixture["input_xd"],
            fixture["input_yd"], fixture["input_t"])
        s2 = simulate_pa_signals(
            fixture["input_tar_info"], fixture["input_xd"],
            fixture["input_yd"], fixture["input_t"])
        np.testing.assert_array_equal(s1, s2)
