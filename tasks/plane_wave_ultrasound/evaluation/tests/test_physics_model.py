from pathlib import Path

import numpy as np

from src.physics_model import erm_velocity, steering_delay, stolt_fkz


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_erm_velocity_matches_fixture():
    params = np.load(FIXTURES / "param_erm_velocity.npz")
    actual = erm_velocity(float(params["c"]), float(params["txangle"]))
    expected = float(np.load(FIXTURES / "output_erm_velocity.npy"))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_stolt_fkz_matches_fixture():
    params = np.load(FIXTURES / "param_stolt_fkz.npz")
    actual = stolt_fkz(params["f"], params["Kx"], float(params["c"]), float(params["txangle"]))
    expected = np.load(FIXTURES / "output_stolt_fkz.npy")
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_steering_delay_matches_fixture():
    params = np.load(FIXTURES / "param_steering_delay.npz")
    actual = steering_delay(
        int(params["nx"]),
        float(params["pitch"]),
        float(params["c"]),
        float(params["txangle"]),
        float(params["t0"]),
    )
    expected = np.load(FIXTURES / "output_steering_delay.npy")
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
