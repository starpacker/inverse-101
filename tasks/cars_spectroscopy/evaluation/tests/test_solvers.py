import os
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_model import forward_operator
from src.solvers import run_inversion


class TestRunInversion:
    @pytest.fixture(scope='class')
    def inversion_result(self):
        """Run inversion once for all tests in this class."""
        nu_axis = np.linspace(2280, 2330, 200)
        true_params = {
            'nu': nu_axis, 'temperature': 2400, 'pressure': 1.0,
            'x_mol': 0.79, 'species': 'N2', 'pump_lw': 1.0,
            'slit_params': [0.5, 2.0, 0, 0]
        }
        clean_signal = forward_operator(true_params)

        initial_guesses = {
            'temperature': 2000,
            'x_mol': 0.79,
            'pressure': 1.0,
            'pump_lw': 1.0,
        }
        result = run_inversion(clean_signal, nu_axis, initial_guesses)
        return result, clean_signal

    def test_returns_required_keys(self, inversion_result):
        result, _ = inversion_result
        assert 'best_params' in result
        assert 'y_pred' in result
        assert 'success' in result

    def test_temperature_within_50K(self, inversion_result):
        result, _ = inversion_result
        T_pred = result['best_params']['temperature']
        assert abs(T_pred - 2400) < 50, f"Temperature {T_pred} too far from 2400"

    def test_spectrum_shape(self, inversion_result):
        result, clean_signal = inversion_result
        assert result['y_pred'].shape == clean_signal.shape

    def test_ncc_above_threshold(self, inversion_result):
        result, clean_signal = inversion_result
        y_pred = result['y_pred']
        ncc = np.dot(clean_signal, y_pred) / (np.linalg.norm(clean_signal) * np.linalg.norm(y_pred))
        assert ncc > 0.99, f"NCC {ncc} too low"
