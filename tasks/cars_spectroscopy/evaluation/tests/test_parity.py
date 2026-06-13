"""
Parity tests: verify the cleaned src/ code produces identical results
to the original carspy_code.py.
"""
import os
import sys
import numpy as np
import pytest

TASK_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
REPO_ROOT = os.path.join(TASK_ROOT, '..', '..')

sys.path.insert(0, TASK_ROOT)
sys.path.insert(0, REPO_ROOT)

from src.physics_model import (
    gaussian_line, lorentz_line, asym_Gaussian,
    LineStrength, forward_operator, downsample
)
from src.preprocessing import load_and_preprocess_data

# Import original code functions
import carspy_code as orig


class TestLineFunctionParity:
    def test_gaussian_line(self):
        w = np.linspace(2280, 2330, 200)
        np.testing.assert_allclose(
            gaussian_line(w, 2305, 1.5),
            orig.gaussian_line(w, 2305, 1.5),
            rtol=1e-12
        )

    def test_lorentz_line(self):
        w = np.linspace(2280, 2330, 200)
        np.testing.assert_allclose(
            lorentz_line(w, 2305, 1.5),
            orig.lorentz_line(w, 2305, 1.5),
            rtol=1e-12
        )

    def test_asym_gaussian(self):
        w = np.linspace(2280, 2330, 200)
        np.testing.assert_allclose(
            asym_Gaussian(w, 2305, 0.5, 2.0, 0, 0, 0),
            orig.asym_Gaussian(w, 2305, 0.5, 2.0, 0, 0, 0),
            rtol=1e-12
        )


class TestLineStrengthParity:
    def setup_method(self):
        self.ls_new = LineStrength('N2')
        self.ls_orig = orig.LineStrength('N2')

    def test_term_values(self):
        for mode in ('sum', 'Gv', 'Fv'):
            for v in (0, 1):
                for j in (0, 5, 15, 29):
                    new = self.ls_new.term_values(v, j, mode)
                    old = self.ls_orig.term_values(v, j, mode)
                    np.testing.assert_allclose(new, old, rtol=1e-12,
                        err_msg=f"v={v}, j={j}, mode={mode}")

    def test_line_pos(self):
        for branch in (0, 2, -2):
            for j in range(0, 25):
                new = self.ls_new.line_pos(0, j, branch)
                old = self.ls_orig.line_pos(0, j, branch)
                np.testing.assert_allclose(new, old, rtol=1e-12,
                    err_msg=f"j={j}, branch={branch}")

    def test_pop_factor(self):
        for T in (1000, 2400):
            for j in (0, 5, 10):
                new = self.ls_new.pop_factor(T, 0, j, branch=0)
                old = self.ls_orig.pop_factor(T, 0, j, branch=0)
                np.testing.assert_allclose(new, old, rtol=1e-12,
                    err_msg=f"T={T}, j={j}")


class TestForwardOperatorParity:
    def test_full_spectrum(self):
        nu_axis = np.linspace(2280, 2330, 200)
        params = {
            'nu': nu_axis, 'temperature': 2400, 'pressure': 1.0,
            'x_mol': 0.79, 'species': 'N2', 'pump_lw': 1.0,
            'slit_params': [0.5, 2.0, 0, 0]
        }
        new = forward_operator(params)
        old = orig.forward_operator(params)
        np.testing.assert_allclose(new, old, rtol=1e-10)

    def test_different_temperature(self):
        nu_axis = np.linspace(2280, 2330, 200)
        params = {
            'nu': nu_axis, 'temperature': 1200, 'pressure': 1.0,
            'x_mol': 0.79, 'species': 'N2', 'pump_lw': 1.0,
            'slit_params': [0.5, 2.0, 0, 0]
        }
        new = forward_operator(params)
        old = orig.forward_operator(params)
        np.testing.assert_allclose(new, old, rtol=1e-10)


class TestPreprocessingParity:
    def test_no_noise(self):
        signal = np.array([0.0, 0.5, 1.0, 0.8, 0.3])
        nu = np.array([2280, 2290, 2300, 2310, 2320], dtype=float)
        new_s, new_nu = load_and_preprocess_data(signal, nu, noise_level=0.0)
        old_s, old_nu = orig.load_and_preprocess_data(signal, nu, noise_level=0.0)
        np.testing.assert_allclose(new_s, old_s, rtol=1e-12)
        np.testing.assert_allclose(new_nu, old_nu, rtol=1e-12)
