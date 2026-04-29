import os
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_model import (
    gaussian_line, lorentz_line, asym_Gaussian, downsample,
    LineStrength, forward_operator
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'physics_model_fixtures.npz')


@pytest.fixture
def fixtures():
    return np.load(FIXTURE_PATH)


class TestGaussianLine:
    def test_output_matches_fixture(self, fixtures):
        w = fixtures['input_w']
        w0 = float(fixtures['input_w0'])
        sigma = float(fixtures['input_sigma_gauss'])
        result = gaussian_line(w, w0, sigma)
        np.testing.assert_allclose(result, fixtures['output_gaussian_line'], rtol=1e-10)

    def test_zero_sigma_returns_zeros(self):
        w = np.linspace(2280, 2330, 50)
        assert np.all(gaussian_line(w, 2305, 0) == 0)

    def test_peak_at_center(self):
        w = np.linspace(2280, 2330, 1001)
        result = gaussian_line(w, 2305, 2.0)
        peak_idx = np.argmax(result)
        assert abs(w[peak_idx] - 2305) < 0.1


class TestLorentzLine:
    def test_output_matches_fixture(self, fixtures):
        w = fixtures['input_w']
        w0 = float(fixtures['input_w0'])
        sigma = float(fixtures['input_sigma_lorentz'])
        result = lorentz_line(w, w0, sigma)
        np.testing.assert_allclose(result, fixtures['output_lorentz_line'], rtol=1e-10)

    def test_zero_sigma_returns_zeros(self):
        w = np.linspace(2280, 2330, 50)
        assert np.all(lorentz_line(w, 2305, 0) == 0)


class TestAsymGaussian:
    def test_output_matches_fixture(self, fixtures):
        w = fixtures['input_w']
        result = asym_Gaussian(w, 2305.0, sigma=0.5, k=2.0, a_sigma=0, a_k=0, offset=0)
        np.testing.assert_allclose(result, fixtures['output_asym_gaussian'], rtol=1e-10)

    def test_normalized_to_one(self):
        w = np.linspace(2280, 2330, 200)
        result = asym_Gaussian(w, 2305, sigma=1.0, k=2.0, a_sigma=0, a_k=0, offset=0)
        assert abs(result.max() - 1.0) < 1e-10


class TestDownsample:
    def test_output_matches_fixture(self, fixtures):
        w_coarse = fixtures['input_w_coarse']
        w_fine = fixtures['input_w_fine']
        spec_fine = fixtures['input_spec_fine']
        result = downsample(w_coarse, w_fine, spec_fine, mode='local_mean')
        np.testing.assert_allclose(result, fixtures['output_downsample'], rtol=1e-10)


class TestLineStrength:
    def setup_method(self):
        self.ls = LineStrength('N2')

    def test_term_values_sum(self, fixtures):
        result = self.ls.term_values(0, 5, mode='sum')
        np.testing.assert_allclose(result, float(fixtures['output_term_values_sum']), rtol=1e-10)

    def test_term_values_Gv(self, fixtures):
        result = self.ls.term_values(0, 5, mode='Gv')
        np.testing.assert_allclose(result, float(fixtures['output_term_values_Gv']), rtol=1e-10)

    def test_term_values_Fv(self, fixtures):
        result = self.ls.term_values(0, 5, mode='Fv')
        np.testing.assert_allclose(result, float(fixtures['output_term_values_Fv']), rtol=1e-10)

    def test_line_pos(self, fixtures):
        result = self.ls.line_pos(0, 5, branch=0)
        np.testing.assert_allclose(result, float(fixtures['output_line_pos']), rtol=1e-10)

    def test_pop_factor(self, fixtures):
        result = self.ls.pop_factor(2400, 0, 5, branch=0)
        np.testing.assert_allclose(result, float(fixtures['output_pop_factor']), rtol=1e-10)

    def test_int_corr(self, fixtures):
        pt, cd = self.ls.int_corr(5, branch=0)
        np.testing.assert_allclose(pt, float(fixtures['output_int_corr_pt']), rtol=1e-10)
        np.testing.assert_allclose(cd, float(fixtures['output_int_corr_cd']), rtol=1e-10)


class TestForwardOperator:
    def test_output_matches_fixture(self, fixtures):
        nu_axis = fixtures['input_nu_axis']
        params = {
            'nu': nu_axis, 'temperature': 2400, 'pressure': 1.0,
            'x_mol': 0.79, 'species': 'N2', 'pump_lw': 1.0,
            'slit_params': [0.5, 2.0, 0, 0]
        }
        result = forward_operator(params)
        np.testing.assert_allclose(result, fixtures['output_forward_operator'], rtol=1e-10)

    def test_output_normalized(self):
        nu_axis = np.linspace(2280, 2330, 100)
        params = {
            'nu': nu_axis, 'temperature': 1500, 'pressure': 1.0,
            'x_mol': 0.79, 'species': 'N2', 'pump_lw': 1.0,
            'slit_params': [0.5, 2.0, 0, 0]
        }
        result = forward_operator(params)
        assert abs(result.max() - 1.0) < 1e-10
        assert result.min() >= 0
