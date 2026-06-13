import os
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing import load_and_preprocess_data

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'preprocessing_fixtures.npz')


@pytest.fixture
def fixtures():
    return np.load(FIXTURE_PATH)


class TestLoadAndPreprocessData:
    def test_deterministic_output(self, fixtures):
        signal_in = fixtures['input_signal']
        nu_in = fixtures['input_nu']
        out, nu_out = load_and_preprocess_data(signal_in, nu_in, noise_level=0.0)
        np.testing.assert_allclose(out, fixtures['output_signal'], rtol=1e-10)
        np.testing.assert_allclose(nu_out, fixtures['output_nu'], rtol=1e-10)

    def test_normalized_to_one(self):
        signal = np.array([0.0, 2.0, 5.0, 3.0, 1.0])
        nu = np.arange(5, dtype=float)
        out, _ = load_and_preprocess_data(signal, nu, noise_level=0.0)
        assert abs(out.max() - 1.0) < 1e-10

    def test_negative_clipped(self):
        signal = np.array([-1.0, 0.5, 1.0])
        nu = np.arange(3, dtype=float)
        out, _ = load_and_preprocess_data(signal, nu, noise_level=0.0)
        assert out.min() >= 0

    def test_noisy_output_shape(self):
        signal = np.ones(50)
        nu = np.arange(50, dtype=float)
        out, _ = load_and_preprocess_data(signal, nu, noise_level=0.1)
        assert out.shape == (50,)

    def test_noisy_output_statistics(self):
        np.random.seed(0)
        signal = np.ones(10000)
        nu = np.arange(10000, dtype=float)
        out, _ = load_and_preprocess_data(signal, nu, noise_level=0.01)
        assert abs(out.mean() - 1.0) < 0.05
