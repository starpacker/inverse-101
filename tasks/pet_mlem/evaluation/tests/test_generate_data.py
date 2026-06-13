"""Tests for src/generate_data.py."""
import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")

import sys
sys.path.insert(0, TASK_DIR)

from src.generate_data import create_activity_phantom, generate_synthetic_data


class TestCreateActivityPhantom:
    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "generate_data_phantom.npz"))
        phantom = create_activity_phantom(int(fix["param_N"]))
        np.testing.assert_allclose(phantom, fix["output_phantom"], rtol=1e-10)

    def test_shape(self):
        phantom = create_activity_phantom(64)
        assert phantom.shape == (64, 64)

    def test_non_negative(self):
        phantom = create_activity_phantom(64)
        assert np.all(phantom >= 0)

    def test_has_hot_lesions(self):
        """Phantom should have values > 3 (hot lesion regions)."""
        phantom = create_activity_phantom(128)
        assert np.max(phantom) >= 4.0


class TestGenerateSyntheticData:
    def test_output_keys(self):
        data = generate_synthetic_data(N=32, n_angles=10, count_level=100)
        required = ['phantom', 'sino_clean', 'sino_noisy', 'background',
                     'theta', 'N', 'n_angles', 'count_level']
        for k in required:
            assert k in data, f"Missing key: {k}"

    def test_deterministic(self):
        d1 = generate_synthetic_data(N=32, n_angles=10, seed=42)
        d2 = generate_synthetic_data(N=32, n_angles=10, seed=42)
        np.testing.assert_array_equal(d1['sino_noisy'], d2['sino_noisy'])

    def test_noisy_differs_from_clean(self):
        data = generate_synthetic_data(N=32, n_angles=10, count_level=100)
        assert not np.allclose(data['sino_noisy'], data['sino_clean'])
