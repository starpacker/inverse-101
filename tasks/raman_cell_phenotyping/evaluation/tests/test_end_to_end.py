"""End-to-end integration test."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.preprocessing import load_observation, preprocess_volume
from src.solvers import unmix
from src.physics_model import reconstruction_error
from src.visualization import compute_ncc, compute_nrmse


class TestEndToEnd:
    """Integration test: run the full pipeline and check output quality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        import random
        random.seed(12345)
        obs = load_observation("data")
        self.volume = obs["spectral_volume"]
        self.axis = obs["spectral_axis"]

    def test_preprocessing_output_shape(self):
        processed, proc_axis = preprocess_volume(self.volume, self.axis)
        assert processed.shape[:3] == (40, 40, 10)
        assert proc_axis.shape[0] == processed.shape[-1]
        # Fingerprint region should give roughly 400-500 channels
        assert 300 < processed.shape[-1] < 600

    def test_preprocessing_output_range(self):
        processed, _ = preprocess_volume(self.volume, self.axis)
        # After min-max normalisation, range should be [0, 1]
        assert processed.min() >= -1e-6
        assert processed.max() <= 1 + 1e-6

    def test_unmixing_produces_valid_abundances(self):
        processed, _ = preprocess_volume(self.volume, self.axis)
        abundance_maps, endmembers = unmix(processed, n_endmembers=5)

        assert len(abundance_maps) == 5
        assert len(endmembers) == 5

        # Abundances should be mostly non-negative and sum near 1
        total = sum(a for a in abundance_maps)
        np.testing.assert_allclose(total, 1.0, atol=0.1)

    def test_reconstruction_quality(self):
        processed, _ = preprocess_volume(self.volume, self.axis)
        abundance_maps, endmembers = unmix(processed, n_endmembers=5)

        flat_obs = processed.reshape(-1, processed.shape[-1])
        endmember_matrix = np.stack(endmembers)
        flat_abund = np.stack([a.ravel() for a in abundance_maps], axis=-1)
        rmse = reconstruction_error(flat_obs, endmember_matrix, flat_abund)

        # RMSE should be small relative to data range [0, 1]
        assert rmse < 0.05
