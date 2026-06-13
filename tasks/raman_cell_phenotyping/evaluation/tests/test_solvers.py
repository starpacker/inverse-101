"""Unit tests for spectral unmixing solvers."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.solvers import extract_endmembers_nfindr, estimate_abundances_fcls, unmix


class TestNFINDR:
    def test_output_shape(self):
        np.random.seed(0)
        data = np.random.rand(100, 20)
        endmembers = extract_endmembers_nfindr(data, 3)
        assert endmembers.shape == (3, 20)

    def test_endmembers_from_data(self):
        np.random.seed(0)
        data = np.random.rand(100, 20)
        endmembers = extract_endmembers_nfindr(data, 3)
        # Each endmember should be close to one of the data points
        for em in endmembers:
            dists = np.linalg.norm(data - em, axis=1)
            assert dists.min() < 1e-6


class TestFCLS:
    def test_sum_to_one(self):
        np.random.seed(42)
        K, B = 3, 20
        endmembers = np.random.rand(K, B) + 0.1
        abundances_true = np.random.dirichlet(np.ones(K), size=50)
        data = abundances_true @ endmembers

        abundances_est = estimate_abundances_fcls(data, endmembers)
        # Should approximately sum to one
        sums = abundances_est.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=0.05)

    def test_nonnegative(self):
        np.random.seed(42)
        K, B = 3, 20
        endmembers = np.random.rand(K, B) + 0.1
        abundances_true = np.random.dirichlet(np.ones(K), size=50)
        data = abundances_true @ endmembers

        abundances_est = estimate_abundances_fcls(data, endmembers)
        assert np.all(abundances_est >= -1e-6)


class TestUnmix:
    def test_output_structure(self):
        np.random.seed(42)
        volume = np.random.rand(4, 4, 2, 20)
        maps, endmembers = unmix(volume, n_endmembers=3)
        assert len(maps) == 3
        assert len(endmembers) == 3
        assert maps[0].shape == (4, 4, 2)
        assert endmembers[0].shape == (20,)
