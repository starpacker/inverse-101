"""Tests for metrics computation."""

import os
import numpy as np
import pytest

from src.visualization import compute_ncc, compute_nrmse


class TestNCC:
    """Tests for normalized cross-correlation."""

    def test_identical_signals(self):
        """NCC of identical signals should be 1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(compute_ncc(x, x), 1.0)

    def test_known_value(self, fixtures_dir):
        """NCC should match precomputed fixture value."""
        fix = np.load(os.path.join(fixtures_dir, 'metrics_fixtures.npz'))
        est = fix['input_estimate']
        ref = fix['input_reference']
        expected = fix['output_ncc'][0]
        result = compute_ncc(est, ref)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_with_mask(self):
        """NCC with mask should only use masked pixels."""
        est = np.array([1.0, 0.0, 3.0])
        ref = np.array([1.0, 999.0, 3.0])
        mask = np.array([True, False, True])
        ncc_masked = compute_ncc(est, ref, mask=mask)
        ncc_clean = compute_ncc(np.array([1.0, 3.0]), np.array([1.0, 3.0]))
        assert np.isclose(ncc_masked, ncc_clean)

    def test_zero_signal(self):
        """NCC with zero signal should return 0."""
        assert compute_ncc(np.zeros(5), np.ones(5)) == 0.0


class TestNRMSE:
    """Tests for normalized root mean square error."""

    def test_identical_signals(self):
        """NRMSE of identical signals should be 0.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isclose(compute_nrmse(x, x), 0.0)

    def test_known_value(self, fixtures_dir):
        """NRMSE should match precomputed fixture value."""
        fix = np.load(os.path.join(fixtures_dir, 'metrics_fixtures.npz'))
        est = fix['input_estimate']
        ref = fix['input_reference']
        expected = fix['output_nrmse'][0]
        result = compute_nrmse(est, ref)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_with_mask(self):
        """NRMSE with mask should only use masked pixels."""
        est = np.array([1.0, 0.0, 3.0])
        ref = np.array([1.0, 999.0, 3.0])
        mask = np.array([True, False, True])
        nrmse = compute_nrmse(est, ref, mask=mask)
        assert np.isclose(nrmse, 0.0)
