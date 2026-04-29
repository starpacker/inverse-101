"""Unit tests for visualization.py."""

import os
import sys
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.visualization import compute_metrics, compute_batch_metrics

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/visualization")


class TestComputeMetrics:
    def test_perfect_match(self):
        ref = np.random.rand(64, 64)
        m = compute_metrics(ref, ref)
        np.testing.assert_allclose(m["nrmse"], 0.0, atol=1e-10)
        np.testing.assert_allclose(m["ncc"], 1.0, atol=1e-10)

    def test_scaled_copy(self):
        """A scaled copy should have NCC close to 1."""
        ref = np.random.rand(64, 64)
        est = 2.0 * ref
        m = compute_metrics(est, ref)
        np.testing.assert_allclose(m["ncc"], 1.0, atol=1e-10)

    def test_orthogonal(self):
        """Orthogonal signals should have NCC close to 0."""
        ref = np.array([1.0, 0.0, 0.0])
        est = np.array([0.0, 1.0, 0.0])
        m = compute_metrics(est, ref)
        np.testing.assert_allclose(m["ncc"], 0.0, atol=1e-10)

    def test_nrmse_known(self):
        ref = np.ones(100)
        est = np.ones(100) * 1.1
        m = compute_metrics(est, ref)
        # NRMSE = 0.1 / (1 - 1) => inf since dynamic range is 0
        # Use a reference with nonzero range
        ref2 = np.linspace(0, 1, 100)
        est2 = ref2 + 0.01
        m2 = compute_metrics(est2, ref2)
        assert m2["nrmse"] < 0.02

    def test_fixture_consistency(self):
        with open(os.path.join(FIXTURE_DIR, "output_metrics_sample0.json")) as f:
            expected = json.load(f)
        assert "ncc" in expected
        assert "nrmse" in expected
        assert "psnr" in expected
        assert expected["ncc"] > 0.8
        assert expected["nrmse"] < 0.2


class TestComputeBatchMetrics:
    def test_batch_keys(self):
        refs = np.random.rand(3, 64, 64)
        m = compute_batch_metrics(refs, refs)
        assert "per_sample" in m
        assert "avg_nrmse" in m
        assert "avg_ncc" in m
        assert "avg_psnr" in m

    def test_batch_perfect(self):
        refs = np.random.rand(3, 64, 64)
        m = compute_batch_metrics(refs, refs)
        np.testing.assert_allclose(m["avg_nrmse"], 0.0, atol=1e-10)
        np.testing.assert_allclose(m["avg_ncc"], 1.0, atol=1e-10)
        assert len(m["per_sample"]) == 3
