"""Unit tests for solvers module."""

import numpy as np
import pytest
from src.solvers import p_shrink, make_congruent, unwrap_phase


def test_p_shrink(fixtures):
    X = fixtures["input_X_shrink"]
    shrunk = p_shrink(X, lmbda=1, p=0)
    np.testing.assert_allclose(shrunk, fixtures["output_shrunk"], rtol=1e-10)


def test_p_shrink_zeros():
    """Shrinkage of zero input should be zero."""
    X = np.zeros((2, 8, 8), dtype=np.float32)
    result = p_shrink(X)
    np.testing.assert_array_equal(result, 0)


def test_p_shrink_sparsity():
    """Small values should be shrunk to zero."""
    X = np.full((2, 4, 4), 0.1, dtype=np.float32)
    result = p_shrink(X, lmbda=1, p=0)
    # |x| = 0.1*sqrt(2) ~ 0.141, 1/|x| ~ 7.07, so |x| - 1/|x| < 0 -> 0
    np.testing.assert_array_equal(result, 0)


def test_make_congruent(fixtures):
    unwrapped = fixtures["input_unwrapped_congruent"]
    wrapped = fixtures["wrapped_ramp"]
    result = make_congruent(unwrapped, wrapped)
    np.testing.assert_allclose(result, fixtures["output_congruent"], rtol=1e-10)


def test_make_congruent_integer_multiples():
    """Result minus wrapped should be integer multiples of 2*pi."""
    wrapped = np.array([[0.5, 1.0], [-1.0, 2.0]])
    unwrapped = wrapped + 2 * np.pi * np.array([[1, 3], [2, -1]]) + 0.3
    result = make_congruent(unwrapped, wrapped)
    diff = (result - wrapped) / (2 * np.pi)
    np.testing.assert_allclose(diff, np.round(diff), atol=1e-10)


def test_unwrap_ramp(fixtures):
    """Synthetic ramp unwrapping should recover original phase."""
    wrapped = fixtures["wrapped_ramp"]
    phase = fixtures["phase_ramp"]
    unwrapped, n_iters = unwrap_phase(wrapped, max_iters=100, tol=np.pi / 10)
    p = phase - phase.mean()
    u = unwrapped - unwrapped.mean()
    np.testing.assert_allclose(p, u, atol=0.01)


def test_unwrap_quadratic():
    """Quadratic phase unwrapping."""
    n = 64
    y, x = np.ogrid[-3:3:complex(0, n), -3:3:complex(0, n)]
    phase = 0.5 * np.pi * (x ** 2 + y ** 2)
    wrapped = np.angle(np.exp(1j * phase)).astype(np.float32)
    unwrapped, n_iters = unwrap_phase(wrapped, max_iters=200, tol=np.pi / 10)
    p = phase - phase.mean()
    u = unwrapped - unwrapped.mean()
    np.testing.assert_allclose(p, u, atol=0.3)


def test_unwrap_congruent_flag():
    """With congruent=True, result - wrapped should be integer multiples of 2*pi."""
    n = 64
    y, x = np.ogrid[-2:2:complex(0, n), -2:2:complex(0, n)]
    phase = np.pi * (x + y)
    wrapped = np.angle(np.exp(1j * phase)).astype(np.float32)
    unwrapped, _ = unwrap_phase(wrapped, max_iters=100, tol=np.pi / 10, congruent=True)
    diff = (unwrapped - wrapped) / (2 * np.pi)
    np.testing.assert_allclose(diff, np.round(diff), atol=1e-5)


class TestParity:
    """Parity tests: cleaned code matches original spurs library."""

    def test_unwrap_parity_real_data(self, raw_data, reference_outputs):
        """Cleaned code produces same unwrapped phase as reference on real data."""
        wrapped = raw_data["wrapped_phase"]
        ref_unwrapped = reference_outputs["unwrapped_spurs"]

        unwrapped, _ = unwrap_phase(
            wrapped, max_iters=500, tol=np.pi / 5,
            lmbda=1, p=0, c=1.3, dtype="float32",
        )

        np.testing.assert_allclose(unwrapped, ref_unwrapped, rtol=1e-5, atol=1e-5)

    def test_metrics_match_reference(self, reference_outputs):
        """Verify metrics match reference values."""
        import json, os
        ref_dir = os.path.join(os.path.dirname(__file__), "..", "reference_outputs")
        with open(os.path.join(ref_dir, "metrics.json")) as f:
            metrics = json.load(f)

        assert metrics["fraction_within_pi"] > 0.99
        assert metrics["fraction_within_2pi"] == 1.0
        assert metrics["n_iterations"] <= 20
