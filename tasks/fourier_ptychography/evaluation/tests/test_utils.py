"""Unit tests for src/utils.py."""
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.utils import fft2c, ifft2c, circ, gaussian2D, smooth_amplitude

FIXTURES = Path(__file__).parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# fft2c / ifft2c
# ---------------------------------------------------------------------------

def test_fft2c_deterministic():
    """fft2c must return the same result for the same input."""
    fix = np.load(FIXTURES / "fft2c_roundtrip.npz")
    result = fft2c(fix["input"])
    np.testing.assert_allclose(result, fix["output"], rtol=1e-10, atol=1e-12)


def test_ifft2c_roundtrip():
    """ifft2c(fft2c(x)) must recover x exactly."""
    fix = np.load(FIXTURES / "fft2c_roundtrip.npz")
    recovered = ifft2c(fft2c(fix["input"]))
    np.testing.assert_allclose(recovered, fix["input"], rtol=1e-10, atol=1e-12)


def test_fft2c_unitary():
    """fft2c must preserve the L2 norm (Parseval's theorem)."""
    fix = np.load(FIXTURES / "fft2c_roundtrip.npz")
    x = fix["input"]
    np.testing.assert_allclose(
        np.linalg.norm(fft2c(x)), np.linalg.norm(x), rtol=1e-10
    )


def test_fft2c_dc_at_center():
    """After fft2c, the DC component should be at the array center."""
    n = 16
    x = np.ones((n, n), dtype=complex)
    F = fft2c(x)
    center = n // 2
    assert np.abs(F[center, center]) > np.abs(F[0, 0])


# ---------------------------------------------------------------------------
# circ
# ---------------------------------------------------------------------------

def test_circ_deterministic():
    """circ must return the same mask for the same inputs."""
    fix = np.load(FIXTURES / "circ_mask.npz")
    result = circ(fix["X"], fix["Y"], float(fix["D"]))
    np.testing.assert_array_equal(result, fix["output"])


def test_circ_center_inside():
    """Origin must be inside any circle of positive diameter."""
    X, Y = np.meshgrid([0.0], [0.0])
    assert circ(X, Y, D=1.0)[0, 0]


def test_circ_corner_outside():
    """Far corner of a grid must be outside a small circle."""
    n = 64
    x = np.linspace(-n / 2, n / 2, n)
    X, Y = np.meshgrid(x, x)
    assert not circ(X, Y, D=10.0)[0, 0]


def test_circ_boundary_strict():
    """circ uses strict inequality: point exactly at radius is outside."""
    r = 5.0
    X = np.array([[r]])
    Y = np.array([[0.0]])
    # D = 2r, point at distance r: x²+y² = r² = (D/2)² → NOT strictly less than
    assert not circ(X, Y, D=2 * r)[0, 0]


# ---------------------------------------------------------------------------
# gaussian2D
# ---------------------------------------------------------------------------

def test_gaussian2D_output():
    """gaussian2D must match saved fixture and sum to ~1."""
    expected = np.load(FIXTURES / "output_gaussian2D.npy")
    result = gaussian2D(15, 3.0)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_gaussian2D_sum():
    """Normalized Gaussian kernel must sum to approximately 1."""
    g = gaussian2D(15, 3.0)
    np.testing.assert_allclose(np.sum(g), 1.0, rtol=1e-5)


def test_gaussian2D_shape():
    g = gaussian2D(11, 2.0)
    assert g.shape == (11, 11)


# ---------------------------------------------------------------------------
# smooth_amplitude
# ---------------------------------------------------------------------------

def test_smooth_amplitude_shape():
    """smooth_amplitude must preserve the shape of the input field."""
    rng = np.random.default_rng(0)
    field = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
    out = smooth_amplitude(field, width=3, aleph=0.1)
    assert out.shape == field.shape


def test_smooth_amplitude_aleph_zero():
    """With aleph=0, output must equal input exactly."""
    rng = np.random.default_rng(0)
    field = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    out = smooth_amplitude(field, width=3, aleph=0.0)
    np.testing.assert_allclose(out, field, rtol=1e-10)
