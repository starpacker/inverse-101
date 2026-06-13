"""Unit tests for physics_model module."""

import numpy as np
from src.physics_model import (
    apply_gradient_x,
    apply_gradient_y,
    apply_divergence,
    make_laplace_kernel,
    solve_poisson_dct,
)


def test_apply_gradient_x(fixtures):
    phase = fixtures["input_phase_for_gradient"]
    gx = apply_gradient_x(phase)
    np.testing.assert_allclose(gx, fixtures["output_gradient_x"], rtol=1e-10)


def test_apply_gradient_y(fixtures):
    phase = fixtures["input_phase_for_gradient"]
    gy = apply_gradient_y(phase)
    np.testing.assert_allclose(gy, fixtures["output_gradient_y"], rtol=1e-10)


def test_gradient_neumann_bc():
    """Last column of Dx and last row of Dy must be zero (Neumann BC)."""
    arr = np.random.randn(16, 16).astype(np.float32)
    gx = apply_gradient_x(arr)
    gy = apply_gradient_y(arr)
    np.testing.assert_array_equal(gx[:, -1], 0)
    np.testing.assert_array_equal(gy[-1, :], 0)


def test_apply_divergence(fixtures):
    gx = fixtures["input_gx_for_div"]
    gy = fixtures["input_gy_for_div"]
    div = apply_divergence(gx, gy)
    np.testing.assert_allclose(div, fixtures["output_divergence"], rtol=1e-10)


def test_divergence_is_adjoint():
    """<Dv, u> = <v, div(u)> (divergence is adjoint of gradient for Neumann BCs)."""
    rng = np.random.default_rng(123)
    v = rng.standard_normal((32, 32)).astype(np.float64)
    ux = rng.standard_normal((32, 32)).astype(np.float64)
    uy = rng.standard_normal((32, 32)).astype(np.float64)

    gx_v = apply_gradient_x(v)
    gy_v = apply_gradient_y(v)
    div_u = apply_divergence(ux, uy)

    lhs = np.sum(gx_v * ux) + np.sum(gy_v * uy)
    rhs = np.sum(v * div_u)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_make_laplace_kernel(fixtures):
    K = make_laplace_kernel(32, 32)
    np.testing.assert_allclose(K, fixtures["output_laplace_kernel"], rtol=1e-10)


def test_laplace_kernel_dc_zero():
    """DC component (0,0) eigenvalue should be zero."""
    K = make_laplace_kernel(64, 64)
    assert K[0, 0] == 0.0


def test_solve_poisson_dct(fixtures):
    rhs = fixtures["input_rhs"]
    K = fixtures["output_laplace_kernel"]
    sol = solve_poisson_dct(rhs, K)
    np.testing.assert_allclose(sol, fixtures["output_poisson_sol"], rtol=1e-5)


def test_poisson_roundtrip():
    """Gradient of a smooth field -> divergence -> Poisson solve recovers field (up to constant)."""
    rng = np.random.default_rng(0)
    n = 32
    # Smooth field
    y, x = np.ogrid[0:1:complex(0, n), 0:1:complex(0, n)]
    field = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    field = field.astype(np.float32)

    gx = apply_gradient_x(field)
    gy = apply_gradient_y(field)
    rhs = apply_divergence(gx, gy)
    K = make_laplace_kernel(n, n)
    recovered = solve_poisson_dct(rhs, K)

    # Should match up to a constant
    diff = (field - field.mean()) - (recovered - recovered.mean())
    np.testing.assert_allclose(diff, 0, atol=1e-4)
