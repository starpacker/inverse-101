"""Tests for utils module."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils import fft2c, ifft2c, circ, gaussian2D, cart2pol, aspw, GenerateNonUniformFermat, smooth_amplitude


class TestFft2cIfft2c:
    """Tests for centered unitary FFT/IFFT pair."""

    def test_roundtrip_identity(self):
        """ifft2c(fft2c(x)) should recover x exactly."""
        rng = np.random.default_rng(0)
        field = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        roundtrip = ifft2c(fft2c(field))
        np.testing.assert_allclose(roundtrip, field, rtol=1e-10)

    def test_parseval_energy_conservation(self):
        """Unitary FFT should preserve total energy: sum|X|^2 == sum|x|^2."""
        rng = np.random.default_rng(1)
        field = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        energy_spatial = np.sum(np.abs(field) ** 2)
        energy_freq = np.sum(np.abs(fft2c(field)) ** 2)
        np.testing.assert_allclose(energy_freq, energy_spatial, rtol=1e-10)

    def test_output_shape_and_dtype(self):
        """Output should have the same shape and complex dtype."""
        field = np.ones((8, 8), dtype=np.complex128)
        result = fft2c(field)
        assert result.shape == (8, 8)
        assert np.iscomplexobj(result)

    def test_dc_component_for_constant(self):
        """FFT of a constant field should concentrate energy at DC (center)."""
        field = np.ones((16, 16), dtype=np.complex128)
        F = fft2c(field)
        # DC is at center (8, 8); its magnitude should equal N (=16) for ortho norm
        dc_val = np.abs(F[8, 8])
        np.testing.assert_allclose(dc_val, 16.0, rtol=1e-10)


class TestCirc:
    """Tests for circular aperture function."""

    def test_output_shape(self):
        x, y = np.meshgrid(np.arange(16) - 8, np.arange(16) - 8)
        mask = circ(x.astype(float), y.astype(float), 10.0)
        assert mask.shape == (16, 16)

    def test_center_inside(self):
        """The center of the grid should always be inside the circle."""
        x, y = np.meshgrid(np.arange(32) - 16, np.arange(32) - 16)
        mask = circ(x.astype(float), y.astype(float), 20.0)
        assert mask[16, 16] == True

    def test_far_corner_outside(self):
        """Corners far from center should be outside the circle."""
        x, y = np.meshgrid(np.arange(32) - 16, np.arange(32) - 16)
        mask = circ(x.astype(float), y.astype(float), 10.0)
        assert mask[0, 0] == False


class TestGaussian2D:
    """Tests for 2D Gaussian kernel."""

    def test_normalization(self):
        """Kernel should sum to 1."""
        h = gaussian2D(5, 1.0)
        np.testing.assert_allclose(np.sum(h), 1.0, rtol=1e-10)

    def test_shape(self):
        """Kernel size should be (2*(n-1)//2+1)^2 = (2*2+1)^2 = 5x5 for n=5."""
        h = gaussian2D(5, 1.0)
        assert h.shape == (5, 5)

    def test_symmetry(self):
        """Kernel should be symmetric."""
        h = gaussian2D(7, 2.0)
        np.testing.assert_allclose(h, h.T, rtol=1e-14)
        np.testing.assert_allclose(h, h[::-1, :], rtol=1e-14)


class TestCart2pol:
    """Tests for Cartesian to polar conversion."""

    def test_known_values(self):
        x = np.array([1.0, 0.0, -1.0])
        y = np.array([0.0, 1.0, 0.0])
        th, r = cart2pol(x, y)
        np.testing.assert_allclose(r, [1.0, 1.0, 1.0], rtol=1e-10)
        np.testing.assert_allclose(th, [0.0, np.pi / 2, np.pi], rtol=1e-10)

    def test_origin(self):
        th, r = cart2pol(np.array([0.0]), np.array([0.0]))
        assert r[0] == 0.0


class TestAspw:
    """Tests for angular spectrum propagator."""

    def test_output_shape(self):
        field = np.ones((16, 16), dtype=np.complex128)
        u_prop, H = aspw(field, z=1e-3, wavelength=632.8e-9, L=1e-3)
        assert u_prop.shape == (16, 16)
        assert H.shape == (16, 16)

    def test_zero_distance_identity(self):
        """Propagation by z=0 should return the original field."""
        rng = np.random.default_rng(3)
        field = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        u_prop, _ = aspw(field, z=0.0, wavelength=632.8e-9, L=1e-3, bandlimit=False)
        np.testing.assert_allclose(u_prop, field, rtol=1e-10)

    def test_energy_conservation(self):
        """Propagation should approximately conserve energy (with bandlimit some loss is OK)."""
        rng = np.random.default_rng(4)
        field = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        energy_in = np.sum(np.abs(field) ** 2)
        u_prop, _ = aspw(field, z=1e-3, wavelength=632.8e-9, L=0.5e-3, bandlimit=False)
        energy_out = np.sum(np.abs(u_prop) ** 2)
        np.testing.assert_allclose(energy_out, energy_in, rtol=1e-6)


class TestGenerateNonUniformFermat:
    """Tests for Fermat spiral scan grid generator."""

    def test_output_length(self):
        R, C = GenerateNonUniformFermat(50, radius=100)
        assert len(R) == 50
        assert len(C) == 50

    def test_first_point_at_origin(self):
        """First Fermat point (index 0) should be at the origin."""
        R, C = GenerateNonUniformFermat(10, radius=100)
        np.testing.assert_allclose(R[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(C[0], 0.0, atol=1e-12)

    def test_points_within_radius(self):
        R, C = GenerateNonUniformFermat(100, radius=200)
        distances = np.sqrt(R ** 2 + C ** 2)
        assert np.all(distances <= 200.0 + 1e-10)
