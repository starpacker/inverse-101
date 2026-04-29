"""Tests for physics_model module."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics_model import (
    get_object_patch,
    compute_exit_wave,
    fraunhofer_propagate,
    asp_propagate,
    compute_detector_intensity,
    forward_model,
)


class TestGetObjectPatch:
    """Tests for object patch extraction."""

    def test_output_shape(self):
        obj = np.ones((64, 64), dtype=np.complex128)
        patch = get_object_patch(obj, (5, 10), Np=16)
        assert patch.shape == (16, 16)

    def test_correct_values(self):
        """Patch should match the corresponding slice of the object."""
        rng = np.random.default_rng(0)
        obj = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        row, col = 8, 12
        Np = 16
        patch = get_object_patch(obj, (row, col), Np)
        np.testing.assert_allclose(patch, obj[row:row + Np, col:col + Np], rtol=1e-14)

    def test_returns_copy(self):
        """Patch should be a copy, not a view."""
        obj = np.ones((32, 32), dtype=np.complex128)
        patch = get_object_patch(obj, (0, 0), 16)
        patch[:] = 0
        assert np.all(obj[:16, :16] == 1.0)


class TestComputeExitWave:
    """Tests for exit wave computation (thin-object approximation)."""

    def test_output_shape(self):
        probe = np.ones((16, 16), dtype=np.complex128)
        patch = np.ones((16, 16), dtype=np.complex128)
        esw = compute_exit_wave(probe, patch)
        assert esw.shape == (16, 16)

    def test_pointwise_multiplication(self):
        """Exit wave should be element-wise product of probe and object patch."""
        rng = np.random.default_rng(1)
        probe = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        patch = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        esw = compute_exit_wave(probe, patch)
        np.testing.assert_allclose(esw, probe * patch, rtol=1e-14)

    def test_unit_object_preserves_probe(self):
        """Unit object patch should return the probe unchanged."""
        rng = np.random.default_rng(2)
        probe = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        patch = np.ones((8, 8), dtype=np.complex128)
        esw = compute_exit_wave(probe, patch)
        np.testing.assert_allclose(esw, probe, rtol=1e-14)


class TestFraunhoferPropagate:
    """Tests for Fraunhofer (far-field) propagation."""

    def test_output_shape(self):
        esw = np.ones((16, 16), dtype=np.complex128)
        det_field = fraunhofer_propagate(esw)
        assert det_field.shape == (16, 16)

    def test_energy_conservation(self):
        """Unitary FFT should conserve energy: sum|det_field|^2 == sum|esw|^2."""
        rng = np.random.default_rng(3)
        esw = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        det_field = fraunhofer_propagate(esw)
        np.testing.assert_allclose(
            np.sum(np.abs(det_field) ** 2),
            np.sum(np.abs(esw) ** 2),
            rtol=1e-10,
        )

    def test_linearity(self):
        """Propagation should be linear: F(a*x + b*y) = a*F(x) + b*F(y)."""
        rng = np.random.default_rng(4)
        x = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        y = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        a, b = 2.0 + 1j, -0.5 + 0.3j
        lhs = fraunhofer_propagate(a * x + b * y)
        rhs = a * fraunhofer_propagate(x) + b * fraunhofer_propagate(y)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


class TestAspPropagate:
    """Tests for angular spectrum propagation."""

    def test_output_shape(self):
        esw = np.ones((16, 16), dtype=np.complex128)
        det_field = asp_propagate(esw, zo=0.05, wavelength=632.8e-9, L=1e-3)
        assert det_field.shape == (16, 16)

    def test_complex_output(self):
        esw = np.ones((16, 16), dtype=np.complex128)
        det_field = asp_propagate(esw, zo=0.05, wavelength=632.8e-9, L=1e-3)
        assert np.iscomplexobj(det_field)


class TestComputeDetectorIntensity:
    """Tests for detector intensity computation."""

    def test_nonnegative(self):
        """Intensity (|field|^2) should be nonnegative."""
        rng = np.random.default_rng(5)
        field = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        intensity = compute_detector_intensity(field)
        assert np.all(intensity >= 0)

    def test_values(self):
        """Intensity should equal magnitude squared."""
        rng = np.random.default_rng(6)
        field = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        intensity = compute_detector_intensity(field)
        np.testing.assert_allclose(intensity, np.abs(field) ** 2, rtol=1e-14)

    def test_real_output(self):
        rng = np.random.default_rng(7)
        field = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        intensity = compute_detector_intensity(field)
        assert not np.iscomplexobj(intensity)


class TestForwardModel:
    """Tests for the full CP forward model."""

    def test_output_shapes(self):
        rng = np.random.default_rng(8)
        Np, No = 16, 64
        probe = rng.standard_normal((Np, Np)) + 1j * rng.standard_normal((Np, Np))
        obj = rng.standard_normal((No, No)) + 1j * rng.standard_normal((No, No))
        intensity, esw = forward_model(probe, obj, (10, 10), Np)
        assert intensity.shape == (Np, Np)
        assert esw.shape == (Np, Np)

    def test_nonnegative_intensity(self):
        rng = np.random.default_rng(9)
        Np, No = 16, 32
        probe = rng.standard_normal((Np, Np)) + 1j * rng.standard_normal((Np, Np))
        obj = rng.standard_normal((No, No)) + 1j * rng.standard_normal((No, No))
        intensity, _ = forward_model(probe, obj, (4, 4), Np)
        assert np.all(intensity >= 0)

    def test_parseval_theorem(self):
        """Total detector intensity should equal total |esw|^2 (Parseval for unitary FFT)."""
        rng = np.random.default_rng(10)
        Np, No = 16, 48
        probe = rng.standard_normal((Np, Np)) + 1j * rng.standard_normal((Np, Np))
        obj = rng.standard_normal((No, No)) + 1j * rng.standard_normal((No, No))
        intensity, esw = forward_model(probe, obj, (5, 5), Np, propagator="Fraunhofer")
        np.testing.assert_allclose(
            np.sum(intensity), np.sum(np.abs(esw) ** 2), rtol=1e-10
        )

    def test_unknown_propagator_raises(self):
        probe = np.ones((8, 8), dtype=np.complex128)
        obj = np.ones((32, 32), dtype=np.complex128)
        with pytest.raises(ValueError, match="Unknown propagator"):
            forward_model(probe, obj, (0, 0), 8, propagator="invalid")
