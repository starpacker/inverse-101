"""Tests for generate_data module."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.generate_data import (
    generate_probe,
    generate_usaf_object,
    generate_scan_grid,
    generate_ptychogram,
)


class TestGenerateProbe:
    """Tests for focused probe generation."""

    def test_output_shape(self):
        """Probe should be Nd x Nd."""
        probe, dxp = generate_probe(
            wavelength=632.8e-9, zo=5e-2, Nd=32, dxd=7.2e-5
        )
        assert probe.shape == (32, 32)

    def test_complex_output(self):
        probe, dxp = generate_probe(
            wavelength=632.8e-9, zo=5e-2, Nd=32, dxd=7.2e-5
        )
        assert np.iscomplexobj(probe)

    def test_pixel_size_formula(self):
        """dxp should equal wavelength * zo / (Nd * dxd)."""
        wl, zo, Nd, dxd = 632.8e-9, 5e-2, 64, 7.2e-5
        _, dxp = generate_probe(wl, zo, Nd, dxd)
        expected = wl * zo / (Nd * dxd)
        np.testing.assert_allclose(dxp, expected, rtol=1e-10)

    def test_nonzero_energy(self):
        """Probe should have nonzero energy."""
        probe, _ = generate_probe(
            wavelength=632.8e-9, zo=5e-2, Nd=32, dxd=7.2e-5
        )
        assert np.sum(np.abs(probe) ** 2) > 0


class TestGenerateUsafObject:
    """Tests for USAF 1951 phase test object."""

    def test_output_shape(self):
        obj = generate_usaf_object(No=64, dxp=3.5e-6)
        assert obj.shape == (64, 64)

    def test_complex_dtype(self):
        obj = generate_usaf_object(No=64, dxp=3.5e-6)
        assert np.iscomplexobj(obj)

    def test_unit_amplitude(self):
        """Pure-phase object should have unit amplitude everywhere."""
        obj = generate_usaf_object(No=128, dxp=3.5e-6)
        np.testing.assert_allclose(np.abs(obj), 1.0, rtol=1e-5)

    def test_phase_range(self):
        """Phase values should be in [0, phi_max]."""
        phi_max = np.pi / 2
        obj = generate_usaf_object(No=128, dxp=3.5e-6, phi_max=phi_max)
        phase = np.angle(obj)
        # phase should be 0 (background) or phi_max (bars), both >= 0
        assert np.min(phase) >= -1e-6
        assert np.max(phase) <= phi_max + 1e-6


class TestGenerateScanGrid:
    """Tests for Fermat scan grid generation."""

    def test_output_shapes(self):
        positions, encoder, No = generate_scan_grid(Np=32, dxp=3.5e-6, num_points=20)
        assert positions.shape == (20, 2)
        assert encoder.shape == (20, 2)

    def test_No_is_even(self):
        """Object array size should be even."""
        _, _, No = generate_scan_grid(Np=32, dxp=3.5e-6, num_points=50)
        assert No % 2 == 0

    def test_positions_within_object(self):
        """All scan positions should allow a full Np x Np patch within the object."""
        Np = 32
        positions, _, No = generate_scan_grid(Np=Np, dxp=3.5e-6, num_points=50, radius=50)
        assert np.all(positions >= 0)
        assert np.all(positions[:, 0] + Np <= No)
        assert np.all(positions[:, 1] + Np <= No)

    def test_No_at_least_Np(self):
        """Object size should be at least Np."""
        Np = 64
        _, _, No = generate_scan_grid(Np=Np, dxp=3.5e-6, num_points=10, radius=10)
        assert No >= Np


class TestGeneratePtychogram:
    """Tests for ptychographic diffraction data simulation."""

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        No, Np = 64, 16
        obj = np.exp(1j * rng.uniform(0, np.pi, (No, No))).astype(np.complex128)
        probe = rng.standard_normal((Np, Np)) + 1j * rng.standard_normal((Np, Np))
        positions = np.array([[10, 10], [20, 20], [30, 30]])
        ptycho = generate_ptychogram(obj, probe, positions, Nd=Np)
        assert ptycho.shape == (3, Np, Np)

    def test_nonnegative_values(self):
        """Diffraction intensities should be nonnegative (clipped)."""
        rng = np.random.default_rng(1)
        No, Np = 64, 16
        obj = np.exp(1j * rng.uniform(0, np.pi, (No, No))).astype(np.complex128)
        probe = rng.standard_normal((Np, Np)) + 1j * rng.standard_normal((Np, Np))
        positions = np.array([[5, 5], [15, 15]])
        ptycho = generate_ptychogram(obj, probe, positions, Nd=Np, seed=42)
        assert np.all(ptycho >= 0)

    def test_float32_dtype(self):
        rng = np.random.default_rng(2)
        No, Np = 64, 16
        obj = np.exp(1j * rng.uniform(0, np.pi, (No, No))).astype(np.complex128)
        probe = rng.standard_normal((Np, Np)) + 1j * rng.standard_normal((Np, Np))
        positions = np.array([[10, 10]])
        ptycho = generate_ptychogram(obj, probe, positions, Nd=Np)
        assert ptycho.dtype == np.float32
