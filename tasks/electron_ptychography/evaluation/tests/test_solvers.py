"""Tests for solvers module."""

import os
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.solvers import (
    _compute_com_field,
    _fourier_integrate,
    _cross_correlate_shift,
    _electron_wavelength,
    _build_probe,
    solve_dpc,
    solve_parallax,
    solve_ptychography,
)

FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "solvers"
)


class TestComputeComField:
    @pytest.fixture
    def fix(self):
        return np.load(os.path.join(FIXTURES_DIR, "com_field.npz"))

    def test_values(self, fix):
        com_x, com_y = _compute_com_field(
            fix["input_datacube"], mask=fix["input_mask"]
        )
        np.testing.assert_allclose(com_x, fix["output_com_x"], rtol=1e-10)
        np.testing.assert_allclose(com_y, fix["output_com_y"], rtol=1e-10)

    def test_output_shape(self, fix):
        com_x, com_y = _compute_com_field(fix["input_datacube"])
        assert com_x.shape == (4, 4)
        assert com_y.shape == (4, 4)

    def test_uniform_dp(self):
        """Uniform DP should have CoM at center."""
        data = np.ones((2, 2, 10, 10), dtype=np.float64)
        com_x, com_y = _compute_com_field(data)
        np.testing.assert_allclose(com_x, 4.5, atol=1e-10)
        np.testing.assert_allclose(com_y, 4.5, atol=1e-10)


class TestFourierIntegrate:
    @pytest.fixture
    def fix(self):
        return np.load(os.path.join(FIXTURES_DIR, "fourier_integrate.npz"))

    def test_values(self, fix):
        phase = _fourier_integrate(
            fix["input_com_x"], fix["input_com_y"],
            dx=float(fix["param_dx"]), dy=float(fix["param_dy"]),
        )
        np.testing.assert_allclose(phase, fix["output_phase"], rtol=1e-10)

    def test_zero_gradient(self):
        """Zero gradient field should give zero phase."""
        com_x = np.zeros((8, 8))
        com_y = np.zeros((8, 8))
        phase = _fourier_integrate(com_x, com_y, dx=1.0, dy=1.0)
        np.testing.assert_allclose(phase, 0.0, atol=1e-14)


class TestCrossCorrelateShift:
    @pytest.fixture
    def fix(self):
        return np.load(os.path.join(FIXTURES_DIR, "cross_correlate_shift.npz"))

    def test_values(self, fix):
        sx, sy = _cross_correlate_shift(fix["input_img1"], fix["input_img2"])
        assert sx == float(fix["output_shift_x"])
        assert sy == float(fix["output_shift_y"])

    def test_no_shift(self):
        """Identical images should give zero shift."""
        img = np.random.default_rng(0).random((16, 16))
        sx, sy = _cross_correlate_shift(img, img)
        assert sx == 0.0
        assert sy == 0.0


class TestElectronWavelength:
    @pytest.fixture
    def fix(self):
        return np.load(os.path.join(FIXTURES_DIR, "electron_wavelength.npz"))

    def test_values(self, fix):
        for energy, expected in zip(fix["param_energies"], fix["output_wavelengths"]):
            result = _electron_wavelength(float(energy))
            np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_higher_energy_shorter_wavelength(self):
        """Higher energy electrons should have shorter wavelength."""
        assert _electron_wavelength(300000) < _electron_wavelength(200000)
        assert _electron_wavelength(200000) < _electron_wavelength(80000)

    def test_300kev_approximate(self):
        """300 keV wavelength should be ~0.0197 A (known value)."""
        lam = _electron_wavelength(300000)
        np.testing.assert_allclose(lam, 0.01969, rtol=1e-3)


class TestBuildProbe:
    @pytest.fixture
    def fix(self):
        return np.load(os.path.join(FIXTURES_DIR, "build_probe.npz"))

    def test_values(self, fix):
        probe = _build_probe(
            fix["input_vacuum_probe"],
            defocus=float(fix["param_defocus"]),
            energy=float(fix["param_energy"]),
            R_pixel_size=float(fix["param_R_pixel_size"]),
        )
        np.testing.assert_allclose(probe, fix["output_probe"], rtol=1e-10)

    def test_output_shape(self, fix):
        probe = _build_probe(
            fix["input_vacuum_probe"],
            defocus=float(fix["param_defocus"]),
            energy=float(fix["param_energy"]),
            R_pixel_size=float(fix["param_R_pixel_size"]),
        )
        assert probe.shape == fix["input_vacuum_probe"].shape
        assert np.iscomplexobj(probe)


class TestSolveDpc:
    def test_output_shape(self):
        """DPC should return a phase image with scan dimensions."""
        rng = np.random.default_rng(42)
        dc = rng.random((4, 4, 16, 16)).astype(np.float32)
        mask = np.ones((16, 16), dtype=bool)
        phase = solve_dpc(dc, energy=300000, dp_mask=mask, com_rotation=0.0,
                          R_pixel_size=1.0, max_iter=2)
        assert phase.shape == (4, 4)

    def test_uniform_gives_flat_phase(self):
        """Uniform datacube should produce near-zero phase."""
        dc = np.ones((4, 4, 16, 16), dtype=np.float32)
        mask = np.ones((16, 16), dtype=bool)
        phase = solve_dpc(dc, energy=300000, dp_mask=mask, com_rotation=0.0,
                          R_pixel_size=1.0, max_iter=5)
        np.testing.assert_allclose(phase, 0.0, atol=1e-10)


class TestSolveParallax:
    def test_output_types(self):
        """Parallax should return phase array and aberrations dict."""
        rng = np.random.default_rng(42)
        dc = rng.random((8, 8, 16, 16)).astype(np.float32)
        phase, aberrations = solve_parallax(
            dc, energy=300000, com_rotation=0.0, R_pixel_size=1.0,
        )
        assert phase.shape == (8, 8)
        assert "C1" in aberrations
        assert "rotation_Q_to_R_rads" in aberrations
        assert "transpose" in aberrations


class TestSolvePtychography:
    def test_output_shapes(self):
        """Ptychography should return phase, complex obj, probe, and errors."""
        rng = np.random.default_rng(42)
        Rx, Ry, Qx, Qy = 4, 4, 8, 8
        dc = rng.random((Rx, Ry, Qx, Qy)).astype(np.float32) + 0.1
        probe_vac = np.ones((Qx, Qy), dtype=np.float32)
        probe_vac[2:6, 2:6] = 5.0

        obj_phase, obj_complex, probe_recon, errors = solve_ptychography(
            dc, probe_vac, energy=300000, defocus=100.0,
            com_rotation=0.0, max_iter=2, step_size=0.5,
            batch_fraction=2, seed=42, R_pixel_size=1.0,
        )
        assert obj_phase.ndim == 2
        assert obj_complex.ndim == 2
        assert obj_phase.shape == obj_complex.shape
        assert probe_recon.shape == (Qx, Qy)
        assert len(errors) == 2
        assert all(isinstance(e, float) for e in errors)

    def test_errors_are_finite(self):
        """NMSE values should be finite and non-negative."""
        rng = np.random.default_rng(42)
        dc = rng.random((4, 4, 8, 8)).astype(np.float32) + 0.1
        probe_vac = np.ones((8, 8), dtype=np.float32)
        probe_vac[2:6, 2:6] = 5.0

        _, _, _, errors = solve_ptychography(
            dc, probe_vac, energy=300000, defocus=100.0,
            com_rotation=0.0, max_iter=3, step_size=0.5,
            batch_fraction=2, seed=42, R_pixel_size=1.0,
        )
        for e in errors:
            assert np.isfinite(e)
            assert e >= 0
