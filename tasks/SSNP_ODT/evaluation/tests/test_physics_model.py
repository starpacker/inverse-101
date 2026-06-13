"""Tests for physics_model module."""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics_model import SSNPConfig, SSNPForwardModel


def _small_config(n_angles=2):
    """Create a small SSNPConfig for testing."""
    return SSNPConfig(
        volume_shape=(4, 16, 16),
        res=(0.3, 0.3, 0.5),
        n0=1.33,
        NA=0.55,
        wavelength_um=0.532,
        res_um=(0.1, 0.1, 0.2),
        n_angles=n_angles,
    )


def _make_model(config=None):
    if config is None:
        config = _small_config()
    return SSNPForwardModel(config, device="cpu")


# ── SSNPConfig ───────────────────────────────────────────────────────────

class TestSSNPConfig:
    def test_from_metadata(self):
        """from_metadata should correctly compute normalised resolution."""
        metadata = {
            "volume_shape": [8, 16, 16],
            "res_um": [0.1, 0.1, 0.2],
            "n0": 1.33,
            "NA": 0.55,
            "wavelength_um": 0.532,
            "n_angles": 4,
            "ri_contrast_scale": 0.01,
            "tiff_scale": 1.0,
        }
        config = SSNPConfig.from_metadata(metadata)

        assert config.volume_shape == (8, 16, 16)
        assert config.n0 == 1.33
        assert config.NA == 0.55
        assert config.n_angles == 4

        # res = res_um * n0 / wavelength
        expected_res = tuple(s * 1.33 / 0.532 for s in [0.1, 0.1, 0.2])
        np.testing.assert_allclose(config.res, expected_res, rtol=1e-10)

    def test_from_metadata_preserves_res_um(self):
        """res_um should be stored unchanged."""
        metadata = {
            "volume_shape": [4, 8, 8],
            "res_um": [0.15, 0.15, 0.3],
            "n0": 1.0,
            "NA": 0.65,
            "wavelength_um": 0.5,
            "n_angles": 8,
            "ri_contrast_scale": 0.01,
            "tiff_scale": 1.0,
        }
        config = SSNPConfig.from_metadata(metadata)
        assert config.res_um == (0.15, 0.15, 0.3)


# ── Precomputed quantities ───────────────────────────────────────────────

class TestPrecomputedQuantities:
    def test_kz_shape(self):
        """kz should have shape (Ny, Nx)."""
        model = _make_model()
        assert model.kz.shape == (16, 16)

    def test_kz_positive(self):
        """kz values should be positive (clamped away from zero)."""
        model = _make_model()
        assert torch.all(model.kz > 0)

    def test_evanescent_mask_range(self):
        """Evanescent mask values should be in [0, 1]."""
        model = _make_model()
        mask = model.eva_mask
        assert mask.shape == (16, 16)
        assert torch.all(mask >= 0)
        assert torch.all(mask <= 1.0 + 1e-10)

    def test_pupil_shape(self):
        """Pupil should have shape (Ny, Nx) and be complex."""
        model = _make_model()
        assert model.pupil.shape == (16, 16)
        assert model.pupil.is_complex()

    def test_c_gamma_shape_and_range(self):
        """c_gamma should be (Ny, Nx) with positive values."""
        model = _make_model()
        cg = model.c_gamma
        assert cg.shape == (16, 16)
        assert torch.all(cg > 0)


# ── Incident field ───────────────────────────────────────────────────────

class TestIncidentField:
    def test_incident_field_shapes(self):
        """Incident field u and ud should have shape (Ny, Nx)."""
        model = _make_model()
        u, ud = model._make_incident_field(0)
        assert u.shape == (16, 16)
        assert ud.shape == (16, 16)
        assert u.is_complex()
        assert ud.is_complex()

    def test_incident_field_unit_amplitude(self):
        """Tilted plane wave should have roughly unit amplitude everywhere."""
        model = _make_model()
        u, ud = model._make_incident_field(0)
        amplitudes = torch.abs(u).cpu().numpy()
        np.testing.assert_allclose(amplitudes, 1.0, atol=1e-10)

    def test_different_angles_give_different_fields(self):
        """Different angle indices should produce different incident fields."""
        config = _small_config(n_angles=4)
        model = _make_model(config)
        u0, _ = model._make_incident_field(0)
        u1, _ = model._make_incident_field(1)
        assert not torch.allclose(u0, u1), "Different angles should produce different fields"


# ── P operator (propagation) ─────────────────────────────────────────────

class TestPropagation:
    def test_propagation_preserves_shape(self):
        """P operator should preserve (Ny, Nx) shape."""
        model = _make_model()
        u = torch.ones(16, 16, dtype=torch.complex128)
        ud = torch.zeros(16, 16, dtype=torch.complex128)
        u_new, ud_new = model._apply_propagation(u, ud)
        assert u_new.shape == (16, 16)
        assert ud_new.shape == (16, 16)

    def test_propagation_zero_distance_identity(self):
        """Propagation by dz=0 should be (approximately) identity."""
        model = _make_model()
        rng = np.random.default_rng(42)
        u = torch.tensor(rng.random((16, 16)) + 1j * rng.random((16, 16)),
                         dtype=torch.complex128)
        ud = torch.tensor(rng.random((16, 16)) + 1j * rng.random((16, 16)),
                          dtype=torch.complex128)
        u_new, ud_new = model._apply_propagation(u, ud, dz=0.0)

        # cos(0)=1, sin(0)=0, so P is identity (modulo evanescent mask)
        # For non-evanescent components this should be very close
        np.testing.assert_allclose(u_new.cpu().numpy(), u.cpu().numpy(), atol=1e-6)
        np.testing.assert_allclose(ud_new.cpu().numpy(), ud.cpu().numpy(), atol=1e-6)

    def test_propagation_energy_conservation(self):
        """Propagation should approximately conserve energy (Parseval)."""
        model = _make_model()
        rng = np.random.default_rng(99)
        u = torch.tensor(rng.random((16, 16)) + 1j * rng.random((16, 16)),
                         dtype=torch.complex128)
        ud = torch.zeros(16, 16, dtype=torch.complex128)

        energy_before = torch.sum(torch.abs(u) ** 2).item()
        u_new, _ = model._apply_propagation(u, ud, dz=1.0)
        energy_after = torch.sum(torch.abs(u_new) ** 2).item()

        # Energy may decrease slightly due to evanescent damping, but should not increase
        assert energy_after <= energy_before * 1.01, (
            f"Energy increased: {energy_before:.6f} -> {energy_after:.6f}"
        )


# ── Q operator (scattering) ─────────────────────────────────────────────

class TestScattering:
    def test_scattering_preserves_u(self):
        """Q operator should not modify u (only ud changes)."""
        model = _make_model()
        u = torch.ones(16, 16, dtype=torch.complex128) * (1.0 + 0.5j)
        ud = torch.zeros(16, 16, dtype=torch.complex128)
        dn_slice = torch.ones(16, 16, dtype=torch.float64) * 0.01

        u_out, ud_out = model._apply_scattering(u, ud, dn_slice)
        np.testing.assert_allclose(u_out.cpu().numpy(), u.cpu().numpy(), atol=1e-15)

    def test_scattering_zero_contrast_identity(self):
        """Zero RI contrast should leave both u and ud unchanged."""
        model = _make_model()
        u = torch.ones(16, 16, dtype=torch.complex128) * (1.0 + 0.5j)
        ud = torch.ones(16, 16, dtype=torch.complex128) * 0.3j
        dn_slice = torch.zeros(16, 16, dtype=torch.float64)

        u_out, ud_out = model._apply_scattering(u, ud, dn_slice)
        np.testing.assert_allclose(u_out.cpu().numpy(), u.cpu().numpy(), atol=1e-15)
        np.testing.assert_allclose(ud_out.cpu().numpy(), ud.cpu().numpy(), atol=1e-15)

    def test_scattering_modifies_ud(self):
        """Non-zero RI contrast should modify ud."""
        model = _make_model()
        u = torch.ones(16, 16, dtype=torch.complex128)
        ud = torch.zeros(16, 16, dtype=torch.complex128)
        dn_slice = torch.ones(16, 16, dtype=torch.float64) * 0.05

        _, ud_out = model._apply_scattering(u, ud, dn_slice)
        assert not torch.allclose(ud_out, torch.zeros_like(ud_out)), \
            "Scattering with nonzero Δn should modify ud"


# ── Forward model ────────────────────────────────────────────────────────

class TestForwardModel:
    def test_forward_single_output_shape(self):
        """forward_single should return (Ny, Nx) intensity."""
        model = _make_model()
        nz, ny, nx = model.config.volume_shape
        dn = torch.zeros(nz, ny, nx, dtype=torch.float64)
        intensity = model.forward_single(dn, 0)
        assert intensity.shape == (ny, nx)

    def test_forward_output_shape(self):
        """forward should return (n_angles, Ny, Nx) intensities."""
        model = _make_model()
        nz, ny, nx = model.config.volume_shape
        dn = torch.zeros(nz, ny, nx, dtype=torch.float64)
        intensities = model.forward(dn)
        assert intensities.shape == (model.config.n_angles, ny, nx)

    def test_forward_nonnegative_intensity(self):
        """Intensities should be non-negative (|phi|^2 >= 0)."""
        model = _make_model()
        nz, ny, nx = model.config.volume_shape
        rng = np.random.default_rng(42)
        dn = torch.tensor(rng.uniform(-0.01, 0.01, (nz, ny, nx)), dtype=torch.float64)
        with torch.no_grad():
            intensities = model.forward(dn)
        assert torch.all(intensities >= 0)

    def test_simulate_measurements_matches_forward(self):
        """simulate_measurements should be identical to forward."""
        model = _make_model()
        nz, ny, nx = model.config.volume_shape
        dn = torch.zeros(nz, ny, nx, dtype=torch.float64)
        with torch.no_grad():
            I1 = model.forward(dn)
            I2 = model.simulate_measurements(dn)
        np.testing.assert_allclose(I1.cpu().numpy(), I2.cpu().numpy(), atol=1e-15)

    def test_forward_differentiable(self):
        """Forward model should support autograd (gradient computation)."""
        model = _make_model()
        nz, ny, nx = model.config.volume_shape
        dn = torch.zeros(nz, ny, nx, dtype=torch.float64, requires_grad=True)
        intensity = model.forward_single(dn, 0)
        loss = intensity.sum()
        loss.backward()
        assert dn.grad is not None
        assert dn.grad.shape == (nz, ny, nx)


# ── Fixture-based tests ─────────────────────────────────────────────────

class TestFixtureParity:
    """Tests against precomputed fixture data."""

    @pytest.fixture
    def fixture_data(self):
        fixture_path = os.path.join(
            os.path.dirname(__file__), "..", "fixtures", "basic_pipeline.npz"
        )
        if not os.path.exists(fixture_path):
            pytest.skip("Fixture file basic_pipeline.npz not found")
        return np.load(fixture_path)

    def test_kz_matches_fixture(self, fixture_data):
        """kz grid should match the precomputed fixture."""
        config = SSNPConfig(
            volume_shape=(4, 16, 16),
            res=(0.3, 0.3, 0.5),
            n0=1.33,
            NA=0.55,
            wavelength_um=0.532,
            res_um=(0.1, 0.1, 0.2),
            n_angles=3,
        )
        model = SSNPForwardModel(config, device="cpu")
        kz_computed = model.kz.cpu().numpy()
        kz_fixture = fixture_data["kz"]
        np.testing.assert_allclose(kz_computed, kz_fixture, rtol=1e-10)

    def test_intensity_matches_fixture(self, fixture_data):
        """Forward model intensity should match the precomputed fixture."""
        config = SSNPConfig(
            volume_shape=(4, 16, 16),
            res=(0.3, 0.3, 0.5),
            n0=1.33,
            NA=0.55,
            wavelength_um=0.532,
            res_um=(0.1, 0.1, 0.2),
            n_angles=3,
        )
        model = SSNPForwardModel(config, device="cpu")
        delta_n = fixture_data["delta_n"]
        dn_tensor = torch.tensor(delta_n, dtype=torch.float64)

        with torch.no_grad():
            intensity = model.forward(dn_tensor).cpu().numpy()

        intensity_fixture = fixture_data["intensity"]
        np.testing.assert_allclose(intensity, intensity_fixture, rtol=1e-6, atol=1e-10)
