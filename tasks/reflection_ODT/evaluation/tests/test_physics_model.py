"""Tests for the physics_model module (ReflectionBPMConfig and ReflectionBPMForwardModel)."""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics_model import ReflectionBPMConfig, ReflectionBPMForwardModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_metadata():
    return {
        "volume_shape": [2, 16, 16],
        "n0": 1.5,
        "NA_obj": 0.55,
        "wavelength_um": 0.532,
        "res_um": [0.1, 0.1, 0.5],
        "ri_contrast": 0.02,
        "illumination_rings": [
            {"NA": 0.3, "n_angles": 2, "type": "BF"},
        ],
        "dz_layer": 0.5,
        "dz_gap": 10.0,
    }


def _build_model(metadata=None):
    if metadata is None:
        metadata = _small_metadata()
    config = ReflectionBPMConfig.from_metadata(metadata)
    return ReflectionBPMForwardModel(config, device="cpu"), config


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestReflectionBPMConfig:
    def test_from_metadata_volume_shape(self):
        config = ReflectionBPMConfig.from_metadata(_small_metadata())
        assert config.volume_shape == (2, 16, 16)

    def test_from_metadata_n_angles(self):
        meta = _small_metadata()
        meta["illumination_rings"] = [
            {"NA": 0.2, "n_angles": 4, "type": "BF"},
            {"NA": 0.6, "n_angles": 6, "type": "DF"},
        ]
        config = ReflectionBPMConfig.from_metadata(meta)
        assert config.n_angles == 10

    def test_res_normalization(self):
        """res should equal res_um * n0 / wavelength."""
        meta = _small_metadata()
        config = ReflectionBPMConfig.from_metadata(meta)
        expected = tuple(r * meta["n0"] / meta["wavelength_um"]
                         for r in meta["res_um"])
        np.testing.assert_allclose(config.res, expected, rtol=1e-12)

    def test_default_dz_values(self):
        meta = _small_metadata()
        del meta["dz_layer"]
        del meta["dz_gap"]
        config = ReflectionBPMConfig.from_metadata(meta)
        assert config.dz_layer == 0.5
        assert config.dz_gap == 10.0


# ---------------------------------------------------------------------------
# Precomputed kernel tests
# ---------------------------------------------------------------------------


class TestPrecomputedKernels:
    def test_c_gamma_shape(self):
        model, config = _build_model()
        _, ny, nx = config.volume_shape
        assert model.c_gamma.shape == (ny, nx)

    def test_c_gamma_positive(self):
        model, _ = _build_model()
        assert torch.all(model.c_gamma > 0)

    def test_kz_shape(self):
        model, config = _build_model()
        _, ny, nx = config.volume_shape
        assert model.kz.shape == (ny, nx)

    def test_evanescent_mask_range(self):
        """Evanescent mask values should be in (0, 1]."""
        model, _ = _build_model()
        assert torch.all(model.eva_mask > 0)
        assert torch.all(model.eva_mask <= 1.0 + 1e-12)

    def test_pupil_shape_and_dtype(self):
        model, config = _build_model()
        _, ny, nx = config.volume_shape
        assert model.pupil.shape == (ny, nx)
        assert model.pupil.dtype == torch.complex128


# ---------------------------------------------------------------------------
# BPM operator tests
# ---------------------------------------------------------------------------


class TestBPMOperators:
    def test_propagate_preserves_shape(self):
        model, _ = _build_model()
        u = torch.ones(16, 16, dtype=torch.complex128)
        u_out = model._bpm_propagate(u, dz=1.0)
        assert u_out.shape == (16, 16)

    def test_propagate_zero_distance_identity(self):
        """Propagation by dz=0 should return (approximately) the input field."""
        model, _ = _build_model()
        rng = np.random.default_rng(7)
        u = torch.tensor(rng.standard_normal((16, 16)) +
                         1j * rng.standard_normal((16, 16)),
                         dtype=torch.complex128)
        u_out = model._bpm_propagate(u, dz=0.0)
        # eva_mask is ~1 in the center, so approximate identity
        np.testing.assert_allclose(u_out.numpy(), u.numpy(), atol=1e-6)

    def test_scatter_unit_phase(self):
        """Scattering with dn=0 should not change the field."""
        model, _ = _build_model()
        rng = np.random.default_rng(3)
        u = torch.tensor(rng.standard_normal((16, 16)) +
                         1j * rng.standard_normal((16, 16)),
                         dtype=torch.complex128)
        dn = torch.zeros(16, 16, dtype=torch.float64)
        u_out = model._bpm_scatter(u, dn, dz=1.0)
        np.testing.assert_allclose(u_out.numpy(), u.numpy(), atol=1e-14)

    def test_scatter_preserves_amplitude(self):
        """Scattering is a pure phase modulation, so |u_out| == |u_in|."""
        model, _ = _build_model()
        rng = np.random.default_rng(42)
        u = torch.tensor(rng.standard_normal((16, 16)) +
                         1j * rng.standard_normal((16, 16)),
                         dtype=torch.complex128)
        dn = torch.tensor(rng.uniform(-0.05, 0.05, (16, 16)),
                          dtype=torch.float64)
        u_out = model._bpm_scatter(u, dn, dz=1.0)
        np.testing.assert_allclose(torch.abs(u_out).numpy(),
                                   torch.abs(u).numpy(), atol=1e-14)

    def test_reflect_flips_sign(self):
        model, _ = _build_model()
        u = torch.tensor([[1.0 + 2j, -3.0 + 4j]], dtype=torch.complex128)
        u_ref = model._reflect(u)
        np.testing.assert_allclose(u_ref.numpy(), -u.numpy(), atol=1e-15)


# ---------------------------------------------------------------------------
# Forward model integration tests
# ---------------------------------------------------------------------------


class TestForwardModel:
    def test_forward_single_ring_output_shape(self):
        model, config = _build_model()
        _, ny, nx = config.volume_shape
        dn = torch.zeros(*config.volume_shape, dtype=torch.float64)
        I = model.forward_single_ring(dn, na=0.3, angle_idx=0,
                                      n_angles_in_ring=2)
        assert I.shape == (ny, nx)

    def test_forward_output_shape(self):
        model, config = _build_model()
        dn = torch.zeros(*config.volume_shape, dtype=torch.float64)
        I = model.forward(dn)
        assert I.shape == (config.n_angles, config.volume_shape[1],
                           config.volume_shape[2])

    def test_forward_nonnegative(self):
        model, config = _build_model()
        rng = np.random.default_rng(10)
        dn = torch.tensor(rng.uniform(-0.01, 0.01, config.volume_shape),
                          dtype=torch.float64)
        with torch.no_grad():
            I = model.forward(dn)
        assert torch.all(I >= 0)

    def test_simulate_measurements_alias(self):
        """simulate_measurements should return identical result to forward."""
        model, config = _build_model()
        dn = torch.zeros(*config.volume_shape, dtype=torch.float64)
        with torch.no_grad():
            a = model.forward(dn)
            b = model.simulate_measurements(dn)
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-15)

    def test_zero_sample_uniform_bf_intensity(self):
        """For dn=0, all BF angles should give the same intensity pattern."""
        meta = _small_metadata()
        meta["illumination_rings"] = [
            {"NA": 0.3, "n_angles": 3, "type": "BF"},
        ]
        model, config = _build_model(meta)
        dn = torch.zeros(*config.volume_shape, dtype=torch.float64)
        with torch.no_grad():
            I = model.forward(dn)
        # All 3 BF angles on a blank sample should produce equal total power
        totals = I.sum(dim=(1, 2)).numpy()
        np.testing.assert_allclose(totals, totals[0], rtol=1e-6)

    def test_repr_contains_key_info(self):
        model, _ = _build_model()
        r = repr(model)
        assert "ReflectionBPMForwardModel" in r
        assert "NA_obj" in r
        assert "BF" in r
