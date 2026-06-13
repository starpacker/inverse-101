"""Tests for solvers module."""

import os
import sys
import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.solvers import run_mpie, compute_reconstruction_error, _apply_constraints_cp, _com_stabilization
from src.preprocessing import PtyData, PtyState
from src.utils import fft2c


# ---------------------------------------------------------------------------
# Helpers to build small synthetic reconstruction problems
# ---------------------------------------------------------------------------

def _make_pty_data(Np=16, J=4, rng=None):
    """Create a minimal PtyData with synthetic diffraction patterns."""
    if rng is None:
        rng = np.random.default_rng(0)
    ptychogram = rng.random((J, Np, Np)).astype(np.float32) + 0.1
    encoder = np.zeros((J, 2), dtype=np.float64)
    energy_at_pos = np.sum(ptychogram, axis=(-1, -2))
    return PtyData(
        ptychogram=ptychogram,
        encoder=encoder,
        wavelength=632.8e-9,
        zo=0.05,
        dxd=55e-6,
        Nd=Np,
        No=32,
        entrancePupilDiameter=1e-3,
        energy_at_pos=energy_at_pos,
        max_probe_power=float(np.sqrt(np.max(energy_at_pos))),
    )


def _make_pty_state(Np=16, No=32, J=4, rng=None):
    """Create a minimal PtyState with synthetic object, probe, and positions."""
    if rng is None:
        rng = np.random.default_rng(1)
    wavelength = 632.8e-9
    zo = 0.05
    dxo = wavelength * zo / (Np * 55e-6)

    obj = np.ones((No, No), dtype=np.complex128) + 0.01 * rng.standard_normal((No, No))
    probe = np.ones((Np, Np), dtype=np.complex128) + 0.01 * rng.standard_normal((Np, Np))

    # Positions must keep the patch inside the object array
    max_pos = No - Np
    positions = rng.integers(0, max(max_pos, 1), size=(J, 2))

    xp = np.linspace(-Np / 2, Np / 2, Np) * dxo
    Xp, Yp = np.meshgrid(xp, xp)

    return PtyState(
        object=obj,
        probe=probe,
        positions=positions,
        No=No,
        Np=Np,
        wavelength=wavelength,
        zo=zo,
        dxo=dxo,
        Xp=Xp,
        Yp=Yp,
    )


def _make_params():
    """Create a minimal params namespace matching setup_params()."""
    p = SimpleNamespace()
    p.probePowerCorrectionSwitch = True
    p.comStabilizationSwitch = 10
    p.probeSmoothenessSwitch = False
    p.probeSmoothnessAleph = 1e-2
    p.probeSmoothenessWidth = 10
    p.l2reg_probe_aleph = 1e-2
    p.l2reg_object_aleph = 1e-2
    return p


# ---------------------------------------------------------------------------
# Tests for compute_reconstruction_error
# ---------------------------------------------------------------------------

class TestComputeReconstructionError:
    """Tests for the normalized amplitude error metric."""

    def test_identical_inputs_give_zero_error(self):
        """Error between identical ptychograms should be zero."""
        rng = np.random.default_rng(10)
        ptychogram = rng.random((4, 16, 16)).astype(np.float64) + 0.01
        error = compute_reconstruction_error(ptychogram, ptychogram)
        np.testing.assert_allclose(error, 0.0, atol=1e-14)

    def test_error_is_nonnegative(self):
        """Error should always be non-negative."""
        rng = np.random.default_rng(11)
        I_meas = rng.random((4, 16, 16)) + 0.01
        I_est = rng.random((4, 16, 16)) + 0.01
        error = compute_reconstruction_error(I_meas, I_est)
        assert error >= 0.0

    def test_error_is_scalar_float(self):
        """Return value should be a Python float."""
        rng = np.random.default_rng(12)
        I_meas = rng.random((2, 8, 8)) + 0.01
        I_est = rng.random((2, 8, 8)) + 0.01
        error = compute_reconstruction_error(I_meas, I_est)
        assert isinstance(error, float)

    def test_error_scales_with_mismatch(self):
        """Larger mismatch should produce larger error."""
        rng = np.random.default_rng(13)
        I_meas = rng.random((3, 16, 16)) + 1.0
        I_close = I_meas + 0.01 * rng.standard_normal(I_meas.shape)
        I_far = I_meas + 1.0 * rng.standard_normal(I_meas.shape)
        # Clip to keep non-negative
        I_close = np.clip(I_close, 0.01, None)
        I_far = np.clip(I_far, 0.01, None)
        err_close = compute_reconstruction_error(I_meas, I_close)
        err_far = compute_reconstruction_error(I_meas, I_far)
        assert err_far > err_close

    def test_known_value(self):
        """Check error formula against a hand-computed example."""
        # I_meas = [[1, 4]], I_est = [[4, 9]]
        # sqrt_meas = [[1, 2]], sqrt_est = [[2, 3]]
        # numerator = (1-2)^2 + (2-3)^2 = 2
        # denominator = 1^2 + 2^2 = 5
        # error = 2/5 = 0.4
        I_meas = np.array([[[1.0, 4.0]]])
        I_est = np.array([[[4.0, 9.0]]])
        error = compute_reconstruction_error(I_meas, I_est)
        np.testing.assert_allclose(error, 0.4, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests for _apply_constraints_cp
# ---------------------------------------------------------------------------

class TestApplyConstraints:
    """Tests for the constraint application helper."""

    def test_probe_power_correction(self):
        """After power correction the probe energy should match max_probe_power."""
        state = _make_pty_state(Np=16, No=32, J=4)
        data = _make_pty_data(Np=16, J=4)
        params = _make_params()
        params.probePowerCorrectionSwitch = True
        params.comStabilizationSwitch = 0  # disable COM
        params.probeSmoothenessSwitch = False

        _apply_constraints_cp(state, data, params, loop=0, apply_l2reg=False)

        probe_power = np.sqrt(np.sum(np.abs(state.probe) ** 2))
        np.testing.assert_allclose(probe_power, data.max_probe_power, rtol=1e-6)

    def test_l2_regularization_shrinks(self):
        """L2 regularization should reduce the magnitudes of object and probe."""
        state = _make_pty_state(Np=16, No=32, J=4)
        data = _make_pty_data(Np=16, J=4)
        params = _make_params()
        params.probePowerCorrectionSwitch = False
        params.comStabilizationSwitch = 0
        params.probeSmoothenessSwitch = False

        obj_energy_before = np.sum(np.abs(state.object) ** 2)
        probe_energy_before = np.sum(np.abs(state.probe) ** 2)

        _apply_constraints_cp(state, data, params, loop=0, apply_l2reg=True)

        obj_energy_after = np.sum(np.abs(state.object) ** 2)
        probe_energy_after = np.sum(np.abs(state.probe) ** 2)

        assert obj_energy_after < obj_energy_before, "L2 should shrink object"
        assert probe_energy_after < probe_energy_before, "L2 should shrink probe"


# ---------------------------------------------------------------------------
# Tests for _com_stabilization
# ---------------------------------------------------------------------------

class TestComStabilization:
    """Tests for probe center-of-mass stabilization."""

    def test_preserves_shapes(self):
        """COM stabilization should not change array shapes."""
        state = _make_pty_state(Np=16, No=32)
        shape_obj_before = state.object.shape
        shape_probe_before = state.probe.shape
        _com_stabilization(state)
        assert state.object.shape == shape_obj_before
        assert state.probe.shape == shape_probe_before

    def test_centered_probe_unchanged(self):
        """A perfectly centered probe should not shift."""
        Np, No = 16, 32
        dxo = 1e-6
        xp = np.linspace(-Np / 2, Np / 2, Np) * dxo
        Xp, Yp = np.meshgrid(xp, xp)

        # Symmetric Gaussian probe centered at origin
        probe = np.exp(-(Xp ** 2 + Yp ** 2) / (2 * (3 * dxo) ** 2)).astype(np.complex128)
        obj = np.ones((No, No), dtype=np.complex128)

        state = PtyState(
            object=obj.copy(),
            probe=probe.copy(),
            positions=np.array([[4, 4]]),
            No=No, Np=Np,
            wavelength=632.8e-9, zo=0.05, dxo=dxo,
            Xp=Xp, Yp=Yp,
        )
        probe_before = state.probe.copy()
        _com_stabilization(state)
        np.testing.assert_allclose(state.probe, probe_before, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests for run_mpie
# ---------------------------------------------------------------------------

class TestRunMpie:
    """Tests for the mPIE reconstruction solver."""

    def _simulate_data(self, Np=16, No=32, J=4, seed=42):
        """
        Create a consistent synthetic forward problem: generate diffraction
        data from a known object+probe, then initialise solver state.
        """
        rng = np.random.default_rng(seed)
        wavelength = 632.8e-9
        zo = 0.05
        dxd = 55e-6
        dxo = wavelength * zo / (Np * dxd)

        # Ground-truth object: random phase object
        obj_true = np.exp(1j * rng.uniform(-0.5, 0.5, (No, No))).astype(np.complex128)
        # Probe: circular with flat phase
        xp = np.linspace(-Np / 2, Np / 2, Np) * dxo
        Xp, Yp = np.meshgrid(xp, xp)
        R = np.sqrt(Xp ** 2 + Yp ** 2)
        probe_true = (R < 0.4 * Np * dxo).astype(np.complex128)
        probe_true *= np.exp(1j * 0.1 * (Xp ** 2 + Yp ** 2) / (dxo ** 2 * Np))

        # Scan positions (stay within bounds)
        max_pos = No - Np
        positions = rng.integers(0, max(max_pos, 1), size=(J, 2))

        # Forward simulate diffraction patterns
        ptychogram = np.zeros((J, Np, Np), dtype=np.float32)
        for j in range(J):
            r, c = positions[j]
            patch = obj_true[r:r + Np, c:c + Np]
            esw = probe_true * patch
            ESW = fft2c(esw)
            ptychogram[j] = (np.abs(ESW) ** 2).astype(np.float32)

        energy_at_pos = np.sum(ptychogram, axis=(-1, -2))
        data = PtyData(
            ptychogram=ptychogram,
            encoder=positions.astype(np.float64) * dxo,
            wavelength=wavelength,
            zo=zo,
            dxd=dxd,
            Nd=Np,
            No=No,
            entrancePupilDiameter=0.8 * Np * dxo,
            energy_at_pos=energy_at_pos,
            max_probe_power=float(np.sqrt(np.max(energy_at_pos))),
        )

        # Initial state: uniform object + circular probe
        obj_init = np.ones((No, No), dtype=np.complex128) + 0.001 * rng.standard_normal((No, No))
        probe_init = (R < 0.4 * Np * dxo).astype(np.complex128)

        state = PtyState(
            object=obj_init,
            probe=probe_init,
            positions=positions,
            No=No, Np=Np,
            wavelength=wavelength, zo=zo, dxo=dxo,
            Xp=Xp, Yp=Yp,
        )
        return state, data

    def test_output_type_and_error_list(self):
        """run_mpie should return a PtyState with a non-empty error list."""
        state, data = self._simulate_data(Np=16, No=32, J=4)
        params = _make_params()
        result = run_mpie(state, data, params, num_iterations=10, iterations_per_round=5, seed=0)
        assert isinstance(result, PtyState)
        assert len(result.error) == 10
        assert all(isinstance(e, float) for e in result.error)

    def test_output_shapes_preserved(self):
        """Object and probe shapes should be unchanged after reconstruction."""
        state, data = self._simulate_data(Np=16, No=32, J=4)
        params = _make_params()
        No, Np = state.No, state.Np
        result = run_mpie(state, data, params, num_iterations=10, iterations_per_round=5, seed=0)
        assert result.object.shape == (No, No)
        assert result.probe.shape == (Np, Np)

    def test_output_dtypes_complex(self):
        """Object and probe should remain complex after reconstruction."""
        state, data = self._simulate_data(Np=16, No=32, J=4)
        params = _make_params()
        result = run_mpie(state, data, params, num_iterations=10, iterations_per_round=5, seed=0)
        assert np.iscomplexobj(result.object)
        assert np.iscomplexobj(result.probe)

    def test_error_decreases(self):
        """Reconstruction error should decrease from first to last iteration."""
        state, data = self._simulate_data(Np=16, No=32, J=4)
        params = _make_params()
        params.probeSmoothenessSwitch = False
        result = run_mpie(
            state, data, params,
            num_iterations=50, iterations_per_round=25,
            beta_probe=0.25, beta_object=0.25, seed=42,
        )
        # Allow some slack: the average of the last 5 should be less than the first 5
        avg_first = np.mean(result.error[:5])
        avg_last = np.mean(result.error[-5:])
        assert avg_last < avg_first, (
            f"Error did not decrease: first-5 avg={avg_first:.4f}, last-5 avg={avg_last:.4f}"
        )

    def test_deterministic_with_same_seed(self):
        """Two runs with the same seed should produce identical results."""
        state1, data = self._simulate_data(Np=16, No=32, J=4, seed=99)
        params = _make_params()
        result1 = run_mpie(state1, data, params, num_iterations=10, iterations_per_round=5, seed=7)

        state2, _ = self._simulate_data(Np=16, No=32, J=4, seed=99)
        result2 = run_mpie(state2, data, params, num_iterations=10, iterations_per_round=5, seed=7)

        np.testing.assert_allclose(result1.object, result2.object, rtol=1e-12)
        np.testing.assert_allclose(result1.probe, result2.probe, rtol=1e-12)
        np.testing.assert_allclose(result1.error, result2.error, rtol=1e-12)
