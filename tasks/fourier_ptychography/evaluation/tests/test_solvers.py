"""Unit tests for src/solvers.py."""
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.solvers import run_qnewton, compute_reconstruction_error
from src.preprocessing import setup_reconstruction, setup_params, setup_monitor
from src.physics_model import compute_pupil_mask

FIXTURES = Path(__file__).parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# compute_reconstruction_error
# ---------------------------------------------------------------------------

def test_compute_reconstruction_error_deterministic():
    """compute_reconstruction_error must return the saved fixture value."""
    fix = np.load(FIXTURES / "reconstruction_error.npz")
    result = compute_reconstruction_error(fix["I_meas"], fix["I_est"])
    np.testing.assert_allclose(result, float(fix["output"]), rtol=1e-10)


def test_compute_reconstruction_error_zero():
    """Identical measurements must give zero error."""
    rng = np.random.default_rng(0)
    I = rng.random((4, 16, 16)).astype(np.float32)
    assert compute_reconstruction_error(I, I) == pytest.approx(0.0, abs=1e-12)


def test_compute_reconstruction_error_nonnegative():
    """Error must always be non-negative."""
    rng = np.random.default_rng(1)
    I_meas = rng.random((3, 8, 8)).astype(np.float32)
    I_est = rng.random((3, 8, 8)).astype(np.float32)
    assert compute_reconstruction_error(I_meas, I_est) >= 0.0


def test_compute_reconstruction_error_scalar():
    """Return value must be a Python float (scalar)."""
    I = np.ones((2, 4, 4), dtype=np.float32)
    result = compute_reconstruction_error(I, I * 0.5)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# run_qnewton
# ---------------------------------------------------------------------------

def test_run_qnewton_returns_state(fpm_data):
    """run_qnewton must return a PtyState with the same object/probe shapes."""
    from src.preprocessing import PtyState
    state = setup_reconstruction(fpm_data, seed=42)
    obj_shape = state.object.shape
    probe_shape = state.probe.shape
    params = setup_params()
    monitor = setup_monitor()
    state = run_qnewton(state, fpm_data, params, monitor, num_iterations=2)
    assert isinstance(state, PtyState)
    assert state.object.shape == obj_shape
    assert state.probe.shape == probe_shape


def test_run_qnewton_error_list_length(fpm_data):
    """Error list must have exactly num_iterations entries."""
    state = setup_reconstruction(fpm_data, seed=42)
    params = setup_params()
    state = run_qnewton(state, fpm_data, params, num_iterations=3)
    assert len(state.error) == 3


def test_run_qnewton_error_finite(fpm_data):
    """All per-iteration errors must be finite and non-negative."""
    state = setup_reconstruction(fpm_data, seed=42)
    params = setup_params()
    state = run_qnewton(state, fpm_data, params, num_iterations=3)
    for e in state.error:
        assert np.isfinite(e)
        assert e >= 0.0


def test_run_qnewton_error_decreasing(fpm_data):
    """Error must decrease (or stay stable) over the first few iterations."""
    state = setup_reconstruction(fpm_data, seed=42)
    params = setup_params()
    state = run_qnewton(state, fpm_data, params, num_iterations=5)
    # Last error should be less than or equal to the first
    assert state.error[-1] <= state.error[0] * 1.1  # allow 10% slack


def test_run_qnewton_complex_arrays(fpm_data):
    """Object and probe must remain complex after reconstruction."""
    state = setup_reconstruction(fpm_data, seed=42)
    params = setup_params()
    state = run_qnewton(state, fpm_data, params, num_iterations=2)
    assert np.iscomplexobj(state.object)
    assert np.iscomplexobj(state.probe)
