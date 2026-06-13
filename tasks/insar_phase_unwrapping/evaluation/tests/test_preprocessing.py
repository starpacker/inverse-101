"""Unit tests for preprocessing module."""

import numpy as np
import pytest
from src.preprocessing import load_data, extract_phase_and_coherence, est_wrapped_gradient


def test_load_data(raw_data):
    assert "interferogram" in raw_data
    assert "wrapped_phase" in raw_data
    assert "snaphu_unwrapped_phase" in raw_data
    assert raw_data["interferogram"].shape == (778, 947)
    assert raw_data["interferogram"].dtype == np.complex64


def test_extract_phase_and_coherence(raw_data):
    phase, coh = extract_phase_and_coherence(raw_data["interferogram"])
    assert phase.shape == (778, 947)
    assert phase.dtype == np.float32
    assert coh.dtype == np.float32
    assert phase.min() >= -np.pi
    assert phase.max() <= np.pi
    assert coh.min() >= 0
    assert coh.max() <= 1.0


def test_est_wrapped_gradient(fixtures):
    wrapped = fixtures["input_wrapped_for_gradient"]
    phi_x, phi_y = est_wrapped_gradient(wrapped)
    np.testing.assert_allclose(phi_x, fixtures["output_phi_x"], rtol=1e-10)
    np.testing.assert_allclose(phi_y, fixtures["output_phi_y"], rtol=1e-10)


def test_wrapped_gradient_range():
    """Wrapped gradients must lie in [-pi, pi]."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(-np.pi, np.pi, (64, 64)).astype(np.float32)
    phi_x, phi_y = est_wrapped_gradient(arr)
    assert np.all(np.abs(phi_x) <= np.pi + 1e-6)
    assert np.all(np.abs(phi_y) <= np.pi + 1e-6)
