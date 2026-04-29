"""Tests for physics_model module."""

import os
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics_model import compute_com, ptychographic_forward


class TestComputeCom:
    def test_uniform_dp(self):
        """Uniform DP should have CoM at center."""
        data = np.ones((2, 2, 10, 10), dtype=np.float32)
        com_x, com_y = compute_com(data)
        np.testing.assert_allclose(com_x, 4.5, atol=1e-10)
        np.testing.assert_allclose(com_y, 4.5, atol=1e-10)

    def test_delta_dp(self):
        """Delta function DP should have CoM at the delta position."""
        data = np.zeros((1, 1, 10, 10), dtype=np.float32)
        data[0, 0, 3, 7] = 1.0
        com_x, com_y = compute_com(data)
        np.testing.assert_allclose(com_x[0, 0], 3.0, atol=1e-10)
        np.testing.assert_allclose(com_y[0, 0], 7.0, atol=1e-10)

    def test_with_mask(self):
        """Masked CoM should only consider pixels within mask."""
        data = np.zeros((1, 1, 10, 10), dtype=np.float32)
        data[0, 0, 2, 2] = 1.0  # inside mask
        data[0, 0, 8, 8] = 100.0  # outside mask
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:5, 0:5] = True
        com_x, com_y = compute_com(data, mask=mask)
        np.testing.assert_allclose(com_x[0, 0], 2.0, atol=1e-10)
        np.testing.assert_allclose(com_y[0, 0], 2.0, atol=1e-10)

    def test_output_shape(self):
        data = np.ones((3, 5, 8, 8), dtype=np.float32)
        com_x, com_y = compute_com(data)
        assert com_x.shape == (3, 5)
        assert com_y.shape == (3, 5)


class TestPtychographicForward:
    def test_output_shape(self):
        obj = np.ones((32, 32), dtype=complex)
        probe = np.ones((8, 8), dtype=complex)
        positions = np.array([[0, 0], [4, 4], [8, 8]])
        intensities = ptychographic_forward(obj, probe, positions)
        assert intensities.shape == (3, 8, 8)

    def test_nonnegative_intensities(self):
        rng = np.random.default_rng(42)
        obj = rng.random((32, 32)) * np.exp(1j * rng.random((32, 32)))
        probe = rng.random((8, 8)) * np.exp(1j * rng.random((8, 8)))
        positions = np.array([[0, 0], [10, 10]])
        intensities = ptychographic_forward(obj, probe, positions)
        assert np.all(intensities >= 0)

    def test_parseval_theorem(self):
        """Total intensity should equal N * sum(|exit_wave|^2) for unnormalized FFT."""
        rng = np.random.default_rng(42)
        obj = rng.random((32, 32)) * np.exp(1j * rng.random((32, 32)))
        probe = rng.random((8, 8)) * np.exp(1j * rng.random((8, 8)))
        positions = np.array([[5, 5]])
        intensities = ptychographic_forward(obj, probe, positions)

        exit_wave = probe * obj[5:13, 5:13]
        N = exit_wave.size  # unnormalized FFT: sum(|F{x}|^2) = N * sum(|x|^2)
        expected_total = N * np.sum(np.abs(exit_wave) ** 2)
        actual_total = np.sum(intensities[0])
        np.testing.assert_allclose(actual_total, expected_total, rtol=1e-10)
