"""Tests for preprocessing module."""
import os

import numpy as np
import pytest
import torch

from src.preprocessing import Fringe, crop_images, get_crop_offset

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures')


class TestFringe:
    """Test four-step phase-shifting fringe analysis."""

    def test_solve_known_sinusoid(self):
        """Verify phase recovery from synthetic four-step phase-shifted images."""
        inputs = np.load(os.path.join(FIXTURES_DIR, 'input_fringe_solve.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_fringe_solve.npz'))
        params = np.load(os.path.join(FIXTURES_DIR, 'param_fringe_solve.npz'))

        imgs = inputs['imgs']
        N_cam = int(params['N_cam'])
        H = int(params['H'])
        W = int(params['W'])

        FR = Fringe()
        a, b, psi = FR.solve(imgs)

        # Check shapes
        assert a.shape == (2, N_cam, H, W)
        assert b.shape == (2, N_cam, H, W)
        assert psi.shape == (2, N_cam, H, W)

        # Check DC component recovery
        np.testing.assert_allclose(a[0], float(expected['true_a']), atol=1e-10)
        np.testing.assert_allclose(a[1], float(expected['true_a']), atol=1e-10)

        # Check phase recovery (up to wrapping)
        np.testing.assert_allclose(psi[0], expected['true_phase_x'], atol=1e-10)
        np.testing.assert_allclose(psi[1], expected['true_phase_y'], atol=1e-10)


class TestCropImages:
    """Test image cropping utilities."""

    def test_crop_center(self):
        """Verify center crop produces correct dimensions."""
        inputs = np.load(os.path.join(FIXTURES_DIR, 'input_crop_images.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_crop_images.npz'))

        # Use a larger array to test multi-image cropping
        imgs = np.random.rand(8, 2, 2048, 2048)
        filmsize = inputs['filmsize']
        cropped = crop_images(imgs, filmsize)
        assert cropped.shape == (8, 2, 768, 768)

    def test_crop_preserves_center(self):
        """Verify that the center pixel is preserved after cropping."""
        inputs = np.load(os.path.join(FIXTURES_DIR, 'input_crop_images.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_crop_images.npz'))

        imgs = inputs['imgs']
        filmsize = inputs['filmsize']
        cropped = crop_images(imgs, filmsize)

        center = expected['center_pixel']
        assert cropped[0, 0, center[0], center[1]] == float(expected['expected_center_value'])

    def test_get_crop_offset(self):
        config = np.load(os.path.join(FIXTURES_DIR, 'config_crop_offset.npz'))
        offset = get_crop_offset(config['filmsize'])
        expected = config['expected_offset']
        assert offset[0] == expected[0]
        assert offset[1] == expected[1]
