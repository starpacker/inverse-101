"""Unit tests for visualization.py"""

import numpy as np
import pytest
import os
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.visualization import (
    plot_shapelet_decomposition,
    plot_lensing_stages,
    plot_reconstruction,
    plot_unlensed_stages,
)


@pytest.fixture
def dummy_image_small():
    return np.random.rand(16, 16)


@pytest.fixture
def dummy_image_large():
    return np.random.rand(64, 64)


class TestPlotShapeletDecomposition:
    def test_creates_file(self, dummy_image_small, tmp_path):
        out = str(tmp_path / 'test_fig1.png')
        plot_shapelet_decomposition(
            dummy_image_small, dummy_image_small,
            dummy_image_small, dummy_image_small,
            save_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_no_save(self, dummy_image_small):
        """Should not raise when save_path is None."""
        plot_shapelet_decomposition(
            dummy_image_small, dummy_image_small,
            dummy_image_small, dummy_image_small,
            save_path=None)

    def test_different_sized_inputs(self):
        """Panels can have different sizes."""
        plot_shapelet_decomposition(
            np.random.rand(100, 100), np.random.rand(100, 100),
            np.random.rand(20, 20), np.random.rand(20, 20),
            save_path=None)


class TestPlotLensingStages:
    def test_creates_file(self, dummy_image_small, tmp_path):
        out = str(tmp_path / 'test_fig2.png')
        images = [dummy_image_small] * 5
        labels = ['a', 'b', 'c', 'd', 'e']
        plot_lensing_stages(images, labels, save_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_no_save(self, dummy_image_small):
        images = [dummy_image_small] * 5
        labels = ['a', 'b', 'c', 'd', 'e']
        plot_lensing_stages(images, labels, save_path=None)


class TestPlotReconstruction:
    def test_creates_file(self, dummy_image_small, tmp_path):
        out = str(tmp_path / 'test_fig3.png')
        panels = [(dummy_image_small, f'panel {i}') for i in range(6)]
        plot_reconstruction(panels, save_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_no_save(self, dummy_image_small):
        panels = [(dummy_image_small, f'panel {i}') for i in range(6)]
        plot_reconstruction(panels, save_path=None)

    def test_mixed_sizes(self):
        """Row 1 and row 2 can have different image sizes."""
        small = np.random.rand(16, 16)
        large = np.random.rand(64, 64)
        panels = [
            (small, 'a'), (small, 'b'), (small, 'c'),
            (large, 'd'), (large, 'e'), (large, 'f'),
        ]
        plot_reconstruction(panels, save_path=None)


class TestPlotUnlensedStages:
    def test_creates_file(self, dummy_image_small, tmp_path):
        out = str(tmp_path / 'test_fig4.png')
        images = [dummy_image_small] * 4
        plot_unlensed_stages(images, save_path=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_no_save(self, dummy_image_small):
        images = [dummy_image_small] * 4
        plot_unlensed_stages(images, save_path=None)
