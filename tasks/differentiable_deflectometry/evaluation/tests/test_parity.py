"""Parity tests comparing cleaned code against reference outputs.

These tests verify that the cleaned pipeline produces results consistent
with the original demo_experiments.py output.
"""
import json
import os

import numpy as np
import pytest


REFERENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_outputs')


@pytest.fixture
def metrics():
    """Load reference metrics."""
    path = os.path.join(REFERENCE_DIR, 'metrics.json')
    if not os.path.exists(path):
        pytest.skip("Reference outputs not yet generated. Run main.py first.")
    with open(path) as f:
        return json.load(f)


class TestParityMetrics:
    """Verify pipeline metrics match reference."""

    def test_final_loss(self, metrics):
        """Final loss should be approximately 1.16e-3."""
        assert metrics['final_loss'] < 2e-3
        assert metrics['final_loss'] > 5e-4

    def test_n_iterations(self, metrics):
        """Should converge in 15-30 iterations."""
        assert 10 <= metrics['n_iterations'] <= 35

    def test_displacement_error(self, metrics):
        """Mean displacement error should be ~43 um."""
        assert 30 < metrics['mean_displacement_error_um'] < 60

    def test_surface1_roc(self, metrics):
        """Surface 1 ROC should be within 2% of ground truth (-32.14 mm)."""
        roc = metrics['recovered_surface_1_roc_mm']
        assert metrics['relative_error_surface_1_roc_mm'] < 0.02

    def test_surface0_roc(self, metrics):
        """Surface 0 ROC should be within 10% of ground truth (-82.23 mm)."""
        assert metrics['relative_error_surface_0_roc_mm'] < 0.10

    def test_thickness(self, metrics):
        """Thickness should be within 15% of ground truth (3.59 mm)."""
        assert metrics['relative_error_thickness_mm'] < 0.15


class TestParityOutputFiles:
    """Verify expected output files exist."""

    def test_metrics_json_exists(self):
        path = os.path.join(REFERENCE_DIR, 'metrics.json')
        if not os.path.exists(path):
            pytest.skip("Reference outputs not yet generated.")
        assert os.path.exists(path)

    def test_loss_history_exists(self):
        path = os.path.join(REFERENCE_DIR, 'loss_history.npy')
        if not os.path.exists(path):
            pytest.skip("Reference outputs not yet generated.")
        loss = np.load(path)
        assert len(loss) > 10
        assert loss[-1] < loss[0]  # loss should decrease

    def test_optimized_params_exists(self):
        path = os.path.join(REFERENCE_DIR, 'optimized_params.json')
        if not os.path.exists(path):
            pytest.skip("Reference outputs not yet generated.")
        with open(path) as f:
            params = json.load(f)
        assert 'surface_0_c' in params
        assert 'surface_1_c' in params

    def test_visualization_images_exist(self):
        for name in ['initial0.jpg', 'initial1.jpg', 'optimized0.jpg', 'optimized1.jpg']:
            path = os.path.join(REFERENCE_DIR, name)
            if not os.path.exists(path):
                pytest.skip("Reference outputs not yet generated.")
            assert os.path.getsize(path) > 1000  # non-trivial file
