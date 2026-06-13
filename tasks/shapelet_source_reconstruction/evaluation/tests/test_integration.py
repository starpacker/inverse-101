"""Integration tests: end-to-end pipeline validation."""

import numpy as np
import pytest
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

REF_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_outputs')


@pytest.fixture
def reference_metrics():
    metrics_path = os.path.join(REF_DIR, 'metrics.json')
    if not os.path.exists(metrics_path):
        pytest.skip("Reference outputs not generated yet. Run main.py first.")
    with open(metrics_path) as f:
        return json.load(f)


@pytest.fixture
def lensing_outputs():
    path = os.path.join(REF_DIR, 'lensing_outputs.npz')
    if not os.path.exists(path):
        pytest.skip("Reference outputs not generated yet. Run main.py first.")
    return np.load(path)


@pytest.fixture
def deconv_outputs():
    path = os.path.join(REF_DIR, 'deconv_outputs.npz')
    if not os.path.exists(path):
        pytest.skip("Reference outputs not generated yet. Run main.py first.")
    return np.load(path)


class TestEndToEnd:
    def test_metrics_exist(self, reference_metrics):
        assert 'chi2_reduced_lensing' in reference_metrics
        assert 'chi2_reduced_deconv' in reference_metrics

    def test_chi2_lensing_reasonable(self, reference_metrics):
        """Reduced chi2 should be O(1) for a good fit."""
        chi2 = reference_metrics['chi2_reduced_lensing']
        assert 0.1 < chi2 < 10.0, f"chi2_reduced_lensing={chi2} out of range"

    def test_chi2_deconv_reasonable(self, reference_metrics):
        chi2 = reference_metrics['chi2_reduced_deconv']
        assert 0.1 < chi2 < 10.0, f"chi2_reduced_deconv={chi2} out of range"

    def test_num_coefficients(self, reference_metrics):
        assert reference_metrics['num_shapelet_coeffs_decomp'] == 11476
        assert reference_metrics['num_shapelet_coeffs_recon'] == 231

    def test_lensing_image_shapes(self, lensing_outputs):
        assert lensing_outputs['image_real'].shape == (64, 64)
        assert lensing_outputs['model_lens'].shape == (64, 64)
        assert lensing_outputs['image_hr_nolens'].shape == (320, 320)
        assert lensing_outputs['source_recon_2d'].shape == (320, 320)

    def test_deconv_image_shapes(self, deconv_outputs):
        assert deconv_outputs['image_real_dc'].shape == (64, 64)
        assert deconv_outputs['model_dc'].shape == (64, 64)
        assert deconv_outputs['source_deconv_2d'].shape == (320, 320)

    def test_residuals_normalized(self, lensing_outputs):
        """Reduced residuals should have std close to 1."""
        res = lensing_outputs['residuals_lens']
        assert 0.5 < np.std(res) < 2.0

    def test_figures_exist(self):
        for i in range(1, 6):
            fig_path = os.path.join(REF_DIR, f'fig{i}_*.png')
            import glob
            matches = glob.glob(fig_path)
            assert len(matches) > 0, f"Figure {i} not found in {REF_DIR}"
