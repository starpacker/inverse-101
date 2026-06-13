"""End-to-end integration tests for diffusion_mri_dti."""

import json
import numpy as np
import pytest
import sys, os

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)


class TestEndToEnd:
    def test_main_runs(self, task_dir):
        """Full pipeline runs without error."""
        from main import main
        metrics = main()
        assert metrics is not None

    def test_metrics_schema(self, task_dir):
        """metrics.json has required fields."""
        metrics_path = os.path.join(task_dir, 'evaluation', 'metrics.json')
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert 'baseline' in metrics
        assert 'ncc_boundary' in metrics
        assert 'nrmse_boundary' in metrics
        assert len(metrics['baseline']) >= 2

    def test_reference_outputs_exist(self, task_dir):
        """Reference output files exist with correct shapes."""
        ref_dir = os.path.join(task_dir, 'evaluation', 'reference_outputs')

        for name in ['dti_ols.npz', 'dti_wls.npz']:
            path = os.path.join(ref_dir, name)
            assert os.path.exists(path), f"Missing {name}"
            data = np.load(path)
            assert 'fa_map' in data
            assert 'md_map' in data
            assert data['fa_map'].shape[0] == 1  # batch dim

    def test_metrics_boundaries(self, task_dir):
        """WLS metrics meet the boundary thresholds."""
        metrics_path = os.path.join(task_dir, 'evaluation', 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)

        wls = metrics['baseline'][1]  # WLS is second entry
        assert wls['ncc_vs_ref'] >= metrics['ncc_boundary']
        assert wls['nrmse_vs_ref'] <= metrics['nrmse_boundary']

    def test_fa_range(self, task_dir):
        """Estimated FA is in [0, 1]."""
        ref_dir = os.path.join(task_dir, 'evaluation', 'reference_outputs')
        data = np.load(os.path.join(ref_dir, 'dti_wls.npz'))
        fa = data['fa_map']
        assert np.all(fa >= 0) and np.all(fa <= 1)

    def test_md_non_negative(self, task_dir):
        """Estimated MD is non-negative."""
        ref_dir = os.path.join(task_dir, 'evaluation', 'reference_outputs')
        data = np.load(os.path.join(ref_dir, 'dti_wls.npz'))
        md = data['md_map']
        assert np.all(md >= 0)
