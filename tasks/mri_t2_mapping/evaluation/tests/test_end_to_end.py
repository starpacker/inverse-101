"""End-to-end integration test for MRI T2 mapping pipeline."""

import os
import json
import numpy as np
import pytest

from src.preprocessing import (
    load_multi_echo_data,
    load_ground_truth,
    load_metadata,
    preprocess_signal,
)
from src.solvers import fit_t2_loglinear, fit_t2_nonlinear
from src.visualization import compute_ncc, compute_nrmse


class TestEndToEnd:
    """Full pipeline integration test."""

    def test_pipeline_produces_valid_t2_map(self, task_dir):
        """Full pipeline should produce T2 map meeting quality thresholds."""
        # Load data
        signal = load_multi_echo_data(task_dir)
        T2_gt, M0_gt, tissue_mask = load_ground_truth(task_dir)
        meta = load_metadata(task_dir)
        TE = meta['echo_times_ms']

        # Preprocess
        signal_2d = preprocess_signal(signal)
        mask_2d = tissue_mask[0]
        T2_gt_2d = T2_gt[0]

        # Fit
        T2_ll, M0_ll = fit_t2_loglinear(signal_2d, TE, mask=mask_2d)
        T2_nls, M0_nls = fit_t2_nonlinear(
            signal_2d, TE, mask=mask_2d,
            T2_init=T2_ll, M0_init=M0_ll,
        )

        # Evaluate
        ncc = compute_ncc(T2_nls, T2_gt_2d, mask=mask_2d)
        nrmse = compute_nrmse(T2_nls, T2_gt_2d, mask=mask_2d)

        # Check against metrics boundaries
        metrics_path = os.path.join(task_dir, 'evaluation', 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)

        assert ncc >= metrics['ncc_boundary'], \
            f"NCC {ncc:.4f} < boundary {metrics['ncc_boundary']}"
        assert nrmse <= metrics['nrmse_boundary'], \
            f"NRMSE {nrmse:.4f} > boundary {metrics['nrmse_boundary']}"

    def test_reference_outputs_exist(self, task_dir):
        """Reference output files should exist."""
        ref_dir = os.path.join(task_dir, 'evaluation', 'reference_outputs')
        assert os.path.exists(os.path.join(ref_dir, 'T2_map_nonlinear.npz'))
        assert os.path.exists(os.path.join(ref_dir, 'T2_map_loglinear.npz'))

    def test_reference_output_shapes(self, task_dir):
        """Reference outputs should have correct shapes."""
        ref_dir = os.path.join(task_dir, 'evaluation', 'reference_outputs')
        nls = np.load(os.path.join(ref_dir, 'T2_map_nonlinear.npz'))
        assert nls['T2_map'].shape == (1, 256, 256)
        assert nls['M0_map'].shape == (1, 256, 256)

    def test_metrics_json_schema(self, task_dir):
        """metrics.json should follow the required schema."""
        metrics_path = os.path.join(task_dir, 'evaluation', 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)

        assert 'baseline' in metrics
        assert 'ncc_boundary' in metrics
        assert 'nrmse_boundary' in metrics
        assert isinstance(metrics['baseline'], list)
        assert len(metrics['baseline']) >= 1
        for entry in metrics['baseline']:
            assert 'method' in entry
            assert 'ncc_vs_ref' in entry
            assert 'nrmse_vs_ref' in entry
