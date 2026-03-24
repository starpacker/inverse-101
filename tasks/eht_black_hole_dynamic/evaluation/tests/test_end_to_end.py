"""End-to-end integration tests."""

import os
import json
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.path.insert(0, TASK_DIR)


def test_reference_outputs_exist():
    """Check that all reference output files exist."""
    ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
    assert os.path.exists(os.path.join(ref_dir, 'ground_truth.npy'))
    assert os.path.exists(os.path.join(ref_dir, 'metrics.json'))
    assert os.path.exists(os.path.join(ref_dir, 'static_reconstruction.npy'))
    assert os.path.exists(os.path.join(ref_dir, 'starwarps_reconstruction.npy'))


def test_ground_truth_shape():
    """Ground truth video should have correct shape."""
    gt = np.load(os.path.join(TASK_DIR, 'evaluation', 'reference_outputs',
                              'ground_truth.npy'))
    assert gt.shape == (12, 30, 30)


def test_reference_metrics_reasonable():
    """Reference metrics should be in a reasonable range."""
    with open(os.path.join(TASK_DIR, 'evaluation', 'reference_outputs',
                           'metrics.json')) as f:
        metrics = json.load(f)

    for name in ['Static per-frame', 'StarWarps']:
        assert name in metrics
        avg = metrics[name]['average']
        assert 0.0 < avg['nrmse'] < 1.0
        assert 0.0 < avg['ncc'] <= 1.0


def test_starwarps_beats_static():
    """StarWarps should achieve higher NCC than static baseline."""
    with open(os.path.join(TASK_DIR, 'evaluation', 'reference_outputs',
                           'metrics.json')) as f:
        metrics = json.load(f)

    sw_ncc = metrics['StarWarps']['average']['ncc']
    static_ncc = metrics['Static per-frame']['average']['ncc']
    assert sw_ncc > static_ncc, \
        f"StarWarps NCC ({sw_ncc:.4f}) should exceed static ({static_ncc:.4f})"


def test_full_pipeline_data_integrity():
    """Verify data loading produces consistent shapes."""
    from src.preprocessing import prepare_data

    data_dir = os.path.join(TASK_DIR, 'data')
    obs, meta, gt = prepare_data(data_dir)

    assert obs['n_frames'] == meta['n_frames']
    assert gt.shape[0] == meta['n_frames']
    assert gt.shape[1] == meta['N']
    assert gt.shape[2] == meta['N']
