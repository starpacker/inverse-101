"""Tests for preprocessing module."""

import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures', 'preprocessing')


def test_load_observation():
    """Test that load_observation returns correct structure."""
    import sys
    sys.path.insert(0, TASK_DIR)
    from src.preprocessing import load_observation

    data_dir = os.path.join(TASK_DIR, 'data')
    obs = load_observation(data_dir)

    assert 'n_frames' in obs
    assert 'frame_times' in obs
    assert 'vis' in obs
    assert 'sigma' in obs
    assert 'uv' in obs
    assert 'station_ids' in obs

    assert obs['n_frames'] == 12
    assert len(obs['vis']) == obs['n_frames']
    assert len(obs['sigma']) == obs['n_frames']
    assert len(obs['uv']) == obs['n_frames']

    # Check shapes
    for t in range(obs['n_frames']):
        assert obs['vis'][t].ndim == 1
        assert obs['sigma'][t].ndim == 1
        assert obs['uv'][t].ndim == 2
        assert obs['uv'][t].shape[1] == 2
        assert len(obs['vis'][t]) == len(obs['sigma'][t])
        assert len(obs['vis'][t]) == len(obs['uv'][t])


def test_load_metadata():
    """Test that load_metadata returns correct keys."""
    import sys
    sys.path.insert(0, TASK_DIR)
    from src.preprocessing import load_metadata

    data_dir = os.path.join(TASK_DIR, 'data')
    meta = load_metadata(data_dir)

    assert meta['N'] == 30
    assert meta['n_frames'] == 12
    assert 'pixel_size_rad' in meta
    assert 'total_flux' in meta
    assert meta['total_flux'] == 2.0


def test_load_ground_truth():
    """Test ground truth loading."""
    import sys
    sys.path.insert(0, TASK_DIR)
    from src.preprocessing import load_ground_truth

    gt = load_ground_truth(TASK_DIR)
    assert gt.shape == (12, 30, 30)
    assert gt.dtype == np.float64
    # Check each frame sums to total_flux (2.0)
    for t in range(12):
        np.testing.assert_allclose(gt[t].sum(), 2.0, rtol=1e-10)


def test_build_per_frame_models():
    """Test that per-frame models are built correctly."""
    import sys
    sys.path.insert(0, TASK_DIR)
    from src.preprocessing import load_observation, load_metadata, build_per_frame_models

    data_dir = os.path.join(TASK_DIR, 'data')
    obs = load_observation(data_dir)
    meta = load_metadata(data_dir)
    models = build_per_frame_models(obs, meta)

    assert len(models) == meta['n_frames']
    for m in models:
        assert m.matrix.shape == (28, 900)  # 28 baselines, 30*30 pixels
