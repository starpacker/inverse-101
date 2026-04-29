"""End-to-end integration tests for pet_mlem."""
import json, os, sys
import numpy as np
import pytest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

from src.visualization import compute_ncc, compute_nrmse
from src.preprocessing import load_ground_truth


@pytest.fixture
def task_dir():
    return TASK_DIR


class TestEndToEnd:
    def test_main_runs(self, task_dir):
        from main import main
        metrics = main()
        assert metrics is not None

    def test_data_files_exist(self):
        data_dir = os.path.join(TASK_DIR, "data")
        assert os.path.exists(os.path.join(data_dir, "raw_data.npz"))
        assert os.path.exists(os.path.join(data_dir, "ground_truth.npz"))
        assert os.path.exists(os.path.join(data_dir, "meta_data.json"))

    def test_metrics_schema(self):
        path = os.path.join(TASK_DIR, 'evaluation', 'metrics.json')
        assert os.path.exists(path)
        with open(path) as f:
            m = json.load(f)
        assert 'baseline' in m
        assert 'ncc_boundary' in m
        assert 'nrmse_boundary' in m
        assert len(m['baseline']) >= 2

    def test_reference_outputs_exist(self):
        ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
        for name in ['recon_mlem.npz', 'recon_osem.npz']:
            path = os.path.join(ref_dir, name)
            assert os.path.exists(path), f"Missing {name}"
            data = np.load(path)
            assert 'reconstruction' in data
            assert data['reconstruction'].shape[0] == 1

    def test_reconstructions_non_negative(self):
        ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
        for name in ['recon_mlem.npz', 'recon_osem.npz']:
            data = np.load(os.path.join(ref_dir, name))
            assert np.all(data['reconstruction'] >= 0)

    def test_osem_quality(self):
        """OSEM should achieve NCC > 0.95."""
        ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
        recon = np.load(os.path.join(ref_dir, 'recon_osem.npz'))['reconstruction'][0]
        gt = load_ground_truth(TASK_DIR)[0]
        mask = gt > 0
        ncc = compute_ncc(recon, gt, mask=mask)
        assert ncc > 0.95, f"OSEM NCC too low: {ncc}"

    def test_osem_beats_mlem(self):
        """OSEM should have equal or better NCC than MLEM (with fewer equiv iters)."""
        ref_dir = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
        gt = load_ground_truth(TASK_DIR)[0]
        mask = gt > 0
        mlem = np.load(os.path.join(ref_dir, 'recon_mlem.npz'))['reconstruction'][0]
        osem = np.load(os.path.join(ref_dir, 'recon_osem.npz'))['reconstruction'][0]
        ncc_mlem = compute_ncc(mlem, gt, mask=mask)
        ncc_osem = compute_ncc(osem, gt, mask=mask)
        assert ncc_osem >= ncc_mlem * 0.99  # OSEM at least comparable
