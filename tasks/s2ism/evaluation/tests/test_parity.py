"""
Parity tests: verify the cleaned src/ code produces results matching
the original s2ISM_code.py reference implementation.
"""

import numpy as np
import pytest
from pathlib import Path


TASK_DIR = Path(__file__).parent.parent.parent
DATA_DIR = TASK_DIR / 'data'
REF_DIR = TASK_DIR / 'evaluation' / 'reference_outputs'


@pytest.fixture
def loaded_data():
    raw = np.load(DATA_DIR / 'raw_data.npz')
    gt = np.load(DATA_DIR / 'ground_truth.npz')
    return {
        'measurements': raw['measurements'][0].astype(np.float64),
        'psf': raw['psf'][0].astype(np.float64),
        'ground_truth': gt['ground_truth'][0].astype(np.float64),
    }


def test_forward_model_parity(loaded_data):
    """Forward model applied to ground truth should approximate (noiseless) measurements."""
    from src.physics_model import forward_model
    gt = loaded_data['ground_truth']
    psf = loaded_data['psf']
    blurred = forward_model(gt, psf)
    noiseless = blurred.sum(axis=0)
    noisy = loaded_data['measurements']
    # Poisson noise makes exact match impossible, but correlation should be very high
    ncc = np.dot(noiseless.ravel(), noisy.ravel()) / (
        np.linalg.norm(noiseless.ravel()) * np.linalg.norm(noisy.ravel()))
    assert ncc > 0.93, f"Forward model NCC with measurements should be >0.93, got {ncc:.4f}"


@pytest.mark.skipif(not (REF_DIR / 'reconstruction.npz').exists(),
                    reason="Reference outputs not yet generated (run main.py first)")
def test_reconstruction_ncc(loaded_data):
    """Reconstruction NCC vs ground truth should meet the metrics boundary."""
    import json
    ref = np.load(REF_DIR / 'reconstruction.npz')
    recon = ref['reconstruction'][0]
    gt = loaded_data['ground_truth']

    gt0 = gt[0].ravel().astype(np.float64)
    r0 = recon[0].ravel().astype(np.float64)
    ncc = np.dot(gt0, r0) / (np.linalg.norm(gt0) * np.linalg.norm(r0))

    metrics_path = TASK_DIR / 'evaluation' / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert ncc >= metrics['ncc_boundary'], \
            f"Reconstruction NCC {ncc:.4f} below boundary {metrics['ncc_boundary']:.4f}"
    else:
        assert ncc > 0.8, f"Reconstruction NCC {ncc:.4f} too low"
