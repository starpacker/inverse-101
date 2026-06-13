"""Parity tests: verify cleaned code produces same results as reference outputs."""

import os
import numpy as np
import scipy.io as sio
import pytest

from src.preprocessing import load_ground_truth
from src.visualization import psnr, calculate_ssim

REF_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_outputs')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
FIXTURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'parity_fixtures.npz')


@pytest.fixture
def fixtures():
    return dict(np.load(FIXTURES_PATH, allow_pickle=True))


@pytest.fixture
def reference_recon():
    return sio.loadmat(os.path.join(REF_DIR, 'kaist_crop256_01_result.mat'))['img']


@pytest.fixture
def ground_truth():
    return load_ground_truth(os.path.join(DATA_DIR, 'kaist_crop256_01.mat'))


def test_reference_output_shape(reference_recon):
    """Reference reconstruction should be 256x256x31."""
    assert reference_recon.shape == (256, 256, 31)


def test_reference_psnr(ground_truth, reference_recon, fixtures):
    """PSNR should match reference within tolerance."""
    computed_psnr = psnr(ground_truth, reference_recon)
    ref_psnr = float(fixtures['ref_psnr'])
    np.testing.assert_allclose(computed_psnr, ref_psnr, rtol=1e-3)


def test_reference_ssim(ground_truth, reference_recon, fixtures):
    """SSIM should match reference within tolerance."""
    computed_ssim = calculate_ssim(ground_truth, reference_recon)
    ref_ssim = float(fixtures['ref_ssim'])
    np.testing.assert_allclose(computed_ssim, ref_ssim, rtol=1e-3)


def test_metrics_json_exists():
    """metrics.json should exist in reference outputs."""
    assert os.path.exists(os.path.join(REF_DIR, 'metrics.json'))


def test_reference_recon_range(reference_recon):
    """Reconstruction values should be in reasonable range."""
    assert reference_recon.min() >= -0.5
    assert reference_recon.max() <= 1.5
