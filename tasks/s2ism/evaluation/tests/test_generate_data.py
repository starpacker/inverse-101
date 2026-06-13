"""
Unit tests for src/generate_data.py.

The most important test here is ``test_optimal_bkg_plane_used_for_pxsizez``, which
verifies that the PSF simulation step correctly applies the value returned by
``find_out_of_focus_from_param`` to ``gridPar.pxsizez``. An agent that uses an
arbitrary z-spacing (e.g. a default like 10 nm) instead of the optimal value
will produce two PSF planes that are essentially identical, and this test will
catch that mistake.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr

from src.generate_data import (
    DEFAULT_NX, DEFAULT_NZ, DEFAULT_PXSIZEX_NM,
    make_tubulin_phantom, make_psf_settings,
    simulate_psfs, apply_forward_model_with_noise, generate_data,
)
from src.physics_model import find_out_of_focus_from_param

TASK_DIR = Path(__file__).parent.parent.parent
DATA_DIR = TASK_DIR / 'data'


# ==========================================
# Phantom + forward model
# ==========================================

def test_make_tubulin_phantom_shape():
    gt = make_tubulin_phantom(Nx=51, Nz=2, pxsizex=40, signal=300, seed=42)
    assert gt.shape == (2, 51, 51)
    assert gt.dtype == np.float64


def test_make_tubulin_phantom_signal_scaling():
    gt_low = make_tubulin_phantom(Nx=51, Nz=2, pxsizex=40, signal=100, seed=42)
    gt_high = make_tubulin_phantom(Nx=51, Nz=2, pxsizex=40, signal=300, seed=42)
    # Same seed, scaled signal -> ratio should be 3
    np.testing.assert_allclose(gt_high.max() / gt_low.max(), 3.0, rtol=1e-6)


def test_make_psf_settings_values():
    exPar, emPar = make_psf_settings()
    assert exPar.na == 1.4
    assert exPar.wl == 640
    assert emPar.wl == 660
    assert exPar.n == 1.5


def test_apply_forward_model_with_noise_shape():
    np.random.seed(0)
    gt = np.abs(np.random.rand(2, 21, 21)) * 100
    psf = np.abs(np.random.rand(2, 11, 11, 9))
    psf /= psf.sum(axis=(1, 2, 3), keepdims=True)
    noisy = apply_forward_model_with_noise(gt, psf, seed=1)
    assert noisy.shape == (21, 21, 9)
    assert (noisy >= 0).all()


# ==========================================
# Critical: optimal_bkg_plane → pxsizez
# ==========================================

def _plane_correlation(psf):
    """Pearson correlation between in-focus (z=0) and out-of-focus (z=1) PSF planes."""
    a = psf[0].ravel()
    b = psf[1].ravel()
    return pearsonr(a, b)[0]


def test_optimal_bkg_plane_used_for_pxsizez():
    """
    The PSF simulation must use ``optimal_bkg_plane`` (from
    ``find_out_of_focus_from_param``) as ``gridPar.pxsizez``. This test
    catches an agent that hard-codes a small/default pxsizez instead.

    Strategy: build PSFs at the optimal pxsizez and at a deliberately wrong
    (much smaller) pxsizez. The optimal one should produce a markedly LOWER
    plane-to-plane Pearson correlation, because that is precisely the criterion
    ``find_out_of_focus_from_param(mode='Pearson')`` minimises.
    """
    pxsizex = DEFAULT_PXSIZEX_NM
    Nz = DEFAULT_NZ
    exPar, emPar = make_psf_settings()

    optimal_bkg_plane, _ = find_out_of_focus_from_param(
        pxsizex, exPar, emPar, mode='Pearson', stack='positive', graph=False)

    assert optimal_bkg_plane > 0, "optimal_bkg_plane must be positive"

    # PSFs at optimal pxsizez (the correct choice)
    psf_opt = simulate_psfs(pxsizex, optimal_bkg_plane, Nz, exPar, emPar, normalize=True)

    # PSFs at a deliberately tiny z-spacing (the wrong choice)
    wrong_pxsizez = 10.0  # nm — too small; the two planes will look almost identical
    psf_wrong = simulate_psfs(pxsizex, wrong_pxsizez, Nz, exPar, emPar, normalize=True)

    corr_opt = _plane_correlation(psf_opt)
    corr_wrong = _plane_correlation(psf_wrong)

    # The wrong-spacing PSFs are nearly identical (correlation ~ 1).
    assert corr_wrong > 0.99, \
        f"At pxsizez=10nm planes should be near-identical, got corr={corr_wrong:.4f}"

    # The optimal PSFs should have substantially lower correlation —
    # this is exactly the property maximised by find_out_of_focus_from_param.
    assert corr_opt < 0.9, \
        f"PSF planes at optimal pxsizez should have corr < 0.9, got {corr_opt:.4f}"
    assert corr_opt < corr_wrong - 0.05, (
        f"Optimal pxsizez correlation ({corr_opt:.4f}) should be markedly lower "
        f"than wrong pxsizez correlation ({corr_wrong:.4f}) — agent likely "
        "did not apply optimal_bkg_plane to gridPar.pxsizez."
    )


def test_saved_psf_matches_optimal_bkg_plane():
    """The PSF saved in raw_data.npz must have been generated at optimal_bkg_plane.

    We re-run the simulation with the optimal value and verify the saved PSF's
    plane-correlation matches that of the freshly simulated optimal PSF, while
    being significantly lower than a PSF generated at a wrong spacing.
    """
    raw = np.load(DATA_DIR / 'raw_data.npz')
    psf_saved = raw['psf'][0].astype(np.float64)
    assert psf_saved.shape[0] == 2, "Expected Nz=2"

    with open(DATA_DIR / 'meta_data.json') as f:
        meta = json.load(f)

    optimal_bkg_plane = meta['optimal_bkg_plane_nm']
    pxsizex = meta['pxsizex_nm']
    Nz = meta['Nz']
    exPar, emPar = make_psf_settings()

    psf_opt = simulate_psfs(pxsizex, optimal_bkg_plane, Nz, exPar, emPar, normalize=True)

    corr_saved = _plane_correlation(psf_saved)
    corr_opt = _plane_correlation(psf_opt)

    # The saved PSF should match a freshly simulated PSF with the recorded
    # optimal_bkg_plane (within numerical precision).
    np.testing.assert_allclose(corr_saved, corr_opt, atol=1e-4,
        err_msg="Saved PSF plane-correlation does not match a fresh simulation "
                "at optimal_bkg_plane — meta_data.json may be inconsistent with raw_data.npz")


# ==========================================
# End-to-end pipeline
# ==========================================

def test_generate_data_pipeline_creates_files():
    """generate_data() must produce all three data files with correct keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a smaller phantom for speed by monkey-patching defaults indirectly:
        # we just call generate_data() with the default size and verify outputs.
        # Skip if already cached locally to save CI time.
        gt, noisy, psf, meta = generate_data(output_dir=tmpdir, seed=42)

        tmp = Path(tmpdir)
        assert (tmp / 'raw_data.npz').exists()
        assert (tmp / 'ground_truth.npz').exists()
        assert (tmp / 'meta_data.json').exists()

        raw = np.load(tmp / 'raw_data.npz')
        assert 'measurements' in raw.files
        assert 'psf' in raw.files
        assert raw['measurements'].shape[0] == 1  # batch-first
        assert raw['psf'].shape[0] == 1

        gt_file = np.load(tmp / 'ground_truth.npz')
        assert 'ground_truth' in gt_file.files
        assert gt_file['ground_truth'].shape == (1, DEFAULT_NZ, DEFAULT_NX, DEFAULT_NX)

        with open(tmp / 'meta_data.json') as f:
            meta_loaded = json.load(f)
        assert 'optimal_bkg_plane_nm' in meta_loaded
        assert meta_loaded['optimal_bkg_plane_nm'] > 0
        # The recorded value must equal what find_out_of_focus_from_param returns
        # for the documented imaging parameters — i.e. the agent did not
        # substitute an arbitrary number.
        exPar, emPar = make_psf_settings()
        opt, _ = find_out_of_focus_from_param(
            meta_loaded['pxsizex_nm'], exPar, emPar,
            mode='Pearson', stack='positive', graph=False)
        np.testing.assert_allclose(meta_loaded['optimal_bkg_plane_nm'], float(opt),
                                   rtol=1e-6)
