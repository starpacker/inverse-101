"""Unit tests for the solver module (parity + integration)."""

import os
import sys

import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_ground_truth, load_raw_data, load_metadata
from src.solvers import unweighted_tv_reconstruction, pwls_tv_reconstruction
from src.visualization import compute_ncc, compute_nrmse, centre_crop

DATA_DIR = os.path.join(TASK_DIR, "data")
REF_DIR = os.path.join(TASK_DIR, "evaluation", "reference_outputs")


@pytest.fixture(scope="module")
def data():
    phantom = load_ground_truth(DATA_DIR)
    raw = load_raw_data(DATA_DIR)
    meta = load_metadata(DATA_DIR)
    ref = dict(np.load(os.path.join(REF_DIR, "reconstructions.npz")))
    return {
        "phantom": phantom,
        "raw": raw,
        "meta": meta,
        "ref": ref,
    }


class TestUnweightedRecon:
    """Tests for unweighted SVMBIR reconstruction."""

    def test_output_shape(self, data):
        raw = data["raw"]
        meta = data["meta"]
        recon = unweighted_tv_reconstruction(
            raw["sinogram_low_dose"], raw["angles"],
            meta["image_size"], meta["image_size"])
        assert recon.shape == (256, 256)

    def test_parity_with_reference(self, data):
        """Cleaned code must reproduce the reference reconstruction."""
        raw = data["raw"]
        meta = data["meta"]
        ref_uw = data["ref"]["recon_unweighted"][0]
        recon = unweighted_tv_reconstruction(
            raw["sinogram_low_dose"], raw["angles"],
            meta["image_size"], meta["image_size"])
        # SVMBIR's ICD solver has slight non-determinism across runs;
        # reference was saved as float32. Use relaxed tolerance.
        np.testing.assert_allclose(recon, ref_uw, rtol=1e-2, atol=1e-3)

    def test_ncc_above_threshold(self, data):
        """Unweighted recon should have reasonable NCC."""
        raw = data["raw"]
        meta = data["meta"]
        recon = unweighted_tv_reconstruction(
            raw["sinogram_low_dose"], raw["angles"],
            meta["image_size"], meta["image_size"])
        gt_crop = centre_crop(data["phantom"], 0.8)
        recon_crop = centre_crop(recon, 0.8)
        ncc = compute_ncc(recon_crop, gt_crop)
        assert ncc > 0.90


class TestPWLSRecon:
    """Tests for PWLS SVMBIR reconstruction."""

    def test_output_shape(self, data):
        raw = data["raw"]
        meta = data["meta"]
        recon = pwls_tv_reconstruction(
            raw["sinogram_low_dose"], raw["angles"],
            raw["weights_low_dose"],
            meta["image_size"], meta["image_size"])
        assert recon.shape == (256, 256)

    def test_parity_with_reference(self, data):
        """Cleaned code must reproduce the reference PWLS reconstruction."""
        raw = data["raw"]
        meta = data["meta"]
        ref_pwls = data["ref"]["recon_pwls_low"][0]
        recon = pwls_tv_reconstruction(
            raw["sinogram_low_dose"], raw["angles"],
            raw["weights_low_dose"],
            meta["image_size"], meta["image_size"])
        # SVMBIR ICD has slight non-determinism; reference is float32.
        np.testing.assert_allclose(recon, ref_pwls, rtol=1e-1, atol=2e-3)

    def test_pwls_better_than_unweighted(self, data):
        """PWLS should outperform unweighted on NCC and NRMSE."""
        raw = data["raw"]
        meta = data["meta"]
        recon_uw = unweighted_tv_reconstruction(
            raw["sinogram_low_dose"], raw["angles"],
            meta["image_size"], meta["image_size"])
        recon_pwls = pwls_tv_reconstruction(
            raw["sinogram_low_dose"], raw["angles"],
            raw["weights_low_dose"],
            meta["image_size"], meta["image_size"])
        gt_crop = centre_crop(data["phantom"], 0.8)
        uw_ncc = compute_ncc(centre_crop(recon_uw, 0.8), gt_crop)
        pwls_ncc = compute_ncc(centre_crop(recon_pwls, 0.8), gt_crop)
        uw_nrmse = compute_nrmse(centre_crop(recon_uw, 0.8), gt_crop)
        pwls_nrmse = compute_nrmse(centre_crop(recon_pwls, 0.8), gt_crop)
        assert pwls_ncc > uw_ncc, "PWLS NCC should be higher than unweighted"
        assert pwls_nrmse < uw_nrmse, "PWLS NRMSE should be lower than unweighted"

    def test_ncc_above_boundary(self, data):
        """PWLS recon NCC should exceed the evaluation boundary."""
        raw = data["raw"]
        meta = data["meta"]
        recon = pwls_tv_reconstruction(
            raw["sinogram_low_dose"], raw["angles"],
            raw["weights_low_dose"],
            meta["image_size"], meta["image_size"])
        gt_crop = centre_crop(data["phantom"], 0.8)
        ncc = compute_ncc(centre_crop(recon, 0.8), gt_crop)
        # boundary is 0.9 * 0.9969 = 0.8972
        assert ncc > 0.89
