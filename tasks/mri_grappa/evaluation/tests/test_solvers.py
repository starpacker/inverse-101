"""Unit tests for src/solvers.py."""
import os, sys, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing import prepare_data
from src.physics_model import fully_sampled_recon
from src.solvers import grappa_reconstruct, grappa_image_recon
from src.visualization import compute_metrics

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
REF_DIR = os.path.join(os.path.dirname(__file__), "../reference_outputs")

@pytest.fixture(scope="module")
def data():
    return prepare_data(DATA_DIR, R=2, acs_width=20)

class TestGRAPPA:
    def test_kspace_shape(self, data):
        kus, calib, kf, _, _ = data
        res = grappa_reconstruct(kus, calib, (5, 5), 0.01)
        assert res.shape == kf.shape

    def test_fills_holes(self, data):
        kus, calib, _, _, _ = data
        res = grappa_reconstruct(kus, calib, (5, 5), 0.01)
        # Locations that were zero in undersampled should now be filled
        was_zero = np.abs(kus[..., 0]) == 0
        filled = np.abs(res[..., 0]) > 0
        # Most holes should be filled (some edge holes may remain)
        assert filled[was_zero].sum() / was_zero.sum() > 0.9

    def test_image_quality(self, data):
        kus, calib, kf, _, _ = data
        recon = grappa_image_recon(kus, calib, (5, 5), 0.01)
        ref = fully_sampled_recon(kf)
        m = compute_metrics(recon, ref)
        assert m["ssim"] > 0.95
        assert m["ncc"] > 0.999

    def test_parity_with_reference(self, data):
        kus, calib, _, _, _ = data
        recon = grappa_image_recon(kus, calib, (5, 5), 0.01)
        ref = np.load(os.path.join(REF_DIR, "grappa_reconstruction.npz"))["reconstruction"][0]
        m = compute_metrics(recon, ref.astype(np.float64))
        assert m["ncc"] > 0.9999, f"Parity failed: {m}"
