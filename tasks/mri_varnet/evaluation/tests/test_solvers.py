"""Unit tests for src/solvers.py."""
import os, sys, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing import prepare_data, apply_mask, get_complex_kspace, load_observation
from src.solvers import load_varnet, varnet_reconstruct, varnet_reconstruct_batch
from src.visualization import compute_metrics
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
REF_DIR = os.path.join(os.path.dirname(__file__), "../reference_outputs")

@pytest.fixture(scope="module")
def model():
    return load_varnet(os.path.join(DATA_DIR, "varnet_knee_state_dict.pt"))

class TestVarNet:
    def test_single_slice(self, model):
        obs = load_observation(DATA_DIR)
        ks = get_complex_kspace(obs)
        masked, mask = apply_mask(ks[0])
        recon = varnet_reconstruct(model, masked, mask)
        assert recon.shape[0] == 640 and recon.shape[1] == 368

    def test_batch(self, model):
        ks, gt, meta = prepare_data(DATA_DIR)
        recons, zfs = varnet_reconstruct_batch(
            model, ks, target_h=320, target_w=320)
        assert recons.shape == (1, 320, 320)
        assert zfs.shape == (1, 320, 320)

    def test_parity_with_reference(self, model):
        ks, gt, meta = prepare_data(DATA_DIR)
        recons, _ = varnet_reconstruct_batch(
            model, ks, target_h=320, target_w=320)
        ref = np.load(os.path.join(REF_DIR, "varnet_reconstruction.npz"))["reconstruction"]
        m = compute_metrics(recons[0], ref[0])
        assert m["ncc"] > 0.999, f"Parity failed: {m}"

    def test_better_than_zerofill(self, model):
        ks, gt, _ = prepare_data(DATA_DIR)
        recons, zfs = varnet_reconstruct_batch(
            model, ks, target_h=320, target_w=320)
        m_vn = compute_metrics(recons[0], gt[0])
        m_zf = compute_metrics(zfs[0], gt[0])
        assert m_vn["ssim"] > m_zf["ssim"]
