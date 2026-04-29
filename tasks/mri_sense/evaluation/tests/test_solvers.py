"""Unit tests for src/solvers.py."""
import os, sys, pytest, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.preprocessing import prepare_data
from src.solvers import conjugate_gradient, cgsense_reconstruct, cgsense_image_recon
from src.visualization import compute_metrics

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
REF_DIR = os.path.join(os.path.dirname(__file__), "../reference_outputs")

@pytest.fixture(scope="module")
def data():
    return prepare_data(DATA_DIR, R=4, acs_width=16)

class TestConjugateGradient:
    """Unit tests for the CG solver itself."""

    def test_solves_identity(self):
        """CG should solve I*x = b exactly."""
        b = np.array([1.0, 2.0, 3.0])
        x, info = conjugate_gradient(lambda v: v, b)
        np.testing.assert_allclose(x, b, rtol=1e-10)
        assert info == 0

    def test_solves_spd_system(self):
        """CG should solve a symmetric positive-definite system."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 10))
        A = A.T @ A + np.eye(10)  # SPD
        b = rng.standard_normal(10)
        x_expected = np.linalg.solve(A, b)
        x, info = conjugate_gradient(lambda v: A @ v, b)
        np.testing.assert_allclose(x, x_expected, rtol=1e-4)
        assert info == 0

    def test_solves_complex_hermitian(self):
        """CG should solve a complex Hermitian positive-definite system."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))
        A = A.conj().T @ A + np.eye(8)  # HPD
        b = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        x_expected = np.linalg.solve(A, b)
        x, info = conjugate_gradient(lambda v: A @ v, b)
        np.testing.assert_allclose(x, x_expected, rtol=1e-4)
        assert info == 0

    def test_zero_rhs(self):
        """CG with b=0 should return x=0."""
        b = np.zeros(5)
        x, info = conjugate_gradient(lambda v: v, b)
        np.testing.assert_allclose(x, 0, atol=1e-12)


class TestCGSENSE:
    def test_output_shape(self, data):
        kus, sens, _, _, _ = data
        recon = cgsense_reconstruct(kus, sens)
        assert recon.shape == (128, 128)

    def test_output_complex(self, data):
        kus, sens, _, _, _ = data
        recon = cgsense_reconstruct(kus, sens)
        assert np.iscomplexobj(recon)

    def test_image_recon_normalized(self, data):
        kus, sens, _, _, _ = data
        recon = cgsense_image_recon(kus, sens)
        assert recon.max() <= 1.0 + 1e-10
        assert recon.min() >= -1e-10

    def test_better_than_zerofill(self, data):
        kus, sens, _, phantom, _ = data
        gt = phantom / phantom.max()
        recon = cgsense_image_recon(kus, sens)
        from src.physics_model import zero_filled_recon
        zf = zero_filled_recon(kus)
        zf_norm = zf / zf.max()
        m_s = compute_metrics(recon, gt)
        m_zf = compute_metrics(zf_norm, gt)
        assert m_s["ncc"] > m_zf["ncc"]

    def test_parity_with_reference(self, data):
        kus, sens, _, _, _ = data
        recon = cgsense_image_recon(kus, sens)
        ref = np.load(os.path.join(REF_DIR, "sense_reconstruction.npz"))["reconstruction"][0]
        m = compute_metrics(recon, ref.astype(np.float64))
        assert m["ncc"] > 0.9999, f"Parity failed: {m}"
