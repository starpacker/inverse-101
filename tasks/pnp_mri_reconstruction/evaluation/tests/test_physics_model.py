import os
import pytest
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "physics_model")

import sys
sys.path.insert(0, TASK_DIR)
from src.physics_model import MRIForwardModel


class TestGenerateMask:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.input_fix = np.load(os.path.join(FIXTURE_DIR, "input_generate_mask.npz"),
                                allow_pickle=True)
        self.output_fix = np.load(os.path.join(FIXTURE_DIR, "output_generate_mask.npz"),
                                allow_pickle=True)

    def test_shape(self):
        image_size = self.input_fix["image_size"]
        mask = MRIForwardModel.generate_mask(image_size, 36)
        expected_shape = tuple(self.output_fix["mask_shape"])
        assert mask.shape == expected_shape

    def test_dtype(self):
        image_size = self.input_fix["image_size"]
        mask = MRIForwardModel.generate_mask(image_size, 36)
        assert mask.dtype == bool

    def test_num_sampled(self):
        image_size = self.input_fix["image_size"]
        mask = MRIForwardModel.generate_mask(image_size, 36)
        expected_sum = int(self.output_fix["mask_sum"])
        assert mask.sum() == expected_sum

    def test_even_size_required(self):
        with pytest.raises(ValueError):
            MRIForwardModel.generate_mask(np.array([321, 320]), 36)


class TestForward:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.output_fix = np.load(os.path.join(FIXTURE_DIR, "output_forward.npz"))
        image_size = np.array([320, 320])
        mask = MRIForwardModel.generate_mask(image_size, 36)
        self.model = MRIForwardModel(mask)

    def test_output_shape(self):
        x = np.random.randn(320, 320)
        y = self.model.forward(x)
        expected_shape = tuple(self.output_fix["y_shape"])
        assert y.shape == expected_shape

    def test_masked_zeros(self):
        x = np.random.randn(320, 320)
        y = self.model.forward(x)
        assert np.all(y[~self.model.mask] == 0)


class TestAdjoint:
    @pytest.fixture(autouse=True)
    def setup(self):
        image_size = np.array([320, 320])
        mask = MRIForwardModel.generate_mask(image_size, 36)
        self.model = MRIForwardModel(mask)

    def test_adjoint_identity(self):
        """Test <Ax, y> = <x, A^H y> for random x, y."""
        np.random.seed(42)
        x = np.random.randn(320, 320)
        z = np.random.randn(320, 320) + 1j * np.random.randn(320, 320)

        ax = self.model.forward(x)
        ah_z = self.model.adjoint(z)

        lhs = np.vdot(ax.flatten(), z.flatten())
        rhs = np.vdot(x.flatten(), ah_z.flatten())

        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


class TestGrad:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.output_fix = np.load(os.path.join(FIXTURE_DIR, "output_grad.npz"))
        image_size = np.array([320, 320])
        mask = MRIForwardModel.generate_mask(image_size, 36)
        self.model = MRIForwardModel(mask)
        from src.preprocessing import load_observation, normalize_image
        obs = load_observation(os.path.join(TASK_DIR, "data"))
        self.img = normalize_image(obs["img"])

    def test_grad_shape(self):
        y = self.model.forward(self.img)
        g, _ = self.model.grad(self.img, y)
        expected_shape = tuple(self.output_fix["output_grad_shape"])
        assert g.shape == expected_shape

    def test_grad_at_true_is_zero(self):
        """Gradient at x_true should be zero (since Ax_true = y)."""
        y = self.model.forward(self.img)
        g, cost = self.model.grad(self.img, y)
        np.testing.assert_allclose(g, 0, atol=1e-10)
        np.testing.assert_allclose(cost, 0, atol=1e-10)

    def test_grad_is_real(self):
        y = self.model.forward(self.img)
        g, _ = self.model.grad(self.img, y)
        assert np.isreal(g).all()


class TestIFFTRecon:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.output_fix = np.load(os.path.join(FIXTURE_DIR, "output_ifft_recon.npz"))
        image_size = np.array([320, 320])
        mask = MRIForwardModel.generate_mask(image_size, 36)
        self.model = MRIForwardModel(mask)

    def test_output_shape(self):
        from src.preprocessing import load_observation, normalize_image
        obs = load_observation(os.path.join(TASK_DIR, "data"))
        img = normalize_image(obs["img"])
        y = self.model.forward(img)
        recon = self.model.ifft_recon(y)
        expected_shape = tuple(self.output_fix["output_shape"])
        assert recon.shape == expected_shape

    def test_snr(self):
        from src.preprocessing import load_observation, normalize_image
        from src.visualization import compute_snr
        obs = load_observation(os.path.join(TASK_DIR, "data"))
        img = normalize_image(obs["img"])
        y = self.model.forward(img)
        recon = self.model.ifft_recon(y)
        snr = compute_snr(img, recon)
        expected_snr = float(self.output_fix["output_snr"])
        np.testing.assert_allclose(snr, expected_snr, rtol=1e-3)
