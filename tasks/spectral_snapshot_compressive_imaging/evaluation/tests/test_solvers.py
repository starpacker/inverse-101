"""Unit tests for src/solvers.py.

Note: The pretrained HSI_SDeCNN network is provided as-is and not tested here.
We only test the solver logic we implemented: TV denoiser, CNN/TV scheduling,
band-by-band denoising wrapper, and the overall GAP reconstruction."""

import os
import numpy as np
import pytest

from src.solvers import (load_denoiser, TV_denoiser, _use_cnn,
                          _denoise_cnn_bands, gap_denoise)
from src.preprocessing import load_mask, load_ground_truth, generate_measurement

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'solver_fixtures.npz')
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', 'reference_outputs', 'deep_denoiser.pth')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


@pytest.fixture
def fixtures():
    return dict(np.load(FIXTURES_PATH, allow_pickle=True))


# ---------------------------------------------------------------------------
# TV denoiser tests
# ---------------------------------------------------------------------------

class TestTVDenoiser:
    def test_output_shape(self, fixtures):
        """Output shape should match input shape."""
        tv_input = fixtures['tv_input']
        tv_output = TV_denoiser(tv_input, 0.1, 5)
        assert tv_output.shape == tv_input.shape

    def test_exact_values(self, fixtures):
        """Should reproduce fixture values deterministically."""
        tv_input = fixtures['tv_input']
        expected = fixtures['tv_output']
        result = TV_denoiser(tv_input, 0.1, 5)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_identity_on_constant(self):
        """Constant image should be unchanged by TV denoising."""
        x = np.ones((16, 16, 3)) * 0.5
        result = TV_denoiser(x, 0.1, 10)
        np.testing.assert_allclose(result, x, atol=1e-10)

    def test_reduces_noise(self):
        """TV denoiser should reduce noise (lower MSE vs clean)."""
        np.random.seed(123)
        clean = np.ones((32, 32, 3)) * 0.5
        noisy = clean + np.random.randn(32, 32, 3) * 0.1
        denoised = TV_denoiser(noisy, 0.3, 20)
        mse_before = np.mean((noisy - clean) ** 2)
        mse_after = np.mean((denoised - clean) ** 2)
        assert mse_after < mse_before


# ---------------------------------------------------------------------------
# _use_cnn schedule tests
# ---------------------------------------------------------------------------

class TestUseCNN:
    def test_no_cnn_before_83(self):
        """CNN should not be used before iteration 83."""
        for k in range(83):
            assert not _use_cnn(k), f"_use_cnn should be False at k={k}"

    def test_cnn_at_expected_iters(self, fixtures):
        """CNN iterations should match fixture schedule."""
        expected = list(fixtures['cnn_iters'])
        actual = [k for k in range(130) if _use_cnn(k)]
        assert actual == expected

    def test_tv_gaps_in_cnn_phase(self, fixtures):
        """TV fallback iterations within CNN phase should match fixture."""
        expected = list(fixtures['tv_iters_in_cnn_phase'])
        actual = [k for k in range(83, 126) if not _use_cnn(k)]
        assert actual == expected

    def test_alternating_pattern(self):
        """CNN phase should follow 3-CNN, 1-TV alternating pattern."""
        for start in range(83, 122, 4):
            # 3 CNN iterations
            for offset in range(3):
                assert _use_cnn(start + offset), f"Expected CNN at k={start + offset}"
            # 1 TV iteration (except the last group which is extended)
            if start + 3 < 122:
                assert not _use_cnn(start + 3), f"Expected TV at k={start + 3}"


# ---------------------------------------------------------------------------
# _denoise_cnn_bands wrapper tests
# ---------------------------------------------------------------------------

class TestDenoiseCNNBands:
    @pytest.fixture
    def model_and_device(self):
        return load_denoiser(CHECKPOINT_PATH)

    def test_output_shape(self, model_and_device):
        """Output should have same shape as input (H, W, nC)."""
        model, device = model_and_device
        np.random.seed(42)
        x = np.random.rand(64, 64, 31).astype(np.float64)
        result = _denoise_cnn_bands(x, model, device, 31, (10, 10, 10))
        assert result.shape == (64, 64, 31)

    def test_deterministic(self, model_and_device):
        """Same input should produce same output."""
        model, device = model_and_device
        np.random.seed(42)
        x = np.random.rand(64, 64, 31).astype(np.float64)
        result1 = _denoise_cnn_bands(x, model, device, 31, (10, 10, 10))
        result2 = _denoise_cnn_bands(x, model, device, 31, (10, 10, 10))
        np.testing.assert_allclose(result1, result2, rtol=1e-5)

    def test_different_noise_levels(self, model_and_device):
        """Different noise levels should produce different outputs."""
        model, device = model_and_device
        np.random.seed(42)
        x = np.random.rand(32, 32, 31).astype(np.float64)
        out_low = _denoise_cnn_bands(x, model, device, 31, (5, 5, 5))
        out_high = _denoise_cnn_bands(x, model, device, 31, (50, 50, 50))
        assert not np.allclose(out_low, out_high)


# ---------------------------------------------------------------------------
# GAP solver integration tests
# ---------------------------------------------------------------------------

class TestGAPDenoise:
    def test_improves_over_iterations(self):
        """GAP solver PSNR should increase over iterations."""
        r, c, nC, step = 256, 256, 31, 1
        mask_3d = load_mask(os.path.join(DATA_DIR, 'mask256.mat'), r, c, nC, step)
        truth = load_ground_truth(os.path.join(DATA_DIR, 'kaist_crop256_01.mat'))
        meas = generate_measurement(truth, mask_3d, step)

        _, psnr_all, _ = gap_denoise(
            meas, mask_3d,
            _lambda=1, accelerate=True, iter_max=20,
            sigma=[130, 130, 130, 130, 130, 130, 130, 130],
            tv_iter_max=5, X_orig=truth,
            checkpoint_path=CHECKPOINT_PATH,
            show_iqa=True,
        )
        # PSNR should generally increase; compare early vs late
        assert psnr_all[-1] > psnr_all[0]
        # Final PSNR should be above 35 dB for this dataset
        assert psnr_all[-1] > 35.0

    def test_returns_correct_shapes(self):
        """GAP solver should return shifted cube and metric lists."""
        r, c, nC, step = 256, 256, 31, 1
        mask_3d = load_mask(os.path.join(DATA_DIR, 'mask256.mat'), r, c, nC, step)
        truth = load_ground_truth(os.path.join(DATA_DIR, 'kaist_crop256_01.mat'))
        meas = generate_measurement(truth, mask_3d, step)

        x, psnr_all, ssim_all = gap_denoise(
            meas, mask_3d,
            _lambda=1, accelerate=True, iter_max=20,
            sigma=[130, 130, 130, 130, 130, 130, 130, 130],
            tv_iter_max=5, X_orig=truth,
            checkpoint_path=CHECKPOINT_PATH,
            show_iqa=True,
        )
        # Shifted output shape
        assert x.shape == (r, c + step * (nC - 1), nC)
        # 124 iterations total
        assert len(psnr_all) == 124
        assert len(ssim_all) == 124
