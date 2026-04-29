"""Tests for physics_model module (rotate_frames, KL basis functions)."""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics_model import (
    rotate_frames,
    compute_kl_basis_svd,
    compute_kl_basis_pca,
    compute_kl_basis_eigh,
    compute_kl_basis,
)


# ---------------------------------------------------------------------------
# rotate_frames
# ---------------------------------------------------------------------------

class TestRotateFrames:
    def test_output_shape(self):
        """Output shape must match input shape."""
        B, N, H, W = 2, 4, 16, 16
        images = torch.randn(B, N, H, W)
        angles = torch.linspace(-10, 10, N)
        out = rotate_frames(images, angles)
        assert out.shape == (B, N, H, W)

    def test_zero_angle_identity(self):
        """Rotation by 0 degrees should leave the image nearly unchanged."""
        B, N, H, W = 1, 3, 16, 16
        images = torch.randn(B, N, H, W)
        angles = torch.zeros(N)
        out = rotate_frames(images, angles)
        # Bilinear interpolation introduces small errors near borders;
        # check inner region only.
        inner = slice(2, -2)
        np.testing.assert_allclose(
            out[0, :, inner, inner].numpy(),
            images[0, :, inner, inner].numpy(),
            atol=0.05,
        )

    def test_180_degree_symmetry(self):
        """Rotating a symmetric image by 180 degrees should be close to itself."""
        H, W = 16, 16
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'
        )
        # Radially symmetric pattern
        img = torch.exp(-(x ** 2 + y ** 2) / 0.3)
        images = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        angles = torch.tensor([180.0])
        out = rotate_frames(images, angles)
        inner = slice(3, -3)
        np.testing.assert_allclose(
            out[0, 0, inner, inner].numpy(),
            images[0, 0, inner, inner].numpy(),
            atol=0.1,
        )

    def test_dtype_float32(self):
        """Output should be float32 when input is float32."""
        images = torch.randn(1, 2, 8, 8, dtype=torch.float32)
        angles = torch.tensor([5.0, -5.0])
        out = rotate_frames(images, angles)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# KL basis construction
# ---------------------------------------------------------------------------

class TestComputeKlBasisSvd:
    def test_output_shape(self):
        """Basis should be (n_pix, K_max)."""
        N, n_pix, K = 8, 64, 3
        ref = torch.randn(N, n_pix)
        basis = compute_kl_basis_svd(ref, K)
        assert basis.shape == (n_pix, K)

    def test_orthonormality(self):
        """Columns of the KL basis should be orthonormal."""
        N, n_pix, K = 10, 32, 4
        ref = torch.randn(N, n_pix)
        basis = compute_kl_basis_svd(ref, K)
        gram = basis.T @ basis  # (K, K)
        np.testing.assert_allclose(
            gram.numpy(), np.eye(K), atol=1e-5
        )

    def test_projection_reduces_variance(self):
        """Projecting data onto K < N basis vectors should capture the most variance."""
        rng = np.random.default_rng(42)
        N, n_pix = 8, 64
        ref = torch.from_numpy(rng.standard_normal((N, n_pix)).astype(np.float32))
        ref = ref - ref.mean(dim=0, keepdim=True)
        basis = compute_kl_basis_svd(ref, K_max=2)
        proj = ref @ basis @ basis.T  # reconstruction
        residual = ref - proj
        # Variance of residual must be smaller than original
        assert residual.var().item() < ref.var().item()


class TestComputeKlBasisEigh:
    def test_output_shape(self):
        N, n_pix, K = 6, 32, 3
        ref = torch.randn(N, n_pix)
        basis = compute_kl_basis_eigh(ref, K)
        assert basis.shape == (n_pix, K)

    def test_spans_same_subspace_as_svd(self):
        """SVD and eigh methods should span the same K-dimensional subspace."""
        rng = np.random.default_rng(123)
        N, n_pix, K = 8, 64, 3
        ref = torch.from_numpy(rng.standard_normal((N, n_pix)).astype(np.float32))
        ref = ref - ref.mean(dim=0, keepdim=True)
        basis_svd = compute_kl_basis_svd(ref, K)
        basis_eigh = compute_kl_basis_eigh(ref, K)
        # Project same data; reconstruction should match.
        recon_svd = ref @ basis_svd @ basis_svd.T
        recon_eigh = ref @ basis_eigh @ basis_eigh.T
        np.testing.assert_allclose(
            recon_svd.numpy(), recon_eigh.numpy(), atol=1e-3
        )


class TestComputeKlBasisDispatch:
    def test_invalid_method_raises(self):
        ref = torch.randn(4, 16)
        with pytest.raises(ValueError, match="method must be one of"):
            compute_kl_basis(ref, K_max=2, method='bogus')

    def test_dispatch_svd(self):
        """Dispatcher with 'svd' should return same result as direct call."""
        ref = torch.randn(6, 32)
        direct = compute_kl_basis_svd(ref, 2)
        dispatched = compute_kl_basis(ref, K_max=2, method='svd')
        np.testing.assert_allclose(
            dispatched.numpy(), direct.numpy(), atol=1e-6
        )
