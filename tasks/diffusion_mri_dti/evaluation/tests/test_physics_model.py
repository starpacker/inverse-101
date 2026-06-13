"""Tests for src/physics_model.py."""

import numpy as np
import pytest
import sys, os

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

from src.physics_model import (
    tensor_from_elements,
    elements_from_tensor,
    tensor_from_eig,
    build_design_matrix,
    stejskal_tanner_signal,
    add_rician_noise,
    compute_fa,
    compute_md,
)


class TestTensorConstruction:
    def test_roundtrip_elements(self):
        """Tensor -> elements -> tensor roundtrip."""
        D = np.array([[1.0, 0.2, 0.1],
                       [0.2, 0.8, 0.15],
                       [0.1, 0.15, 0.6]])
        elems = elements_from_tensor(D)
        assert elems.shape == (6,)
        D_reconstructed = tensor_from_elements(*elems)
        np.testing.assert_allclose(D_reconstructed, D, rtol=1e-12)

    def test_symmetry(self):
        """Constructed tensor is symmetric."""
        D = tensor_from_elements(1.0, 0.2, 0.1, 0.8, 0.15, 0.6)
        np.testing.assert_allclose(D, D.T, rtol=1e-12)

    def test_batch_elements(self):
        """Batch tensor element extraction."""
        D = np.random.randn(5, 3, 3)
        D = (D + np.swapaxes(D, -2, -1)) / 2  # symmetrize
        elems = elements_from_tensor(D)
        assert elems.shape == (5, 6)

    def test_from_eigendecomposition(self):
        """Tensor from eigenvalues/eigenvectors matches direct construction."""
        evals = np.array([1.5, 0.5, 0.3])
        evecs = np.eye(3)
        D = tensor_from_eig(evals, evecs)
        np.testing.assert_allclose(np.diag(D), evals, rtol=1e-12)
        np.testing.assert_allclose(D - np.diag(np.diag(D)), 0, atol=1e-12)


class TestDesignMatrix:
    def test_shape(self):
        """Design matrix has correct shape."""
        bvals = np.array([0, 1000, 1000, 1000], dtype=np.float64)
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        B = build_design_matrix(bvals, bvecs)
        assert B.shape == (4, 7)

    def test_b0_row(self):
        """b=0 row has [1, 0, 0, 0, 0, 0, 0]."""
        bvals = np.array([0, 1000], dtype=np.float64)
        bvecs = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        B = build_design_matrix(bvals, bvecs)
        np.testing.assert_allclose(B[0], [1, 0, 0, 0, 0, 0, 0])

    def test_known_direction(self):
        """b=1000, g=[1,0,0] gives [-b, 0, 0, 0, 0, 0] for tensor columns."""
        bvals = np.array([1000.0])
        bvecs = np.array([[1.0, 0.0, 0.0]])
        B = build_design_matrix(bvals, bvecs)
        np.testing.assert_allclose(B[0, 1:], [-1000, 0, 0, 0, 0, 0])


class TestStejskalTanner:
    def test_b0_signal(self):
        """At b=0, signal equals S0."""
        D = np.diag([1e-3, 0.5e-3, 0.5e-3])
        S0 = 1.0
        bvals = np.array([0.0])
        bvecs = np.array([[0.0, 0.0, 0.0]])
        signal = stejskal_tanner_signal(S0, D, bvals, bvecs)
        np.testing.assert_allclose(signal, [1.0], rtol=1e-10)

    def test_known_decay(self):
        """Signal along x for isotropic tensor matches exp(-b*D)."""
        D_val = 1e-3
        D = np.diag([D_val, D_val, D_val])
        S0 = 1.0
        b = 1000.0
        bvals = np.array([b])
        bvecs = np.array([[1.0, 0.0, 0.0]])
        signal = stejskal_tanner_signal(S0, D, bvals, bvecs)
        expected = np.exp(-b * D_val)
        np.testing.assert_allclose(signal[0], expected, rtol=1e-10)

    def test_anisotropic_direction_dependence(self):
        """Signal differs along principal vs perpendicular directions for anisotropic tensor."""
        D = np.diag([1.5e-3, 0.3e-3, 0.3e-3])  # anisotropic along x
        S0 = 1.0
        b = 1000.0
        bvals = np.array([b, b])
        bvecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        signal = stejskal_tanner_signal(S0, D, bvals, bvecs)
        # Along x (high diffusivity): more signal loss
        # Along y (low diffusivity): less signal loss
        assert signal[0] < signal[1]

    def test_monotonic_with_b(self):
        """Signal decreases with increasing b-value."""
        D = np.diag([1e-3, 1e-3, 1e-3])
        S0 = 1.0
        bvals = np.array([0, 500, 1000, 2000], dtype=np.float64)
        bvecs = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)
        signal = stejskal_tanner_signal(S0, D, bvals, bvecs)
        assert all(signal[i] >= signal[i + 1] for i in range(len(signal) - 1))


class TestRicianNoise:
    def test_shape_preserved(self):
        signal = np.ones((10, 10, 5))
        noisy = add_rician_noise(signal, 0.1, rng=np.random.default_rng(42))
        assert noisy.shape == signal.shape

    def test_non_negative(self):
        signal = np.ones((10, 10, 5))
        noisy = add_rician_noise(signal, 0.5, rng=np.random.default_rng(42))
        assert np.all(noisy >= 0)

    def test_noise_increases_variance(self):
        signal = np.ones((100, 100)) * 0.5
        noisy = add_rician_noise(signal, 0.1, rng=np.random.default_rng(42))
        assert np.std(noisy) > 0.01


class TestScalarMaps:
    def test_fa_isotropic(self):
        """FA is 0 for isotropic tensor."""
        evals = np.array([1e-3, 1e-3, 1e-3])
        fa = compute_fa(evals)
        np.testing.assert_allclose(fa, 0.0, atol=1e-10)

    def test_fa_fully_anisotropic(self):
        """FA is 1 for fully anisotropic tensor (one nonzero eigenvalue)."""
        evals = np.array([1e-3, 0, 0])
        fa = compute_fa(evals)
        np.testing.assert_allclose(fa, 1.0, atol=1e-10)

    def test_fa_range(self):
        """FA is in [0, 1]."""
        rng = np.random.default_rng(42)
        evals = rng.uniform(0, 2e-3, (100, 3))
        fa = compute_fa(evals)
        assert np.all(fa >= 0) and np.all(fa <= 1)

    def test_md_known(self):
        """MD is mean of eigenvalues."""
        evals = np.array([1.5e-3, 0.5e-3, 0.3e-3])
        md = compute_md(evals)
        expected = (1.5e-3 + 0.5e-3 + 0.3e-3) / 3
        np.testing.assert_allclose(md, expected, rtol=1e-10)

    def test_md_batch(self):
        evals = np.array([[1e-3, 1e-3, 1e-3], [2e-3, 1e-3, 0.5e-3]])
        md = compute_md(evals)
        assert md.shape == (2,)
        np.testing.assert_allclose(md[0], 1e-3)
