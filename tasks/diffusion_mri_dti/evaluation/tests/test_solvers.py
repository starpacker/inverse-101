"""Tests for src/solvers.py."""

import numpy as np
import pytest
import sys, os

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

from src.physics_model import (
    stejskal_tanner_signal,
    tensor_from_elements,
    compute_fa,
    compute_md,
)
from src.generate_data import generate_gradient_table
from src.solvers import fit_dti_ols, fit_dti_wls, tensor_eig_decomposition


def _make_clean_dwi(tensor_elems_true, S0_val, bvals, bvecs):
    """Generate clean DWI signal for a single voxel embedded in a small image."""
    D = tensor_from_elements(*tensor_elems_true)
    signal = stejskal_tanner_signal(S0_val, D, bvals, bvecs)  # (N_volumes,)
    # Place in a 3x3 image
    dwi = np.zeros((3, 3, len(bvals)), dtype=np.float64)
    dwi[1, 1, :] = signal
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    return dwi, mask


class TestOLSFit:
    def test_exact_recovery_clean(self):
        """OLS recovers exact tensor elements from clean (noiseless) data."""
        bvals, bvecs = generate_gradient_table(n_directions=30, b_value=1000.0)
        true_elems = [1.2e-3, 0.0, 0.0, 0.4e-3, 0.0, 0.4e-3]  # diagonal tensor
        S0_val = 1.0
        dwi, mask = _make_clean_dwi(true_elems, S0_val, bvals, bvecs)

        tensor_est, S0_est = fit_dti_ols(dwi, bvals, bvecs, mask=mask)

        np.testing.assert_allclose(tensor_est[1, 1], true_elems, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(S0_est[1, 1], S0_val, rtol=1e-6)

    def test_shape(self):
        bvals, bvecs = generate_gradient_table(n_directions=30, b_value=1000.0)
        dwi = np.ones((5, 5, len(bvals)))
        mask = np.ones((5, 5), dtype=bool)
        tensor_est, S0_est = fit_dti_ols(dwi, bvals, bvecs, mask=mask)
        assert tensor_est.shape == (5, 5, 6)
        assert S0_est.shape == (5, 5)

    def test_mask_exclusion(self):
        """Unmasked voxels should remain zero."""
        bvals, bvecs = generate_gradient_table(n_directions=30, b_value=1000.0)
        dwi = np.ones((5, 5, len(bvals)))
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        tensor_est, S0_est = fit_dti_ols(dwi, bvals, bvecs, mask=mask)
        assert np.all(tensor_est[0, 0] == 0)
        assert S0_est[0, 0] == 0


class TestWLSFit:
    def test_exact_recovery_clean(self):
        """WLS also recovers exact tensor from clean data."""
        bvals, bvecs = generate_gradient_table(n_directions=30, b_value=1000.0)
        true_elems = [1.0e-3, 0.1e-3, 0.0, 0.8e-3, 0.0, 0.5e-3]
        S0_val = 1.0
        dwi, mask = _make_clean_dwi(true_elems, S0_val, bvals, bvecs)

        tensor_est, S0_est = fit_dti_wls(dwi, bvals, bvecs, mask=mask)

        np.testing.assert_allclose(tensor_est[1, 1], true_elems, rtol=1e-5, atol=1e-10)
        np.testing.assert_allclose(S0_est[1, 1], S0_val, rtol=1e-5)

    def test_wls_better_than_ols_noisy(self):
        """WLS should be at least as accurate as OLS on noisy data (statistical)."""
        rng = np.random.default_rng(42)
        bvals, bvecs = generate_gradient_table(n_directions=30, b_value=1000.0)
        true_elems = [1.2e-3, 0.0, 0.0, 0.3e-3, 0.0, 0.3e-3]
        S0_val = 1.0
        D = tensor_from_elements(*true_elems)
        true_evals = np.array([1.2e-3, 0.3e-3, 0.3e-3])
        true_fa = compute_fa(true_evals)

        # Run many noisy trials
        n_trials = 50
        fa_errors_ols = []
        fa_errors_wls = []

        for _ in range(n_trials):
            signal = stejskal_tanner_signal(S0_val, D, bvals, bvecs)
            noise_r = rng.normal(0, 0.03, signal.shape)
            noise_i = rng.normal(0, 0.03, signal.shape)
            noisy = np.sqrt((signal + noise_r) ** 2 + noise_i ** 2)

            dwi = np.zeros((3, 3, len(bvals)))
            dwi[1, 1] = noisy
            mask = np.zeros((3, 3), dtype=bool)
            mask[1, 1] = True

            t_ols, _ = fit_dti_ols(dwi, bvals, bvecs, mask=mask)
            t_wls, _ = fit_dti_wls(dwi, bvals, bvecs, mask=mask)

            _, _, fa_ols, _ = tensor_eig_decomposition(t_ols, mask=mask)
            _, _, fa_wls, _ = tensor_eig_decomposition(t_wls, mask=mask)

            fa_errors_ols.append(abs(fa_ols[1, 1] - true_fa))
            fa_errors_wls.append(abs(fa_wls[1, 1] - true_fa))

        # WLS should have lower or equal mean error
        assert np.mean(fa_errors_wls) <= np.mean(fa_errors_ols) * 1.1


class TestEigDecomposition:
    def test_fa_md_from_known_tensor(self):
        """FA and MD from known diagonal tensor."""
        elems = np.zeros((3, 3, 6))
        elems[1, 1] = [1.5e-3, 0, 0, 0.3e-3, 0, 0.3e-3]
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True

        evals, evecs, fa_map, md_map = tensor_eig_decomposition(elems, mask=mask)

        # Check eigenvalues sorted descending
        np.testing.assert_allclose(evals[1, 1], [1.5e-3, 0.3e-3, 0.3e-3], rtol=1e-10)

        # Check FA
        expected_fa = compute_fa(np.array([1.5e-3, 0.3e-3, 0.3e-3]))
        np.testing.assert_allclose(fa_map[1, 1], expected_fa, rtol=1e-10)

        # Check MD
        expected_md = (1.5e-3 + 0.3e-3 + 0.3e-3) / 3
        np.testing.assert_allclose(md_map[1, 1], expected_md, rtol=1e-10)

    def test_isotropic_fa_zero(self):
        elems = np.zeros((3, 3, 6))
        elems[1, 1] = [1e-3, 0, 0, 1e-3, 0, 1e-3]
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True

        _, _, fa_map, _ = tensor_eig_decomposition(elems, mask=mask)
        np.testing.assert_allclose(fa_map[1, 1], 0.0, atol=1e-10)
