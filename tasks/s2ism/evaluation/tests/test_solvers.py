"""Unit tests for src/solvers.py"""

import numpy as np
import torch
import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / 'fixtures'


def test_partial_convolution_rfft():
    from src.solvers import partial_convolution_rfft
    fix = np.load(FIXTURES / 'partial_conv.npz')
    kernel = torch.from_numpy(fix['input_kernel'])
    volume = torch.from_numpy(fix['input_volume'])
    expected = fix['output_conv']
    result = partial_convolution_rfft(kernel, volume, dim1='ijk', dim2='jkl', axis='jk')
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_amd_update_preserves_nonnegativity():
    from src.solvers import amd_update_fft
    from torch.fft import rfftn
    np.random.seed(0)
    Nz, Ny, Nx, Nch = 2, 11, 11, 4
    h = torch.abs(torch.randn(Nz, Ny, Nx, Nch, dtype=torch.float64))
    h = h / h.sum(dim=(1, 2, 3), keepdim=True)
    ht = torch.flip(h, [1, 2])
    flip_ax = [1, 2]
    padding = [h.size(d) for d in flip_ax]
    h_fft = rfftn(h, dim=flip_ax, s=padding)
    ht_fft = rfftn(ht, dim=flip_ax, s=padding)
    img = torch.abs(torch.randn(Ny, Nx, Nch, dtype=torch.float64)) + 0.1
    obj = torch.ones(Nz, Ny, Nx, dtype=torch.float64)
    eps = torch.finfo(torch.float).eps
    obj_new = amd_update_fft(img, obj, h_fft, ht_fft, eps)
    assert torch.all(obj_new >= 0), "ML-EM update must preserve non-negativity"


def test_max_likelihood_reconstruction_shapes():
    """Integration test: check output shapes of the reconstruction."""
    from src.solvers import max_likelihood_reconstruction
    np.random.seed(0)
    Ny, Nx, Nch = 21, 21, 4
    Nz = 2
    dset = np.abs(np.random.randn(Ny, Nx, Nch)) + 0.01
    psf = np.abs(np.random.randn(Nz, 11, 11, Nch))
    psf = psf / psf.sum(axis=(1, 2, 3), keepdims=True)

    recon, counts, diff, n_iter = max_likelihood_reconstruction(
        dset, psf, stop='fixed', max_iter=3, rep_to_save='last', process='cpu')

    assert recon.shape == (Nz, Ny, Nx)
    assert n_iter == 4  # k increments one past max_iter due to while loop


def test_amd_stop_fixed():
    from src.solvers import amd_stop
    o_old = torch.ones(2, 5, 5)
    o_new = torch.ones(2, 5, 5) * 1.1
    pre_flag, flag, counts, diff = amd_stop(
        o_old, o_new, True, True, 'fixed', 5, 1e-3, 100.0, 2, 5)
    assert flag == False, "Should stop at max_iter for 'fixed' mode"
