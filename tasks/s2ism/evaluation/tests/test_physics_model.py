"""Unit tests for src/physics_model.py"""

import numpy as np
import pytest
from pathlib import Path


def test_forward_model_shape():
    from src.physics_model import forward_model
    Nz, Ny, Nx, Nch = 2, 31, 31, 9
    gt = np.random.rand(Nz, Ny, Nx)
    psf = np.random.rand(Nz, 15, 15, Nch)
    result = forward_model(gt, psf)
    assert result.shape == (Nz, Ny, Nx, Nch)


def test_forward_model_non_negative():
    from src.physics_model import forward_model
    Nz, Ny, Nx, Nch = 2, 31, 31, 9
    gt = np.abs(np.random.rand(Nz, Ny, Nx))
    psf = np.abs(np.random.rand(Nz, 15, 15, Nch))
    result = forward_model(gt, psf)
    assert np.all(result >= 0)


def test_psf_width_odd():
    """psf_width must always return an odd number."""
    import brighteyes_ism.simulation.PSF_sim as psf_sim
    from src.physics_model import psf_width

    exPar = psf_sim.simSettings()
    exPar.na = 1.4
    exPar.wl = 640
    exPar.n = 1.5

    result = psf_width(40, 720, 2, exPar, 50 * 1e3 / 500)
    assert result % 2 == 1


def test_find_upsampling_positive():
    from src.physics_model import find_upsampling
    ups = find_upsampling(40, 4)
    assert ups >= 1
    assert isinstance(int(ups), int)
