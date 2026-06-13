"""Unit tests for physics_model.py."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.physics_model import (
    nufft_forward,
    nufft_adjoint,
    multicoil_nufft_forward,
    compute_density_compensation,
    gridding_reconstruct,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "../fixtures/physics_model")


class TestNufftForward:
    def setup_method(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_nufft_forward.npz"))
        out = np.load(os.path.join(FIXTURE_DIR, "output_nufft_forward.npz"))
        self.image = inp["image"]
        self.coord = inp["coord"]
        self.expected = out["kdata"]

    def test_output_shape(self):
        result = nufft_forward(self.image, self.coord)
        assert result.shape == self.expected.shape

    def test_output_values(self):
        result = nufft_forward(self.image, self.coord)
        np.testing.assert_allclose(result, self.expected, rtol=1e-5)

    def test_output_dtype(self):
        result = nufft_forward(self.image, self.coord)
        assert np.iscomplexobj(result)


class TestNufftAdjoint:
    def setup_method(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_nufft_forward.npz"))
        out = np.load(os.path.join(FIXTURE_DIR, "output_nufft_adjoint.npz"))
        self.kdata = np.load(os.path.join(FIXTURE_DIR, "output_nufft_forward.npz"))["kdata"]
        self.coord = inp["coord"]
        self.image_shape = tuple(inp["image"].shape)
        self.expected = out["image"]

    def test_output_shape(self):
        result = nufft_adjoint(self.kdata, self.coord, self.image_shape)
        assert result.shape == self.image_shape

    def test_output_values(self):
        result = nufft_adjoint(self.kdata, self.coord, self.image_shape)
        np.testing.assert_allclose(result, self.expected, rtol=1e-5)

    def test_adjoint_dot_product(self):
        """Test adjoint relationship: <Ax, y> = <x, A^H y>."""
        inp = np.load(os.path.join(FIXTURE_DIR, "input_nufft_forward.npz"))
        image = inp["image"]
        coord = inp["coord"]

        # Forward
        kdata = nufft_forward(image, coord)
        # Random k-space data
        rng = np.random.RandomState(0)
        y = rng.randn(*kdata.shape) + 1j * rng.randn(*kdata.shape)
        # Adjoint
        adj = nufft_adjoint(y, coord, image.shape)

        lhs = np.vdot(kdata, y)
        rhs = np.vdot(image, adj)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-4)


class TestMulticoilForward:
    def setup_method(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_multicoil_forward.npz"))
        out = np.load(os.path.join(FIXTURE_DIR, "output_multicoil_forward.npz"))
        self.image = inp["image"]
        self.coil_maps = inp["coil_maps"]
        self.coord = inp["coord"]
        self.expected = out["kdata"]

    def test_output_shape(self):
        result = multicoil_nufft_forward(self.image, self.coil_maps, self.coord)
        assert result.shape == self.expected.shape

    def test_output_values(self):
        result = multicoil_nufft_forward(self.image, self.coil_maps, self.coord)
        np.testing.assert_allclose(result, self.expected, rtol=1e-5)


class TestDensityCompensation:
    def setup_method(self):
        out = np.load(os.path.join(FIXTURE_DIR, "output_dcf.npz"))
        self.expected_dcf = out["dcf"]
        self.coord = out["coord"]

    def test_dcf_positive(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_nufft_forward.npz"))
        dcf = compute_density_compensation(self.coord, tuple(inp["image"].shape), max_iter=10)
        assert np.all(dcf >= 0)

    def test_dcf_values(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_nufft_forward.npz"))
        dcf = compute_density_compensation(self.coord, tuple(inp["image"].shape), max_iter=10)
        np.testing.assert_allclose(dcf, self.expected_dcf, rtol=1e-4)


class TestGriddingReconstruct:
    def setup_method(self):
        out = np.load(os.path.join(FIXTURE_DIR, "output_gridding.npz"))
        self.expected = out["reconstruction"]

        inp = np.load(os.path.join(FIXTURE_DIR, "input_multicoil_forward.npz"))
        self.coil_maps = inp["coil_maps"]
        self.coord = inp["coord"]

        mc_out = np.load(os.path.join(FIXTURE_DIR, "output_multicoil_forward.npz"))
        self.kdata = mc_out["kdata"]

        dcf_out = np.load(os.path.join(FIXTURE_DIR, "output_dcf.npz"))
        self.dcf = dcf_out["dcf"]

    def test_output_shape(self):
        result = gridding_reconstruct(self.kdata, self.coord, self.coil_maps, self.dcf)
        assert result.shape == self.coil_maps.shape[1:]

    def test_output_values(self):
        result = gridding_reconstruct(self.kdata, self.coord, self.coil_maps, self.dcf)
        np.testing.assert_allclose(result, self.expected, rtol=1e-4)
