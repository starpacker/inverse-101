"""Tests for preprocessing module."""

import os
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing import (
    load_data,
    load_metadata,
    calibrate_datacube,
    compute_dp_mean,
    compute_bf_mask,
    compute_virtual_images,
)

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_DIR = os.path.join(TASK_DIR, "data")
FIXTURES_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "preprocessing")


@pytest.fixture(scope="module")
def data():
    datacube, probe = load_data(DATA_DIR)
    return datacube, probe


@pytest.fixture(scope="module")
def meta():
    return load_metadata(DATA_DIR)


class TestLoadData:
    def test_datacube_shape(self, data):
        datacube, _ = data
        assert datacube.shape == (48, 48, 192, 192)

    def test_probe_shape(self, data):
        _, probe = data
        assert probe.shape == (192, 192)

    def test_datacube_dtype(self, data):
        datacube, _ = data
        assert datacube.dtype == np.float32


class TestLoadMetadata:
    def test_required_keys(self, meta):
        required = [
            "energy_eV", "R_pixel_size_A", "convergence_semiangle_mrad",
            "scan_shape", "detector_shape", "defocus_A",
            "com_rotation_deg",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"

    def test_energy(self, meta):
        assert meta["energy_eV"] == 300000

    def test_scan_shape(self, meta):
        assert meta["scan_shape"] == [48, 48]


class TestCalibration:
    def test_probe_radius(self, data, meta):
        datacube, probe = data
        fix = np.load(os.path.join(FIXTURES_DIR, "calibration.npz"))
        radius, center = calibrate_datacube(
            datacube, probe,
            R_pixel_size=meta["R_pixel_size_A"],
            convergence_semiangle=meta["convergence_semiangle_mrad"],
        )
        np.testing.assert_allclose(radius, fix["probe_radius"], rtol=1e-4)
        np.testing.assert_allclose(center[0], fix["probe_center_qx"], rtol=1e-4)
        np.testing.assert_allclose(center[1], fix["probe_center_qy"], rtol=1e-4)


class TestDpMean:
    def test_shape(self, data):
        datacube, _ = data
        dp_mean = compute_dp_mean(datacube)
        assert dp_mean.shape == (192, 192)

    def test_values(self, data):
        datacube, _ = data
        dp_mean = compute_dp_mean(datacube)
        fix = np.load(os.path.join(FIXTURES_DIR, "dp_mean.npz"))
        np.testing.assert_allclose(dp_mean, fix["dp_mean"], rtol=1e-5)


class TestBfMask:
    def test_shape_and_sum(self, data, meta):
        datacube, _ = data
        dp_mean = compute_dp_mean(datacube)
        mask = compute_bf_mask(dp_mean, threshold=0.8)
        fix = np.load(os.path.join(FIXTURES_DIR, "dp_mask.npz"))
        assert mask.shape == (192, 192)
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, fix["dp_mask"])


class TestVirtualImages:
    def test_shapes(self, data, meta):
        datacube, probe = data
        radius, center = calibrate_datacube(
            datacube, probe,
            R_pixel_size=meta["R_pixel_size_A"],
            convergence_semiangle=meta["convergence_semiangle_mrad"],
        )
        bf, df = compute_virtual_images(datacube, center, radius)
        assert bf.shape == (48, 48)
        assert df.shape == (48, 48)
