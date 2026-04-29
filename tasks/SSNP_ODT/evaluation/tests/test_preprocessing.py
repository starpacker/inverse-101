"""Tests for preprocessing module."""

import os
import sys
import json
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing import load_metadata, load_observation, prepare_data


# ── load_metadata ────────────────────────────────────────────────────────

class TestLoadMetadata:
    def test_loads_valid_json(self, tmp_path):
        """load_metadata should parse a valid meta_data.json."""
        meta = {
            "volume_shape": [4, 8, 8],
            "res_um": [0.1, 0.1, 0.2],
            "wavelength_um": 0.5,
            "n0": 1.0,
            "NA": 0.65,
            "n_angles": 8,
            "ri_contrast_scale": 0.01,
            "tiff_scale": 1.0,
        }
        meta_path = tmp_path / "meta_data.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        result = load_metadata(str(tmp_path))
        assert result["n0"] == 1.0
        assert result["NA"] == 0.65
        assert result["volume_shape"] == [4, 8, 8]
        assert result["n_angles"] == 8

    def test_returns_all_keys(self, tmp_path):
        """Returned dict should contain all required keys."""
        meta = {
            "volume_shape": [4, 8, 8],
            "res_um": [0.1, 0.1, 0.2],
            "wavelength_um": 0.5,
            "n0": 1.0,
            "NA": 0.65,
            "n_angles": 8,
            "ri_contrast_scale": 0.01,
            "tiff_scale": 1.0,
        }
        meta_path = tmp_path / "meta_data.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        result = load_metadata(str(tmp_path))
        expected_keys = {"volume_shape", "res_um", "wavelength_um", "n0",
                         "NA", "n_angles", "ri_contrast_scale", "tiff_scale"}
        assert expected_keys.issubset(set(result.keys()))

    def test_missing_file_raises(self, tmp_path):
        """load_metadata should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_metadata(str(tmp_path))


# ── load_observation ─────────────────────────────────────────────────────

class TestLoadObservation:
    def test_output_dtype_float64(self, tmp_path):
        """load_observation should return float64 array."""
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not installed")

        # Create a small TIFF and metadata
        nz, ny, nx = 4, 8, 8
        raw_data = np.ones((nz, ny, nx), dtype=np.uint16) * 32768
        tifffile.imwrite(str(tmp_path / "sample.tiff"), raw_data)

        meta = {
            "volume_shape": [nz, ny, nx],
            "res_um": [0.1, 0.1, 0.2],
            "wavelength_um": 0.5,
            "n0": 1.0,
            "NA": 0.65,
            "n_angles": 4,
            "ri_contrast_scale": 0.01,
            "tiff_scale": 1.0,
        }
        with open(tmp_path / "meta_data.json", "w") as f:
            json.dump(meta, f)

        result = load_observation(str(tmp_path))
        assert result.dtype == np.float64

    def test_output_shape_matches_tiff(self, tmp_path):
        """Output shape should match the TIFF volume dimensions."""
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not installed")

        nz, ny, nx = 4, 8, 8
        raw_data = np.zeros((nz, ny, nx), dtype=np.uint16)
        tifffile.imwrite(str(tmp_path / "sample.tiff"), raw_data)

        meta = {
            "volume_shape": [nz, ny, nx],
            "res_um": [0.1, 0.1, 0.2],
            "wavelength_um": 0.5,
            "n0": 1.0,
            "NA": 0.65,
            "n_angles": 4,
            "ri_contrast_scale": 0.01,
            "tiff_scale": 1.0,
        }
        with open(tmp_path / "meta_data.json", "w") as f:
            json.dump(meta, f)

        result = load_observation(str(tmp_path))
        assert result.shape == (nz, ny, nx)

    def test_scaling_correctness(self, tmp_path):
        """Verify Δn = raw * (tiff_scale / 65535) * ri_contrast_scale."""
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not installed")

        nz, ny, nx = 2, 4, 4
        raw_val = 65535  # max uint16
        raw_data = np.full((nz, ny, nx), raw_val, dtype=np.uint16)
        tifffile.imwrite(str(tmp_path / "sample.tiff"), raw_data)

        tiff_scale = 2.0
        ri_scale = 0.01
        meta = {
            "volume_shape": [nz, ny, nx],
            "res_um": [0.1, 0.1, 0.2],
            "wavelength_um": 0.5,
            "n0": 1.0,
            "NA": 0.65,
            "n_angles": 4,
            "ri_contrast_scale": ri_scale,
            "tiff_scale": tiff_scale,
        }
        with open(tmp_path / "meta_data.json", "w") as f:
            json.dump(meta, f)

        result = load_observation(str(tmp_path))
        expected = raw_val * (tiff_scale / 65535.0) * ri_scale
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ── prepare_data ─────────────────────────────────────────────────────────

class TestPrepareData:
    def test_returns_tuple_of_two(self, tmp_path):
        """prepare_data should return (phantom_dn, metadata)."""
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not installed")

        nz, ny, nx = 2, 4, 4
        raw_data = np.zeros((nz, ny, nx), dtype=np.uint16)
        tifffile.imwrite(str(tmp_path / "sample.tiff"), raw_data)

        meta = {
            "volume_shape": [nz, ny, nx],
            "res_um": [0.1, 0.1, 0.2],
            "wavelength_um": 0.5,
            "n0": 1.0,
            "NA": 0.65,
            "n_angles": 4,
            "ri_contrast_scale": 0.01,
            "tiff_scale": 1.0,
        }
        with open(tmp_path / "meta_data.json", "w") as f:
            json.dump(meta, f)

        result = prepare_data(str(tmp_path))
        assert isinstance(result, tuple)
        assert len(result) == 2

        phantom_dn, metadata = result
        assert isinstance(phantom_dn, np.ndarray)
        assert isinstance(metadata, dict)
        assert phantom_dn.shape == (nz, ny, nx)
