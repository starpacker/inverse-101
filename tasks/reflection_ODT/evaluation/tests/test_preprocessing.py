"""Tests for the preprocessing module (phantom generation and data loading)."""

import os
import sys
import json
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing import load_metadata, generate_phantom, prepare_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_metadata():
    return {
        "volume_shape": [2, 32, 32],
        "n0": 1.5,
        "NA_obj": 0.55,
        "wavelength_um": 0.532,
        "res_um": [0.1, 0.1, 0.5],
        "ri_contrast": 0.02,
        "illumination_rings": [
            {"NA": 0.3, "n_angles": 2, "type": "BF"},
        ],
        "dz_layer": 0.5,
        "dz_gap": 10.0,
    }


def _write_metadata_to_dir(metadata, tmpdir):
    """Write metadata to a temp directory as meta_data.json."""
    path = os.path.join(tmpdir, "meta_data.json")
    with open(path, "w") as f:
        json.dump(metadata, f)
    return tmpdir


# ---------------------------------------------------------------------------
# load_metadata tests
# ---------------------------------------------------------------------------


class TestLoadMetadata:
    def test_loads_valid_json(self):
        meta = _small_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metadata_to_dir(meta, tmpdir)
            loaded = load_metadata(tmpdir)
        assert loaded["n0"] == meta["n0"]
        assert loaded["volume_shape"] == meta["volume_shape"]

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_metadata(tmpdir)


# ---------------------------------------------------------------------------
# generate_phantom tests
# ---------------------------------------------------------------------------


class TestGeneratePhantom:
    def test_output_shape(self):
        meta = _small_metadata()
        phantom = generate_phantom(meta)
        assert phantom.shape == tuple(meta["volume_shape"])

    def test_output_dtype(self):
        meta = _small_metadata()
        phantom = generate_phantom(meta)
        assert phantom.dtype == np.float64

    def test_value_range(self):
        """Phantom values should be between 0 and ri_contrast (or ri_contrast and 0
        when ri_contrast is negative)."""
        meta = _small_metadata()
        phantom = generate_phantom(meta)
        dn = meta["ri_contrast"]
        lo, hi = min(0, dn), max(0, dn)
        assert phantom.min() >= lo - 1e-12
        assert phantom.max() <= hi + 1e-12

    def test_negative_ri_contrast(self):
        """Phantom with negative ri_contrast should have non-positive values."""
        meta = _small_metadata()
        meta["ri_contrast"] = -0.07
        phantom = generate_phantom(meta)
        assert phantom.max() <= 1e-12
        assert phantom.min() >= -0.07 - 1e-12

    def test_background_is_zero(self):
        """Significant fraction of voxels should be background (zero)."""
        meta = _small_metadata()
        phantom = generate_phantom(meta)
        zero_fraction = np.sum(phantom == 0.0) / phantom.size
        assert zero_fraction > 0.1  # at least 10% background

    def test_four_layer_phantom(self):
        """A 4-layer phantom should have patterned content in each layer."""
        meta = _small_metadata()
        meta["volume_shape"] = [4, 64, 64]
        phantom = generate_phantom(meta)
        assert phantom.shape[0] == 4
        for iz in range(4):
            assert np.any(phantom[iz] != 0), f"Layer {iz} is entirely zero"

    def test_two_layer_phantom_partial(self):
        """With only 2 layers, layers 0 and 1 should still have patterns."""
        meta = _small_metadata()
        meta["volume_shape"] = [2, 32, 32]
        phantom = generate_phantom(meta)
        for iz in range(2):
            assert np.any(phantom[iz] != 0), f"Layer {iz} is entirely zero"


# ---------------------------------------------------------------------------
# prepare_data tests
# ---------------------------------------------------------------------------


class TestPrepareData:
    def test_returns_tuple(self):
        meta = _small_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metadata_to_dir(meta, tmpdir)
            result = prepare_data(tmpdir)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_phantom_shape_matches(self):
        meta = _small_metadata()
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metadata_to_dir(meta, tmpdir)
            phantom, loaded_meta = prepare_data(tmpdir)
        assert phantom.shape == tuple(meta["volume_shape"])
        assert loaded_meta["n0"] == meta["n0"]
