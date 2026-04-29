"""Tests for src/preprocessing.py."""

import os
import sys

import numpy as np
import pytest

# Ensure task root is on the path
TASK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

try:
    import cupy
    cupy.array([1])
    HAS_GPU = True
except Exception:
    HAS_GPU = False

from src.preprocessing import (
    load_raw_data,
    load_metadata,
    shift_scan_positions,
    initialize_psi,
)


DATA_DIR = os.path.join(TASK_DIR, "data")


class TestLoadRawData:
    """Tests for load_raw_data."""

    def test_returns_dict_with_expected_keys(self):
        raw = load_raw_data(DATA_DIR)
        assert isinstance(raw, dict)
        assert "diffraction_patterns" in raw
        assert "scan_positions" in raw
        assert "probe_guess" in raw

    def test_batch_dim_removed(self):
        """After loading, arrays should not have the batch dimension."""
        raw = load_raw_data(DATA_DIR)
        # diffraction_patterns: originally (1, 516, 128, 128) -> (516, 128, 128)
        assert raw["diffraction_patterns"].ndim == 3
        # scan_positions: originally (1, 516, 2) -> (516, 2)
        assert raw["scan_positions"].ndim == 2
        assert raw["scan_positions"].shape[1] == 2

    def test_data_shapes(self):
        raw = load_raw_data(DATA_DIR)
        assert raw["diffraction_patterns"].shape == (516, 128, 128)
        assert raw["scan_positions"].shape == (516, 2)

    def test_dtypes(self):
        raw = load_raw_data(DATA_DIR)
        assert raw["diffraction_patterns"].dtype == np.float32
        assert raw["scan_positions"].dtype == np.float32
        assert np.iscomplexobj(raw["probe_guess"])


class TestLoadMetadata:
    """Tests for load_metadata."""

    def test_returns_dict(self):
        meta = load_metadata(DATA_DIR)
        assert isinstance(meta, dict)

    def test_has_expected_keys(self):
        meta = load_metadata(DATA_DIR)
        assert "n_positions" in meta
        assert "probe_shape" in meta
        assert "diffraction_shape" in meta

    def test_n_positions(self):
        meta = load_metadata(DATA_DIR)
        assert meta["n_positions"] == 516


class TestShiftScanPositions:
    """Tests for shift_scan_positions."""

    def test_minimum_equals_offset(self):
        scan = np.array([[-10.0, 5.0], [20.0, -3.0]], dtype=np.float32)
        shifted = shift_scan_positions(scan, offset=20.0)
        assert np.allclose(np.amin(shifted, axis=0), [20.0, 20.0])

    def test_preserves_relative_positions(self):
        scan = np.array([[0.0, 0.0], [10.0, 5.0], [3.0, 7.0]],
                        dtype=np.float32)
        shifted = shift_scan_positions(scan, offset=20.0)
        # Relative differences should be preserved
        orig_diff = scan[1] - scan[0]
        new_diff = shifted[1] - shifted[0]
        assert np.allclose(orig_diff, new_diff)

    def test_does_not_modify_input(self):
        scan = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        original = scan.copy()
        shift_scan_positions(scan, offset=20.0)
        assert np.array_equal(scan, original)

    def test_default_offset(self):
        scan = np.array([[0.0, 0.0]], dtype=np.float32)
        shifted = shift_scan_positions(scan)
        assert np.allclose(shifted, [[20.0, 20.0]])


class TestInitializePsi:
    """Tests for initialize_psi."""

    def test_output_shape_3d(self):
        scan = np.array([[20.0, 20.0], [100.0, 100.0]], dtype=np.float32)
        probe_shape = (1, 1, 1, 128, 128)
        psi = initialize_psi(scan, probe_shape)
        assert psi.ndim == 3
        assert psi.shape[0] == 1  # single slice

    def test_dtype_complex64(self):
        scan = np.array([[20.0, 20.0]], dtype=np.float32)
        probe_shape = (1, 1, 1, 64, 64)
        psi = initialize_psi(scan, probe_shape)
        assert psi.dtype == np.complex64

    def test_fill_value(self):
        scan = np.array([[20.0, 20.0]], dtype=np.float32)
        probe_shape = (1, 1, 1, 32, 32)
        psi = initialize_psi(scan, probe_shape, fill_value=0.5 + 0j)
        assert np.allclose(psi, 0.5 + 0j)

    def test_psi_larger_than_probe(self):
        scan = np.array([[20.0, 20.0], [50.0, 50.0]], dtype=np.float32)
        probe_shape = (1, 1, 1, 128, 128)
        psi = initialize_psi(scan, probe_shape)
        # psi spatial dims must be larger than probe dims
        assert psi.shape[-2] > probe_shape[-2]
        assert psi.shape[-1] > probe_shape[-1]
