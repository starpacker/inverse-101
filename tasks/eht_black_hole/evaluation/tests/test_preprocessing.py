"""
Tests for preprocessing.py — Decoupled per-function tests
==========================================================

Each test loads its own fixture from evaluation/fixtures/preprocessing/
and verifies function output against pre-computed reference.

Tested functions:
  - load_observation(data_dir) → dict
  - load_metadata(data_dir)    → dict
  - prepare_data(data_dir)     → (vis, uv, meta)
"""

import os
import sys
import json
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)

DATA_DIR = os.path.join(TASK_DIR, "data")
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "preprocessing")


# ═══════════════════════════════════════════════════════════════════════════
# load_observation
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadObservation(unittest.TestCase):
    """
    Fixture: load_observation.npz
      output_vis_noisy  : (M,) complex128
      output_uv_coords  : (M, 2) float64
    """

    def setUp(self):
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "load_observation.npz"))
        from src.preprocessing import load_observation
        self.result = load_observation(DATA_DIR)

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_required_keys(self):
        self.assertIn("vis_noisy", self.result)
        self.assertIn("uv_coords", self.result)

    def test_vis_noisy_shape(self):
        expected_shape = self.fixture["output_vis_noisy"].shape
        self.assertEqual(self.result["vis_noisy"].shape, expected_shape)

    def test_vis_noisy_dtype(self):
        self.assertTrue(np.iscomplexobj(self.result["vis_noisy"]))

    def test_vis_noisy_values(self):
        np.testing.assert_array_equal(
            self.result["vis_noisy"], self.fixture["output_vis_noisy"]
        )

    def test_uv_coords_shape(self):
        expected_shape = self.fixture["output_uv_coords"].shape
        self.assertEqual(self.result["uv_coords"].shape, expected_shape)

    def test_uv_coords_dtype(self):
        self.assertTrue(np.isrealobj(self.result["uv_coords"]))

    def test_uv_coords_values(self):
        np.testing.assert_array_equal(
            self.result["uv_coords"], self.fixture["output_uv_coords"]
        )


# ═══════════════════════════════════════════════════════════════════════════
# load_metadata
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadMetadata(unittest.TestCase):
    """
    Fixture: load_metadata.json
      Full metadata dict with all expected keys and values.
    """

    def setUp(self):
        with open(os.path.join(FIXTURE_DIR, "load_metadata.json")) as f:
            self.expected = json.load(f)
        from src.preprocessing import load_metadata
        self.result = load_metadata(DATA_DIR)

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_required_keys(self):
        for key in self.expected:
            self.assertIn(key, self.result, f"Missing key: {key}")

    def test_values_match(self):
        for key, expected_val in self.expected.items():
            self.assertAlmostEqual(
                self.result[key], expected_val, places=10,
                msg=f"Mismatch for key '{key}'"
            )


# ═══════════════════════════════════════════════════════════════════════════
# prepare_data
# ═══════════════════════════════════════════════════════════════════════════

class TestPrepareData(unittest.TestCase):
    """
    Fixture: prepare_data.npz + prepare_data_meta.json
      output_vis_noisy  : (M,) complex128
      output_uv_coords  : (M, 2) float64
      + metadata dict
    """

    def setUp(self):
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "prepare_data.npz"))
        with open(os.path.join(FIXTURE_DIR, "prepare_data_meta.json")) as f:
            self.expected_meta = json.load(f)
        from src.preprocessing import prepare_data
        self.vis, self.uv, self.meta = prepare_data(DATA_DIR)

    def test_returns_three_elements(self):
        self.assertIsInstance(self.vis, np.ndarray)
        self.assertIsInstance(self.uv, np.ndarray)
        self.assertIsInstance(self.meta, dict)

    def test_vis_values(self):
        np.testing.assert_array_equal(self.vis, self.fixture["output_vis_noisy"])

    def test_uv_values(self):
        np.testing.assert_array_equal(self.uv, self.fixture["output_uv_coords"])

    def test_metadata_values(self):
        for key, expected_val in self.expected_meta.items():
            self.assertAlmostEqual(
                self.meta[key], expected_val, places=10,
                msg=f"Mismatch for key '{key}'"
            )


if __name__ == "__main__":
    unittest.main()
