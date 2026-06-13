"""Unit tests for src/preprocessing.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

from src.preprocessing import (
    default_covariances,
    load_ground_truth,
    load_metadata,
    load_observation,
    select_sample,
)

DATA_DIR = TASK_DIR / "data"
EXPECTED_OBS_KEYS = {"obs_history", "max_val", "min_val", "lat_weight_matrix"}
EXPECTED_GT_KEYS = {"state"}


class TestLoadObservation(unittest.TestCase):
    def test_keys(self):
        obs = load_observation(DATA_DIR)
        self.assertEqual(set(obs.keys()), EXPECTED_OBS_KEYS)

    def test_shapes(self):
        obs = load_observation(DATA_DIR)
        self.assertEqual(obs["obs_history"].shape, (1, 5, 50, 64, 32))
        self.assertEqual(obs["max_val"].shape, (1, 5))
        self.assertEqual(obs["min_val"].shape, (1, 5))
        self.assertEqual(obs["lat_weight_matrix"].shape, (1, 5, 64, 32))

    def test_dtypes(self):
        obs = load_observation(DATA_DIR)
        for k, v in obs.items():
            self.assertEqual(v.dtype, np.float32, msg=f"key={k}")

    def test_max_min_ordering(self):
        obs = load_observation(DATA_DIR)
        self.assertTrue(np.all(obs["max_val"] > obs["min_val"]))


class TestLoadGroundTruth(unittest.TestCase):
    def test_keys_and_shape(self):
        gt = load_ground_truth(DATA_DIR)
        self.assertEqual(set(gt.keys()), EXPECTED_GT_KEYS)
        self.assertEqual(gt["state"].shape, (1, 5, 5, 64, 32))
        self.assertEqual(gt["state"].dtype, np.float32)

    def test_normalised_range(self):
        gt = load_ground_truth(DATA_DIR)
        # State is normalised to [0,1] per channel before bundling.
        self.assertGreaterEqual(float(gt["state"].min()), 0.0)
        self.assertLessEqual(float(gt["state"].max()), 1.0)


class TestLoadMetadata(unittest.TestCase):
    def test_required_fields(self):
        meta = load_metadata(DATA_DIR)
        self.assertEqual(meta["n_channels"], 5)
        self.assertEqual(len(meta["channels"]), 5)
        self.assertEqual(meta["channels"][0], "geopotential")
        self.assertEqual(meta["grid_height"], 64)
        self.assertEqual(meta["grid_width"], 32)
        self.assertEqual(meta["history_len"], 10)


class TestSelectSample(unittest.TestCase):
    def test_strips_batch_dim(self):
        obs = load_observation(DATA_DIR)
        single = select_sample(obs, 0)
        self.assertEqual(single["obs_history"].shape, (5, 50, 64, 32))
        self.assertEqual(single["max_val"].shape, (5,))
        self.assertEqual(single["lat_weight_matrix"].shape, (5, 64, 32))


class TestDefaultCovariances(unittest.TestCase):
    def test_shapes_and_values(self):
        B, R, Q = default_covariances(512)
        for M in (B, R, Q):
            self.assertEqual(M.shape, (512, 512))
        # B and Q are 0.1 * I; R is identity.
        self.assertAlmostEqual(float(B[0, 0]), 0.1)
        self.assertAlmostEqual(float(Q[0, 0]), 0.1)
        self.assertAlmostEqual(float(R[0, 0]), 1.0)
        self.assertAlmostEqual(float(B[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
