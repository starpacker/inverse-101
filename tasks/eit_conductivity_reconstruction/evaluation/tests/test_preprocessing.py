"""Unit tests for preprocessing module."""
import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_observation, load_metadata, prepare_data

DATA_DIR = os.path.join(TASK_DIR, "data")
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "preprocessing")


class TestLoadObservation(unittest.TestCase):
    def test_returns_dict(self):
        obs = load_observation(DATA_DIR)
        self.assertIsInstance(obs, dict)

    def test_contains_bp_keys(self):
        obs = load_observation(DATA_DIR)
        self.assertIn("bp_v0", obs)
        self.assertIn("bp_v1", obs)
        self.assertIn("bp_node", obs)

    def test_no_jac_static_keys(self):
        obs = load_observation(DATA_DIR)
        for k in obs:
            self.assertFalse(k.startswith("jac_static"), f"Found stale key: {k}")


class TestLoadMetadata(unittest.TestCase):
    def test_returns_dict(self):
        meta = load_metadata(DATA_DIR)
        self.assertIsInstance(meta, dict)

    def test_has_experiments(self):
        meta = load_metadata(DATA_DIR)
        self.assertIn("experiments", meta)
        self.assertIn("bp", meta["experiments"])
        self.assertIn("greit", meta["experiments"])
        self.assertIn("jac_dynamic", meta["experiments"])

    def test_no_jac_static(self):
        meta = load_metadata(DATA_DIR)
        self.assertNotIn("jac_static", meta["experiments"])

    def test_bp_config(self):
        meta = load_metadata(DATA_DIR)
        self.assertEqual(meta["experiments"]["bp"]["n_el"], 16)


class TestPrepareData(unittest.TestCase):
    def test_returns_tuple(self):
        obs, meta = prepare_data(DATA_DIR)
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(meta, dict)


if __name__ == "__main__":
    unittest.main()
