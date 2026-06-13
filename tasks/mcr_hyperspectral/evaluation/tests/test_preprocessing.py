"""Tests for preprocessing module."""

import pathlib
import sys

import numpy as np
import pytest

TASK_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

from src.preprocessing import load_observation, load_ground_truth, load_metadata, estimate_initial_spectra

DATA_DIR = TASK_DIR / "data"


class TestLoadObservation:
    def test_keys(self):
        obs = load_observation(DATA_DIR)
        assert "hsi_noisy" in obs
        assert "wn" in obs

    def test_shapes(self):
        obs = load_observation(DATA_DIR)
        assert obs["hsi_noisy"].shape == (5000, 200)
        assert obs["wn"].shape == (200,)

    def test_dtype(self):
        obs = load_observation(DATA_DIR)
        assert obs["hsi_noisy"].dtype == np.float64


class TestLoadGroundTruth:
    def test_keys(self):
        gt = load_ground_truth(DATA_DIR)
        assert "concentrations" in gt
        assert "concentrations_ravel" in gt
        assert "spectra" in gt
        assert "hsi_clean" in gt

    def test_shapes(self):
        gt = load_ground_truth(DATA_DIR)
        assert gt["concentrations"].shape == (50, 100, 3)
        assert gt["concentrations_ravel"].shape == (5000, 3)
        assert gt["spectra"].shape == (3, 200)
        assert gt["hsi_clean"].shape == (5000, 200)

    def test_concentrations_sum_to_one(self):
        gt = load_ground_truth(DATA_DIR)
        totals = gt["concentrations"].sum(axis=-1)
        np.testing.assert_allclose(totals, 1.0, atol=1e-12)


class TestLoadMetadata:
    def test_required_keys(self):
        meta = load_metadata(DATA_DIR)
        for key in ["M", "N", "n_components", "n_freq", "noise_std"]:
            assert key in meta

    def test_values(self):
        meta = load_metadata(DATA_DIR)
        assert meta["M"] == 50
        assert meta["N"] == 100
        assert meta["n_components"] == 3
        assert meta["n_freq"] == 200


class TestEstimateInitialSpectra:
    def test_shape(self):
        obs = load_observation(DATA_DIR)
        initial = estimate_initial_spectra(obs["hsi_noisy"], 3)
        assert initial.shape == (3, 200)

    def test_nonnegative(self):
        obs = load_observation(DATA_DIR)
        initial = estimate_initial_spectra(obs["hsi_noisy"], 3)
        assert np.all(initial >= 0)

    def test_reasonable_scale(self):
        obs = load_observation(DATA_DIR)
        initial = estimate_initial_spectra(obs["hsi_noisy"], 3)
        assert initial.max() > 0
        assert initial.max() <= obs["hsi_noisy"].max() * 2.0
