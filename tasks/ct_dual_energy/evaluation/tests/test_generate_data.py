"""Tests for generate_data.py."""

import os
import sys
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.generate_data import create_phantom, generate_synthetic_data

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")


class TestCreatePhantom:
    """Tests for create_phantom."""

    def test_shape(self):
        tissue, bone = create_phantom(128)
        assert tissue.shape == (128, 128)
        assert bone.shape == (128, 128)

    def test_non_negative(self):
        tissue, bone = create_phantom(128)
        assert np.all(tissue >= 0)
        assert np.all(bone >= 0)

    def test_tissue_values(self):
        tissue, bone = create_phantom(128)
        # Tissue should be 0 or positive (up to ~1.0)
        unique_approx = np.unique(np.round(tissue, 1))
        assert 0.0 in unique_approx
        assert 1.0 in unique_approx

    def test_bone_present(self):
        tissue, bone = create_phantom(128)
        assert bone.max() > 1.0  # cortical bone ~1.5

    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "generate_data_phantom.npz"))
        tissue, bone = create_phantom(128)
        np.testing.assert_allclose(tissue, fix["output_tissue"], rtol=1e-10)
        np.testing.assert_allclose(bone, fix["output_bone"], rtol=1e-10)

    def test_no_overlap_at_full_density(self):
        """Bone regions with density 1.5 should have tissue = 0."""
        tissue, bone = create_phantom(128)
        bone_mask = bone >= 1.5
        assert np.all(tissue[bone_mask] == 0.0)


class TestGenerateSyntheticData:
    """Tests for generate_synthetic_data."""

    def test_output_keys(self):
        data = generate_synthetic_data(size=32, n_angles=10, seed=42)
        expected_keys = {
            "sinogram_low", "sinogram_high",
            "sinogram_low_clean", "sinogram_high_clean",
            "tissue_map", "bone_map",
            "tissue_sinogram", "bone_sinogram",
            "theta", "energies", "spectra", "mus",
        }
        assert expected_keys <= set(data.keys())

    def test_sinogram_shapes(self):
        data = generate_synthetic_data(size=32, n_angles=10, seed=42)
        nBins = data["sinogram_low"].shape[0]
        assert data["sinogram_low"].shape == (nBins, 10)
        assert data["sinogram_high"].shape == (nBins, 10)

    def test_poisson_noise(self):
        """Noisy sinograms should be integer-valued (Poisson samples)."""
        data = generate_synthetic_data(size=32, n_angles=10, seed=42)
        # Poisson output cast to float64 but should be integer-valued
        low = data["sinogram_low"]
        assert np.allclose(low, np.round(low))

    def test_deterministic(self):
        data1 = generate_synthetic_data(size=32, n_angles=10, seed=99)
        data2 = generate_synthetic_data(size=32, n_angles=10, seed=99)
        np.testing.assert_array_equal(data1["sinogram_low"], data2["sinogram_low"])
