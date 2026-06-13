"""Unit tests for src/preprocessing.py."""

import numpy as np
import torch

from src.preprocessing import make_migration_velocity

FIXTURE = np.load("evaluation/fixtures/preprocessing.npz")


class TestMakeMigrationVelocity:
    def test_output_shape(self):
        v = torch.from_numpy(FIXTURE["input_v_patch"])
        v_mig = make_migration_velocity(v, sigma=float(FIXTURE["param_sigma"]))
        assert v_mig.shape == v.shape

    def test_output_dtype(self):
        v = torch.from_numpy(FIXTURE["input_v_patch"])
        v_mig = make_migration_velocity(v, sigma=float(FIXTURE["param_sigma"]))
        assert v_mig.dtype == torch.float32

    def test_output_values(self):
        v = torch.from_numpy(FIXTURE["input_v_patch"])
        v_mig = make_migration_velocity(v, sigma=float(FIXTURE["param_sigma"]))
        expected = torch.from_numpy(FIXTURE["output_v_mig_patch"])
        torch.testing.assert_close(v_mig, expected, rtol=1e-10, atol=0)

    def test_smoothing_reduces_variance(self):
        v = torch.from_numpy(FIXTURE["input_v_patch"])
        v_mig = make_migration_velocity(v, sigma=float(FIXTURE["param_sigma"]))
        assert v_mig.var() < v.var()

    def test_positive_velocities(self):
        v = torch.from_numpy(FIXTURE["input_v_patch"])
        v_mig = make_migration_velocity(v, sigma=float(FIXTURE["param_sigma"]))
        assert (v_mig > 0).all()
