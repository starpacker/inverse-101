"""Tests for preprocessing module using external fixtures."""

import os
import sys
import pickle
import numpy as np
import torch
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")
DATA_DIR = os.path.join(TASK_DIR, "data")


def load_fixture(name):
    with open(os.path.join(FIXTURE_DIR, name), "rb") as f:
        return pickle.load(f)


from src.preprocessing import (
    load_metadata,
    load_observations,
    build_restriction_operator,
    create_dobs_masks,
    create_initial_slowness,
)


class TestLoadMetadata:
    def test_keys_present(self):
        meta = load_metadata(DATA_DIR)
        assert meta["nx"] == 480
        assert meta["ny"] == 480
        assert len(meta["frequencies_MHz"]) == 20

    def test_frequencies(self):
        meta = load_metadata(DATA_DIR)
        assert meta["frequencies_MHz"][0] == 0.3
        assert meta["frequencies_MHz"][-1] == 1.25


class TestLoadObservations:
    def test_receiver_shapes(self):
        obs = load_observations(DATA_DIR, freq=0.3)
        assert obs["receiver_ix"].shape == (256,)
        assert obs["receiver_iy"].shape == (256,)

    def test_dobs_shape(self):
        obs = load_observations(DATA_DIR, freq=0.3)
        assert obs["dobs"].shape == (256, 256)
        assert obs["dobs"].dtype == torch.complex64

    def test_batch_first_storage(self):
        """Verify raw npz stores data with batch dimension."""
        raw = np.load(os.path.join(DATA_DIR, "raw_data.npz"))
        assert raw["receiver_ix"].shape == (1, 256)
        assert raw["dobs_0.3"].shape == (1, 256, 256)


class TestBuildRestrictionOperator:
    def test_against_fixture(self):
        inp = load_fixture("input_build_restriction_operator.pkl")
        expected = load_fixture("output_build_restriction_operator.pkl")

        R = build_restriction_operator(inp["ix"], inp["iy"], inp["nx"], inp["ny"])
        R_dense = R.to_dense().cpu().numpy()

        assert R.shape == tuple(expected["shape"])
        np.testing.assert_allclose(R_dense, expected["R_dense"], rtol=1e-10)

    def test_row_sum_is_one(self):
        inp = load_fixture("input_build_restriction_operator.pkl")
        R = build_restriction_operator(inp["ix"], inp["iy"], inp["nx"], inp["ny"])
        R_dense = R.to_dense()
        assert torch.all(R_dense.sum(dim=1) == 1.0)


class TestCreateDobsMasks:
    def test_against_fixture(self):
        inp = load_fixture("input_create_dobs_masks.pkl")
        expected = load_fixture("output_create_dobs_masks.pkl")

        dobs = torch.tensor(inp["dobs"], dtype=torch.complex64).cuda()
        dobs_masked, mask_esi, mask_misfit = create_dobs_masks(
            dobs, inp["ix"], inp["iy"], inp["dh"], inp["mute_dist"]
        )

        np.testing.assert_allclose(
            dobs_masked.cpu().numpy(), expected["dobs_masked"], rtol=1e-10
        )
        np.testing.assert_array_equal(
            mask_esi.cpu().numpy(), expected["mask_esi"]
        )
        np.testing.assert_array_equal(
            mask_misfit.cpu().numpy(), expected["mask_misfit"]
        )

    def test_diagonal_masked(self):
        """Self-pairs (source == receiver) should be masked."""
        inp = load_fixture("input_create_dobs_masks.pkl")
        dobs = torch.tensor(inp["dobs"], dtype=torch.complex64).cuda()
        _, mask_esi, _ = create_dobs_masks(
            dobs, inp["ix"], inp["iy"], inp["dh"], inp["mute_dist"]
        )
        for i in range(len(inp["ix"])):
            assert mask_esi[i, i].item() == False


class TestInitialSlowness:
    def test_shape_and_value(self):
        s = create_initial_slowness(480, 480, 1480.0)
        assert s.shape == (480, 480)
        assert torch.allclose(s, torch.tensor(1.0 / 1480.0))
