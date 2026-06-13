"""Integration test: check that reference outputs meet evaluation targets."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TASK_DIR = Path(__file__).parents[2]
REFS = TASK_DIR / "evaluation" / "reference_outputs"

pytestmark = pytest.mark.skipif(
    not (REFS / "metrics.json").exists(),
    reason="Reference outputs not yet generated",
)


class TestReferenceOutputs:
    def test_metrics_json_exists(self):
        assert (REFS / "metrics.json").exists()

    def test_velocity_rel_l2(self):
        with open(REFS / "metrics.json") as f:
            m = json.load(f)
        assert m["velocity_rel_l2"] <= 0.10, (
            f"Velocity rel L2 {m['velocity_rel_l2']:.4f} exceeds 10% target"
        )

    def test_data_rel_l2(self):
        with open(REFS / "metrics.json") as f:
            m = json.load(f)
        assert m["data_rel_l2_mean"] <= 0.02, (
            f"Mean data rel L2 {m['data_rel_l2_mean']:.4f} exceeds 2% target"
        )

    def test_v_inv_shape(self):
        v_inv = np.load(REFS / "v_inv.npy")
        data = np.load(TASK_DIR / "data" / "raw_data.npz")
        assert v_inv.shape == data["v_true"].shape

    def test_losses_monotone_trend(self):
        """Loss should decrease over training (not necessarily monotone)."""
        losses = np.load(REFS / "losses.npy")
        first_quarter = losses[:len(losses)//4].mean()
        last_quarter = losses[-len(losses)//4:].mean()
        assert last_quarter < first_quarter, "Loss did not decrease during training"
