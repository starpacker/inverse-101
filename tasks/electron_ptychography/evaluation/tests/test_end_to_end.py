"""End-to-end tests: verify reference outputs exist and have expected properties."""

import os
import json
import numpy as np
import pytest

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
REF_DIR = os.path.join(TASK_DIR, "evaluation", "reference_outputs")
METRICS_PATH = os.path.join(TASK_DIR, "evaluation", "metrics.json")


class TestReferenceOutputs:
    def test_ptycho_phase_exists(self):
        assert os.path.exists(os.path.join(REF_DIR, "ptycho_phase.npy"))

    def test_ptycho_complex_exists(self):
        assert os.path.exists(os.path.join(REF_DIR, "ptycho_complex.npy"))

    def test_dpc_phase_exists(self):
        assert os.path.exists(os.path.join(REF_DIR, "dpc_phase.npy"))

    def test_parallax_phase_exists(self):
        assert os.path.exists(os.path.join(REF_DIR, "parallax_phase.npy"))

    def test_probe_recon_exists(self):
        assert os.path.exists(os.path.join(REF_DIR, "probe_recon.npy"))


class TestPtychoPhase:
    @pytest.fixture
    def phase(self):
        return np.load(os.path.join(REF_DIR, "ptycho_phase.npy"))

    def test_shape(self, phase):
        # Should be a 2D array (cropped object phase)
        assert phase.ndim == 2

    def test_positive_phase_dominant(self, phase):
        # Gold nanoparticles should have predominantly positive phase
        # (positive electrostatic potential)
        assert phase.max() > 0

    def test_phase_range_reasonable(self, phase):
        # Phase should be within [-pi, pi]
        assert phase.min() > -np.pi
        assert phase.max() < np.pi


class TestDpcPhase:
    @pytest.fixture
    def phase(self):
        return np.load(os.path.join(REF_DIR, "dpc_phase.npy"))

    def test_shape(self, phase):
        assert phase.shape == (48, 48)


class TestMetrics:
    def test_metrics_file_exists(self):
        assert os.path.exists(METRICS_PATH)

    def test_metrics_schema(self):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        assert "baseline" in metrics
        assert "ncc_boundary" in metrics
        assert "nrmse_boundary" in metrics
        assert metrics["ncc_boundary"] > 0
        assert metrics["nrmse_boundary"] > 0
