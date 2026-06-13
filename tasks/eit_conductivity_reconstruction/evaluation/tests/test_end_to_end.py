"""End-to-end integration tests for the EIT reconstruction task."""
import os
import sys
import json
import unittest
import numpy as np

# Add task root to path
TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import load_observation, load_metadata, prepare_data
from src.physics_model import (
    PyEITMesh, PyEITProtocol, EITForwardModel,
    create_protocol, set_perm, PyEITAnomaly_Circle, sim2pts,
)
from src.solvers import (
    BPReconstructor, GREITReconstructor,
    JACDynamicReconstructor,
)
from src.visualization import compute_metrics

DATA_DIR = os.path.join(TASK_DIR, "data")
REF_DIR = os.path.join(TASK_DIR, "evaluation", "reference_outputs")


def _build_model(obs, prefix):
    """Reconstruct model from stored arrays."""
    mesh = PyEITMesh(
        node=obs[f"{prefix}_node"],
        element=obs[f"{prefix}_element"],
        perm=obs[f"{prefix}_perm_background"],
        el_pos=obs[f"{prefix}_el_pos"],
        ref_node=int(obs[f"{prefix}_ref_node"]),
    )
    protocol = PyEITProtocol(
        ex_mat=obs[f"{prefix}_ex_mat"],
        meas_mat=obs[f"{prefix}_meas_mat"],
        keep_ba=obs[f"{prefix}_keep_ba"],
    )
    return mesh, protocol, EITForwardModel(mesh, protocol)


class TestDataLoading(unittest.TestCase):
    """Test that data loading works correctly."""

    def test_load_observation(self):
        obs = load_observation(DATA_DIR)
        self.assertIn("bp_v0", obs)
        self.assertIn("bp_v1", obs)
        self.assertIn("bp_node", obs)

    def test_load_metadata(self):
        meta = load_metadata(DATA_DIR)
        self.assertIn("experiments", meta)
        self.assertIn("bp", meta["experiments"])
        self.assertEqual(meta["experiments"]["bp"]["n_el"], 16)

    def test_prepare_data(self):
        obs, meta = prepare_data(DATA_DIR)
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(meta, dict)


class TestForwardModel(unittest.TestCase):
    """Test the forward model produces consistent results."""

    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.obs = obs
        self.mesh, self.proto, self.model = _build_model(obs, "bp")

    def test_solve_eit_shape(self):
        v = self.model.solve_eit()
        self.assertEqual(v.ndim, 1)
        self.assertGreater(len(v), 0)

    def test_solve_eit_reproducible(self):
        v1 = self.model.solve_eit()
        v2 = self.model.solve_eit()
        np.testing.assert_allclose(v1, v2, rtol=1e-12)

    def test_solve_eit_matches_stored(self):
        """Verify forward solver reproduces the stored voltages."""
        v0 = self.model.solve_eit()
        np.testing.assert_allclose(np.real(v0), self.obs["bp_v0"], rtol=1e-10)

    def test_compute_jac_shape(self):
        jac, v0 = self.model.compute_jac()
        n_meas = len(self.obs["bp_v0"])
        n_elem = self.mesh.n_elems
        self.assertEqual(jac.shape, (n_meas, n_elem))
        self.assertEqual(v0.shape, (n_meas,))


class TestBPReconstruction(unittest.TestCase):
    """Test BP reconstruction produces expected results."""

    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.mesh, _, self.model = _build_model(obs, "bp")
        self.v0 = obs["bp_v0"]
        self.v1 = obs["bp_v1"]

    def test_reconstruct_shape(self):
        bp = BPReconstructor(weight="none")
        ds = bp.reconstruct(self.model, self.v1, self.v0, normalize=True)
        self.assertEqual(ds.shape[0], self.mesh.n_nodes)

    def test_reconstruct_matches_reference(self):
        bp = BPReconstructor(weight="none")
        ds = bp.reconstruct(self.model, self.v1, self.v0, normalize=True)
        ds = np.real(ds) * 192.0
        ref = np.load(os.path.join(REF_DIR, "reconstruction_bp.npy"))
        np.testing.assert_allclose(ds, ref, rtol=1e-10)


class TestGREITReconstruction(unittest.TestCase):
    """Test GREIT reconstruction."""

    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.mesh, _, self.model = _build_model(obs, "greit")
        self.v0 = obs["greit_v0"]
        self.v1 = obs["greit_v1"]

    def test_reconstruct_returns_grid(self):
        greit = GREITReconstructor(p=0.50, lamb=0.01, n=32, jac_normalized=True)
        xg, yg, ds = greit.reconstruct(self.model, self.v1, self.v0, normalize=True)
        self.assertEqual(ds.shape, (32, 32))
        self.assertEqual(xg.shape, (32, 32))

    def test_reconstruct_matches_reference(self):
        greit = GREITReconstructor(p=0.50, lamb=0.01, n=32, s=20.0, ratio=0.1, jac_normalized=True)
        xg, yg, ds = greit.reconstruct(self.model, self.v1, self.v0, normalize=True)
        ref = np.load(os.path.join(REF_DIR, "reconstruction_greit.npz"))
        np.testing.assert_allclose(ds, ref["ds"], rtol=1e-10)


class TestJACDynamicReconstruction(unittest.TestCase):
    """Test JAC dynamic reconstruction."""

    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.mesh, _, self.model = _build_model(obs, "jac_dyn")
        self.v0 = obs["jac_dyn_v0"]
        self.v1 = obs["jac_dyn_v1"]

    def test_reconstruct_shape(self):
        jac = JACDynamicReconstructor(p=0.5, lamb=0.01, method="kotre", jac_normalized=True)
        ds = jac.reconstruct(self.model, self.v1, self.v0, normalize=True)
        self.assertEqual(ds.shape[0], self.mesh.n_elems)

    def test_reconstruct_matches_reference(self):
        jac = JACDynamicReconstructor(p=0.5, lamb=0.01, method="kotre", jac_normalized=True)
        ds = jac.reconstruct(self.model, self.v1, self.v0, normalize=True)
        ref = np.load(os.path.join(REF_DIR, "reconstruction_jac_dynamic.npy"))
        np.testing.assert_allclose(np.real(ds), ref, rtol=1e-9)


class TestMetrics(unittest.TestCase):
    """Test that metrics match reference values."""

    def test_metrics_exist(self):
        metrics_path = os.path.join(REF_DIR, "metrics.json")
        self.assertTrue(os.path.exists(metrics_path))
        with open(metrics_path) as f:
            metrics = json.load(f)
        self.assertIn("bp", metrics)
        self.assertIn("jac_dynamic", metrics)
        self.assertNotIn("jac_static", metrics)

    def test_compute_metrics_basic(self):
        gt = np.array([1.0, 2.0, 3.0, 4.0])
        recon = np.array([1.1, 1.9, 3.1, 3.9])
        m = compute_metrics(recon, gt)
        self.assertIn("nrmse", m)
        self.assertIn("ncc", m)
        self.assertGreater(m["ncc"], 0.9)  # Should be high for similar arrays


if __name__ == "__main__":
    unittest.main()
