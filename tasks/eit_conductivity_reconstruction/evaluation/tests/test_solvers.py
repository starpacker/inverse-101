"""Unit tests for solvers module."""
import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import prepare_data
from src.physics_model import PyEITMesh, PyEITProtocol, EITForwardModel
from src.solvers import BPReconstructor, GREITReconstructor, JACDynamicReconstructor

DATA_DIR = os.path.join(TASK_DIR, "data")
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "solvers")


def _build_model(obs, prefix):
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


class TestBPReconstructor(unittest.TestCase):
    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.mesh, _, self.model = _build_model(obs, "bp")
        self.v0 = obs["bp_v0"]
        self.v1 = obs["bp_v1"]

    def test_reconstruct_shape(self):
        bp = BPReconstructor(weight="none")
        ds = bp.reconstruct(self.model, self.v1, self.v0, normalize=True)
        self.assertEqual(ds.shape[0], self.mesh.n_nodes)

    def test_reconstruct_matches_fixture(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "output_bp_reconstruct.npz"))
        bp = BPReconstructor(weight="none")
        ds = bp.reconstruct(self.model, self.v1, self.v0, normalize=True)
        ds_scaled = np.real(ds) * 192.0
        np.testing.assert_allclose(ds_scaled, fix["ds"], rtol=1e-10)


class TestJACDynamicReconstructor(unittest.TestCase):
    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.mesh, _, self.model = _build_model(obs, "jac_dyn")
        self.v0 = obs["jac_dyn_v0"]
        self.v1 = obs["jac_dyn_v1"]

    def test_reconstruct_shape(self):
        jac = JACDynamicReconstructor(p=0.5, lamb=0.01, method="kotre", jac_normalized=True)
        ds = jac.reconstruct(self.model, self.v1, self.v0, normalize=True)
        self.assertEqual(ds.shape[0], self.mesh.n_elems)

    def test_reconstruct_matches_fixture(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "output_jac_dynamic_reconstruct.npz"))
        jac = JACDynamicReconstructor(p=0.5, lamb=0.01, method="kotre", jac_normalized=True)
        ds = jac.reconstruct(self.model, self.v1, self.v0, normalize=True)
        np.testing.assert_allclose(np.real(ds), fix["ds"], rtol=1e-10)


class TestGREITReconstructor(unittest.TestCase):
    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.mesh, _, self.model = _build_model(obs, "greit")
        self.v0 = obs["greit_v0"]
        self.v1 = obs["greit_v1"]

    def test_reconstruct_shape(self):
        greit = GREITReconstructor(p=0.50, lamb=0.01, n=32, jac_normalized=True)
        xg, yg, ds = greit.reconstruct(self.model, self.v1, self.v0, normalize=True)
        self.assertEqual(ds.shape, (32, 32))

    def test_reconstruct_matches_fixture(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "output_greit_reconstruct.npz"))
        greit = GREITReconstructor(p=0.50, lamb=0.01, n=32, s=20.0, ratio=0.1, jac_normalized=True)
        xg, yg, ds = greit.reconstruct(self.model, self.v1, self.v0, normalize=True)
        np.testing.assert_allclose(ds, fix["ds"], rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
