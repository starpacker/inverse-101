"""Unit tests for physics_model module."""
import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import prepare_data
from src.physics_model import (
    PyEITMesh, PyEITProtocol, EITForwardModel, sim2pts,
    create_protocol, set_perm, PyEITAnomaly_Circle,
)

DATA_DIR = os.path.join(TASK_DIR, "data")
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "physics_model")


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


class TestForwardSolve(unittest.TestCase):
    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.obs = obs
        self.mesh, self.proto, self.model = _build_model(obs, "bp")

    def test_solve_shape(self):
        v = self.model.solve_eit()
        self.assertEqual(v.ndim, 1)
        self.assertGreater(len(v), 0)

    def test_solve_reproducible(self):
        v1 = self.model.solve_eit()
        v2 = self.model.solve_eit()
        np.testing.assert_allclose(v1, v2, rtol=1e-12)

    def test_solve_matches_fixture(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "output_forward_solve_bp.npz"))
        v0 = self.model.solve_eit()
        np.testing.assert_allclose(np.real(v0), fix["v0"], rtol=1e-10)


class TestJacobian(unittest.TestCase):
    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.mesh, _, self.model = _build_model(obs, "bp")

    def test_jac_shape(self):
        jac, v0 = self.model.compute_jac()
        n_meas = len(self.obs["bp_v0"]) if hasattr(self, 'obs') else jac.shape[0]
        self.assertEqual(jac.shape[1], self.mesh.n_elems)
        self.assertEqual(v0.shape[0], jac.shape[0])

    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.obs = obs
        self.mesh, _, self.model = _build_model(obs, "bp")

    def test_jac_matches_fixture(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "output_jacobian_bp.npz"))
        jac, v0 = self.model.compute_jac()
        np.testing.assert_array_equal(np.array(jac.shape), fix["jac_shape"])


class TestSim2Pts(unittest.TestCase):
    def setUp(self):
        obs, _ = prepare_data(DATA_DIR)
        self.mesh, _, _ = _build_model(obs, "bp")

    def test_output_shape(self):
        perm_elem = np.ones(self.mesh.n_elems)
        perm_nodes = sim2pts(self.mesh.node, self.mesh.element, perm_elem)
        self.assertEqual(perm_nodes.shape[0], self.mesh.n_nodes)

    def test_matches_fixture(self):
        inp = np.load(os.path.join(FIXTURE_DIR, "input_sim2pts_bp.npz"))
        out = np.load(os.path.join(FIXTURE_DIR, "output_sim2pts_bp.npz"))
        result = sim2pts(self.mesh.node, self.mesh.element, inp["perm_elem"])
        np.testing.assert_allclose(result, out["perm_nodes"], rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
