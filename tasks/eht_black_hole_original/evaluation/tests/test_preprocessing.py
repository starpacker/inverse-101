"""Unit tests for preprocessing module."""
import os
import sys
import json
import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, TASK_DIR)

FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "preprocessing")
DATA_DIR = os.path.join(TASK_DIR, "data")


class TestLoadObservation(unittest.TestCase):
    def setUp(self):
        from src.preprocessing import load_observation
        self.result = load_observation(DATA_DIR)
        self.f = np.load(os.path.join(FIXTURE_DIR, "load_observation.npz"))

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_required_keys(self):
        for key in ["vis_corrupted", "vis_true", "uv_coords",
                     "station_ids", "noise_std_per_vis"]:
            self.assertIn(key, self.result)

    def test_vis_corrupted_shape_and_dtype(self):
        v = self.result["vis_corrupted"]
        self.assertEqual(v.shape, (540,))
        self.assertTrue(np.iscomplexobj(v))

    def test_uv_coords_shape(self):
        self.assertEqual(self.result["uv_coords"].shape, (540, 2))

    def test_station_ids_shape_and_dtype(self):
        s = self.result["station_ids"]
        self.assertEqual(s.shape, (540, 2))
        self.assertTrue(np.issubdtype(s.dtype, np.integer))

    def test_values_match_fixture(self):
        assert_array_equal(self.result["vis_corrupted"],
                           self.f["output_vis_corrupted"])
        assert_array_equal(self.result["uv_coords"],
                           self.f["output_uv_coords"])
        assert_array_equal(self.result["station_ids"],
                           self.f["output_station_ids"])


class TestLoadMetadata(unittest.TestCase):
    def setUp(self):
        from src.preprocessing import load_metadata
        self.result = load_metadata(DATA_DIR)
        with open(os.path.join(FIXTURE_DIR, "load_metadata.json")) as f:
            self.expected = json.load(f)

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_required_keys(self):
        for key in ["N", "pixel_size_uas", "pixel_size_rad",
                     "n_baselines", "n_stations"]:
            self.assertIn(key, self.result)

    def test_values_match_fixture(self):
        for key, val in self.expected.items():
            self.assertAlmostEqual(
                self.result[key], val, places=10,
                msg=f"Mismatch for key '{key}'",
            )


class TestFindTriangles(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "find_triangles.npz"))
        from src.preprocessing import find_triangles
        self.tri, self.tri_st = find_triangles(
            self.f["input_station_ids"], int(self.f["input_n_stations"]),
        )

    def test_triangle_count(self):
        # C(4,3) = 4 triangles for a complete 4-station array
        self.assertEqual(len(self.tri), 4)

    def test_triangle_stations_shape(self):
        self.assertEqual(self.tri_st.shape[1], 3)

    def test_values_match_fixture(self):
        assert_array_equal(self.tri, self.f["output_triangles"])
        assert_array_equal(self.tri_st, self.f["output_triangle_stations"])


class TestFindQuadrangles(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "find_quadrangles.npz"))
        from src.preprocessing import find_quadrangles
        self.quad, self.quad_st = find_quadrangles(
            self.f["input_station_ids"], int(self.f["input_n_stations"]),
        )

    def test_quadrangle_count(self):
        # C(4,4) = 1 quadrangle for a complete 4-station array
        self.assertEqual(len(self.quad), 1)

    def test_quadrangle_shape(self):
        self.assertEqual(self.quad.shape[1], 4)

    def test_values_match_fixture(self):
        assert_array_equal(self.quad, self.f["output_quadrangles"])
        assert_array_equal(self.quad_st, self.f["output_quadrangle_stations"])


class TestComputeClosurePhases(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "compute_closure_phases.npz"))
        from src.preprocessing import compute_closure_phases
        self.result = compute_closure_phases(
            self.f["input_vis"], self.f["input_triangles"],
            self.f["input_station_ids"],
        )

    def test_shape(self):
        self.assertEqual(self.result.shape, self.f["output_cphases"].shape)

    def test_range(self):
        self.assertTrue(np.all(self.result >= -np.pi))
        self.assertTrue(np.all(self.result <= np.pi))

    def test_values_match_fixture(self):
        assert_allclose(self.result, self.f["output_cphases"], rtol=1e-10)


class TestComputeLogClosureAmplitudes(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "compute_log_closure_amplitudes.npz"))
        from src.preprocessing import compute_log_closure_amplitudes
        self.result = compute_log_closure_amplitudes(
            self.f["input_vis"], self.f["input_quadrangles"],
        )

    def test_shape(self):
        self.assertEqual(self.result.shape, self.f["output_log_camps"].shape)

    def test_values_match_fixture(self):
        assert_allclose(self.result, self.f["output_log_camps"], rtol=1e-10)


class TestClosurePhaseSigma(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, "closure_phase_sigma.npz"))
        from src.preprocessing import closure_phase_sigma
        self.result = closure_phase_sigma(
            self.f["input_vis"], self.f["input_noise_std_per_vis"],
            self.f["input_triangles"],
        )

    def test_positive(self):
        self.assertTrue(np.all(self.result > 0))

    def test_values_match_fixture(self):
        assert_allclose(self.result, self.f["output_sigma_cp"], rtol=1e-10)


class TestClosureAmplitudeSigma(unittest.TestCase):
    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR,
                                      "closure_amplitude_sigma.npz"))
        from src.preprocessing import closure_amplitude_sigma
        self.result = closure_amplitude_sigma(
            self.f["input_vis"], self.f["input_noise_std_per_vis"],
            self.f["input_quadrangles"],
        )

    def test_positive(self):
        self.assertTrue(np.all(self.result > 0))

    def test_values_match_fixture(self):
        assert_allclose(self.result, self.f["output_sigma_logca"], rtol=1e-10)


class TestPrepareData(unittest.TestCase):
    def setUp(self):
        from src.preprocessing import prepare_data
        self.obs, self.cdata, self.meta = prepare_data(DATA_DIR)
        self.f = np.load(os.path.join(FIXTURE_DIR, "prepare_data.npz"))
        with open(os.path.join(FIXTURE_DIR, "prepare_data_meta.json")) as fj:
            self.expected_meta = json.load(fj)

    def test_closure_data_keys(self):
        for key in ["cphases", "log_camps", "sigma_cp", "sigma_logca",
                     "triangles", "quadrangles"]:
            self.assertIn(key, self.cdata)

    def test_closure_phases_match(self):
        assert_allclose(self.cdata["cphases"], self.f["output_cphases"],
                        rtol=1e-10)

    def test_log_camps_match(self):
        assert_allclose(self.cdata["log_camps"], self.f["output_log_camps"],
                        rtol=1e-10)

    def test_metadata_match(self):
        for key, val in self.expected_meta.items():
            self.assertAlmostEqual(self.meta[key], val, places=10)


if __name__ == "__main__":
    unittest.main()
