"""Tests for preprocessing module."""

import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

import torch
from src.preprocessing import (
    load_frame_data,
    _find_triangles,
    _find_quadrangles,
    extract_closure_indices,
    compute_nufft_params,
    estimate_flux,
)


class TestLoadFrameData(unittest.TestCase):
    """Tests for load_frame_data."""

    def setUp(self):
        M = 10
        self.raw_data = {
            "vis_0": np.random.randn(M) + 1j * np.random.randn(M),
            "sigma_0": np.abs(np.random.randn(M)) + 0.01,
            "uv_0": np.random.randn(M, 2),
            "station_ids_0": np.array(
                [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3],
                 [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]], dtype=np.int64
            ),
        }

    def test_keys(self):
        """Returned dict should contain the expected keys."""
        frame = load_frame_data(self.raw_data, 0)
        expected_keys = {"vis", "vis_sigma", "uv_coords", "station_ids"}
        self.assertEqual(set(frame.keys()), expected_keys)

    def test_shapes(self):
        """Returned arrays should have the correct shapes."""
        frame = load_frame_data(self.raw_data, 0)
        M = 10
        self.assertEqual(frame["vis"].shape, (M,))
        self.assertEqual(frame["vis_sigma"].shape, (M,))
        self.assertEqual(frame["uv_coords"].shape, (M, 2))
        self.assertEqual(frame["station_ids"].shape, (M, 2))

    def test_values_match_input(self):
        """Returned arrays should match the raw input arrays."""
        frame = load_frame_data(self.raw_data, 0)
        np.testing.assert_array_equal(frame["vis"], self.raw_data["vis_0"])
        np.testing.assert_array_equal(frame["vis_sigma"], self.raw_data["sigma_0"])


class TestFindTriangles(unittest.TestCase):
    """Tests for _find_triangles."""

    def test_complete_graph_4_stations(self):
        """4 stations fully connected should yield 4 triangles (C(4,3)=4)."""
        # Baselines: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        station_ids = np.array([
            [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
        ], dtype=np.int64)
        triangles = _find_triangles(station_ids)
        self.assertEqual(len(triangles), 4)

    def test_triangle_tuple_format(self):
        """Each triangle should be a 6-tuple: (idx, sign, idx, sign, idx, sign)."""
        station_ids = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        triangles = _find_triangles(station_ids)
        self.assertGreaterEqual(len(triangles), 1)
        tri = triangles[0]
        self.assertEqual(len(tri), 6)
        # Signs should be +1 or -1
        for i in range(3):
            self.assertIn(tri[2 * i + 1], [+1, -1])

    def test_no_triangle_with_two_baselines(self):
        """Two baselines cannot form a triangle."""
        station_ids = np.array([[0, 1], [0, 2]], dtype=np.int64)
        triangles = _find_triangles(station_ids)
        self.assertEqual(len(triangles), 0)


class TestFindQuadrangles(unittest.TestCase):
    """Tests for _find_quadrangles."""

    def test_complete_graph_4_stations(self):
        """4 fully connected stations should yield 3 quadrangles (C(4,4)=1,
        but there are 3 pairings of 4 stations)."""
        station_ids = np.array([
            [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
        ], dtype=np.int64)
        quads = _find_quadrangles(station_ids)
        # For 4 stations with all 6 baselines, there is exactly 1 combination
        # of 4 stations (0,1,2,3), yielding exactly 1 quadrangle.
        self.assertEqual(len(quads), 1)

    def test_quad_tuple_format(self):
        """Each quad should be a 4-tuple of baseline indices."""
        station_ids = np.array([
            [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
        ], dtype=np.int64)
        quads = _find_quadrangles(station_ids)
        self.assertGreaterEqual(len(quads), 1)
        quad = quads[0]
        self.assertEqual(len(quad), 4)
        for idx in quad:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(station_ids))

    def test_no_quad_with_three_stations(self):
        """Three stations cannot form a quadrangle."""
        station_ids = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        quads = _find_quadrangles(station_ids)
        self.assertEqual(len(quads), 0)


class TestExtractClosureIndices(unittest.TestCase):
    """Tests for extract_closure_indices."""

    def _make_frame_data(self, n_stations=5):
        """Build synthetic frame data with a fully connected station set."""
        from itertools import combinations
        pairs = list(combinations(range(n_stations), 2))
        M = len(pairs)
        station_ids = np.array(pairs, dtype=np.int64)
        vis = np.random.randn(M) + 1j * np.random.randn(M)
        vis_sigma = np.abs(np.random.randn(M)) + 0.01
        return {
            "vis": vis,
            "vis_sigma": vis_sigma,
            "station_ids": station_ids,
        }

    def test_output_keys(self):
        """Returned dict should contain all expected keys."""
        frame = self._make_frame_data(5)
        result = extract_closure_indices(frame)
        expected = {
            "cphase_ind_list", "cphase_sign_list", "camp_ind_list",
            "cphase_data", "camp_data", "logcamp_data",
        }
        self.assertEqual(set(result.keys()), expected)

    def test_cphase_shapes_consistent(self):
        """Closure phase index and sign lists should all have the same length."""
        frame = self._make_frame_data(5)
        result = extract_closure_indices(frame)
        n_cp = len(result["cphase_data"]["cphase"])
        for arr in result["cphase_ind_list"]:
            self.assertEqual(len(arr), n_cp)
        for arr in result["cphase_sign_list"]:
            self.assertEqual(len(arr), n_cp)

    def test_camp_shapes_consistent(self):
        """Closure amplitude index lists should all have the same length."""
        frame = self._make_frame_data(5)
        result = extract_closure_indices(frame)
        n_ca = len(result["camp_data"]["camp"])
        for arr in result["camp_ind_list"]:
            self.assertEqual(len(arr), n_ca)

    def test_cphase_sigma_positive(self):
        """Closure phase sigmas should all be positive."""
        frame = self._make_frame_data(5)
        result = extract_closure_indices(frame)
        self.assertTrue(np.all(result["cphase_data"]["sigmacp"] > 0))

    def test_logcamp_consistent_with_camp(self):
        """Log closure amplitudes should equal log of closure amplitudes."""
        frame = self._make_frame_data(5)
        result = extract_closure_indices(frame)
        camp = result["camp_data"]["camp"]
        logcamp = result["logcamp_data"]["camp"]
        np.testing.assert_allclose(logcamp, np.log(camp), rtol=1e-10)


class TestComputeNufftParams(unittest.TestCase):
    """Tests for compute_nufft_params."""

    def setUp(self):
        self.M = 16
        self.npix = 32
        self.fov_uas = 120.0
        self.uv = np.random.randn(self.M, 2) * 1e9

    def test_output_keys(self):
        """Returned dict should contain ktraj_vis and pulsefac_vis."""
        result = compute_nufft_params(self.uv, self.npix, self.fov_uas)
        self.assertIn("ktraj_vis", result)
        self.assertIn("pulsefac_vis", result)

    def test_ktraj_shape(self):
        """ktraj_vis should be (1, 2, M) float32 tensor."""
        result = compute_nufft_params(self.uv, self.npix, self.fov_uas)
        kt = result["ktraj_vis"]
        self.assertIsInstance(kt, torch.Tensor)
        self.assertEqual(kt.shape, (1, 2, self.M))
        self.assertEqual(kt.dtype, torch.float32)

    def test_pulsefac_shape(self):
        """pulsefac_vis should be (2, M) float32 tensor."""
        result = compute_nufft_params(self.uv, self.npix, self.fov_uas)
        pf = result["pulsefac_vis"]
        self.assertIsInstance(pf, torch.Tensor)
        self.assertEqual(pf.shape, (2, self.M))
        self.assertEqual(pf.dtype, torch.float32)

    def test_pulsefac_amplitude_bounded(self):
        """Pulse factor amplitude (sqrt(real^2+imag^2)) should be <= 1 (sinc^2 * sinc^2)."""
        result = compute_nufft_params(self.uv, self.npix, self.fov_uas)
        pf = result["pulsefac_vis"].numpy()
        amp = np.sqrt(pf[0] ** 2 + pf[1] ** 2)
        self.assertTrue(np.all(amp <= 1.0 + 1e-7))


class TestEstimateFlux(unittest.TestCase):
    """Tests for estimate_flux."""

    def test_constant_amplitude(self):
        """If all visibilities have the same amplitude, flux = that amplitude."""
        vis = 3.5 * np.exp(1j * np.random.randn(20))
        flux = estimate_flux(vis)
        np.testing.assert_allclose(flux, 3.5, rtol=1e-10)

    def test_return_type(self):
        """Flux should be a Python float."""
        vis = np.random.randn(8) + 1j * np.random.randn(8)
        flux = estimate_flux(vis)
        self.assertIsInstance(flux, float)

    def test_non_negative(self):
        """Estimated flux should be non-negative."""
        vis = np.random.randn(16) + 1j * np.random.randn(16)
        flux = estimate_flux(vis)
        self.assertGreaterEqual(flux, 0.0)


if __name__ == '__main__':
    unittest.main()
