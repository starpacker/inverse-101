"""Tests for src/preprocessing.py"""
import os
import json
import unittest
import numpy as np

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "preprocessing")
DATA_DIR = os.path.join(TASK_DIR, "data")

import sys
sys.path.insert(0, TASK_DIR)


class TestLoadMetadata(unittest.TestCase):
    def test_returns_dict(self):
        from src.preprocessing import load_metadata
        result = load_metadata(DATA_DIR)
        self.assertIsInstance(result, dict)

    def test_has_required_keys(self):
        from src.preprocessing import load_metadata
        result = load_metadata(DATA_DIR)
        for key in ["sample", "color", "wavelength_um", "NA", "magnification",
                     "pixel_size_um", "num_modes", "upsample_ratio", "training"]:
            self.assertIn(key, result)

    def test_values(self):
        from src.preprocessing import load_metadata
        result = load_metadata(DATA_DIR)
        self.assertEqual(result["sample"], "BloodSmearTilt")
        self.assertEqual(result["color"], "g")
        self.assertAlmostEqual(result["wavelength_um"], 0.5126, places=4)
        self.assertAlmostEqual(result["NA"], 0.256, places=3)


class TestLoadRawData(unittest.TestCase):
    def setUp(self):
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "load_raw_data.npz"),
                               allow_pickle=True)
        from src.preprocessing import load_raw_data
        self.result = load_raw_data(DATA_DIR)

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_required_keys(self):
        for key in ["I_low", "na_calib", "mag", "dpix_c", "na_cal"]:
            self.assertIn(key, self.result)

    def test_I_low_shape(self):
        expected = tuple(self.fixture["output_I_low_shape"])
        self.assertEqual(self.result["I_low"].shape, expected)

    def test_na_calib_shape(self):
        np.testing.assert_array_equal(
            self.result["na_calib"], self.fixture["output_na_calib"]
        )

    def test_scalar_values(self):
        np.testing.assert_allclose(self.result["mag"], self.fixture["output_mag"], rtol=1e-5)
        np.testing.assert_allclose(self.result["dpix_c"], self.fixture["output_dpix_c"], rtol=1e-5)
        np.testing.assert_allclose(self.result["na_cal"], self.fixture["output_na_cal"], rtol=1e-5)


class TestComputeOpticalParams(unittest.TestCase):
    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR, "compute_optical_params.npz"),
            allow_pickle=True,
        )
        from src.preprocessing import load_raw_data, load_metadata, compute_optical_params
        metadata = load_metadata(DATA_DIR)
        raw_data = load_raw_data(DATA_DIR)
        self.result = compute_optical_params(raw_data, metadata)

    def test_Fxx1(self):
        np.testing.assert_allclose(
            self.result["Fxx1"], self.fixture["output_Fxx1"], rtol=1e-10
        )

    def test_Fyy1(self):
        np.testing.assert_allclose(
            self.result["Fyy1"], self.fixture["output_Fyy1"], rtol=1e-10
        )

    def test_ledpos_true(self):
        np.testing.assert_array_equal(
            self.result["ledpos_true"], self.fixture["output_ledpos_true"]
        )

    def test_order(self):
        np.testing.assert_array_equal(
            self.result["order"], self.fixture["output_order"]
        )

    def test_dimensions(self):
        self.assertEqual(self.result["M"], int(self.fixture["output_M"]))
        self.assertEqual(self.result["N"], int(self.fixture["output_N"]))
        self.assertEqual(self.result["MM"], int(self.fixture["output_MM"]))
        self.assertEqual(self.result["NN"], int(self.fixture["output_NN"]))

    def test_k_values(self):
        np.testing.assert_allclose(self.result["k0"], self.fixture["output_k0"], rtol=1e-10)
        np.testing.assert_allclose(self.result["kmax"], self.fixture["output_kmax"], rtol=1e-10)

    def test_Isum_properties(self):
        expected_shape = tuple(self.fixture["output_Isum_shape"])
        self.assertEqual(self.result["Isum"].shape, expected_shape)
        np.testing.assert_allclose(
            self.result["Isum"].max(), self.fixture["output_Isum_max"], rtol=1e-5
        )


class TestLoadGroundTruth(unittest.TestCase):
    def setUp(self):
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "load_ground_truth.npz"),
                               allow_pickle=True)
        from src.preprocessing import load_ground_truth
        self.result = load_ground_truth(DATA_DIR)

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_required_keys(self):
        for key in ["I_stack", "zvec"]:
            self.assertIn(key, self.result)

    def test_I_stack_shape(self):
        expected = tuple(self.fixture["output_I_stack_shape"])
        self.assertEqual(self.result["I_stack"].shape, expected)

    def test_zvec_values(self):
        np.testing.assert_allclose(
            self.result["zvec"], self.fixture["output_zvec"], rtol=1e-5
        )

    def test_dtypes(self):
        self.assertEqual(self.result["I_stack"].dtype, np.float32)
        self.assertEqual(self.result["zvec"].dtype, np.float32)


class TestComputePupilAndPropagation(unittest.TestCase):
    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR, "compute_pupil.npz"), allow_pickle=True
        )
        from src.preprocessing import (
            load_raw_data, load_metadata, compute_optical_params,
            compute_pupil_and_propagation,
        )
        metadata = load_metadata(DATA_DIR)
        raw_data = load_raw_data(DATA_DIR)
        optical_params = compute_optical_params(raw_data, metadata)
        self.result = compute_pupil_and_propagation(optical_params)

    def test_Pupil0_shape(self):
        expected = self.fixture["output_Pupil0"]
        self.assertEqual(self.result["Pupil0"].shape, expected.shape)

    def test_Pupil0_values(self):
        np.testing.assert_array_equal(
            self.result["Pupil0"], self.fixture["output_Pupil0"]
        )

    def test_Pupil0_sum(self):
        np.testing.assert_allclose(
            self.result["Pupil0"].sum(), self.fixture["output_Pupil0_sum"], rtol=1e-10
        )

    def test_kzz_shape(self):
        expected_real = self.fixture["output_kzz_real"]
        self.assertEqual(self.result["kzz"].shape, expected_real.shape)

    def test_kzz_values(self):
        np.testing.assert_allclose(
            self.result["kzz"].real.astype("float32"),
            self.fixture["output_kzz_real"],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            self.result["kzz"].imag.astype("float32"),
            self.fixture["output_kzz_imag"],
            atol=1e-5,
        )


class TestComputeZParams(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(FIXTURE_DIR, "compute_z_params.json")) as f:
            self.fixture = json.load(f)
        from src.preprocessing import (
            load_raw_data, load_metadata, compute_optical_params, compute_z_params,
        )
        metadata = load_metadata(DATA_DIR)
        raw_data = load_raw_data(DATA_DIR)
        optical_params = compute_optical_params(raw_data, metadata)
        self.result = compute_z_params(metadata, optical_params)

    def test_values(self):
        for key in ["DOF", "delta_z", "num_z", "z_min", "z_max"]:
            self.assertAlmostEqual(self.result[key], self.fixture[key], places=6)


@unittest.skipUnless(
    __import__("torch").cuda.is_available(), "CUDA required"
)
class TestPrepareData(unittest.TestCase):
    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR, "prepare_data.npz"), allow_pickle=True
        )
        from src.preprocessing import prepare_data
        self.result = prepare_data(DATA_DIR, device="cuda:0")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_required_keys(self):
        for key in ["metadata", "optical_params", "pupil_data", "z_params", "Isum"]:
            self.assertIn(key, self.result)

    def test_Isum_shape(self):
        expected = tuple(self.fixture["output_Isum_shape"])
        self.assertEqual(self.result["Isum"].shape, __import__("torch").Size(expected))

    def test_pupil_data_keys(self):
        for key in ["Pupil0", "kzz"]:
            self.assertIn(key, self.result["pupil_data"])

    def test_Isum_on_device(self):
        self.assertTrue(self.result["Isum"].is_cuda)


if __name__ == "__main__":
    unittest.main()
