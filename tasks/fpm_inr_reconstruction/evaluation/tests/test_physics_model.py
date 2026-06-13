"""Tests for src/physics_model.py"""
import os
import unittest
import numpy as np
import torch

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures", "physics_model")

import sys
sys.path.insert(0, TASK_DIR)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestComputeSpectrumMask(unittest.TestCase):
    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR, "compute_spectrum_mask.npz"),
            allow_pickle=True,
        )
        from src.preprocessing import (
            load_raw_data, load_metadata, compute_optical_params,
            compute_pupil_and_propagation,
        )
        from src.physics_model import FPMForwardModel

        device = "cuda:0"
        data_dir = os.path.join(TASK_DIR, "data")
        metadata = load_metadata(data_dir)
        raw_data = load_raw_data(data_dir)
        optical_params = compute_optical_params(raw_data, metadata)
        pupil_data_np = compute_pupil_and_propagation(optical_params)

        Pupil0 = (
            torch.from_numpy(pupil_data_np["Pupil0"])
            .view(1, 1, optical_params["M"], optical_params["N"])
            .to(device)
        )
        kzz = torch.from_numpy(pupil_data_np["kzz"]).to(device).unsqueeze(0)

        self.fwd = FPMForwardModel(
            Pupil0=Pupil0, kzz=kzz,
            ledpos_true=optical_params["ledpos_true"],
            M=optical_params["M"], N=optical_params["N"],
            MAGimg=optical_params["MAGimg"],
        )

    def test_shape(self):
        dz = torch.tensor([0.0]).cuda()
        result = self.fwd.compute_spectrum_mask(dz, [0])
        expected_shape = tuple(self.fixture["output_shape"])
        self.assertEqual(result.shape, torch.Size(expected_shape))

    def test_values(self):
        dz = torch.tensor([0.0]).cuda()
        result = self.fwd.compute_spectrum_mask(dz, [0])
        result_np = result.cpu().numpy()

        expected_real = self.fixture["output_spectrum_mask_real"]
        expected_imag = self.fixture["output_spectrum_mask_imag"]

        np.testing.assert_allclose(
            result_np.real.astype("float32"), expected_real, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            result_np.imag.astype("float32"), expected_imag, atol=1e-5
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGetLedCoords(unittest.TestCase):
    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR, "get_led_coords.npz"), allow_pickle=True
        )
        from src.preprocessing import (
            load_raw_data, load_metadata, compute_optical_params,
            compute_pupil_and_propagation,
        )
        from src.physics_model import FPMForwardModel

        device = "cuda:0"
        data_dir = os.path.join(TASK_DIR, "data")
        metadata = load_metadata(data_dir)
        raw_data = load_raw_data(data_dir)
        optical_params = compute_optical_params(raw_data, metadata)
        pupil_data_np = compute_pupil_and_propagation(optical_params)

        Pupil0 = (
            torch.from_numpy(pupil_data_np["Pupil0"])
            .view(1, 1, optical_params["M"], optical_params["N"])
            .to(device)
        )
        kzz = torch.from_numpy(pupil_data_np["kzz"]).to(device).unsqueeze(0)

        self.fwd = FPMForwardModel(
            Pupil0=Pupil0, kzz=kzz,
            ledpos_true=optical_params["ledpos_true"],
            M=optical_params["M"], N=optical_params["N"],
            MAGimg=optical_params["MAGimg"],
        )

    def test_values(self):
        led_num = list(self.fixture["input_led_num"])
        x_0, y_0, x_1, y_1 = self.fwd.get_led_coords(led_num)
        np.testing.assert_array_equal(x_0, self.fixture["output_x_0"])
        np.testing.assert_array_equal(y_0, self.fixture["output_y_0"])
        np.testing.assert_array_equal(x_1, self.fixture["output_x_1"])
        np.testing.assert_array_equal(y_1, self.fixture["output_y_1"])

    def test_span_equals_M(self):
        led_num = [0]
        x_0, y_0, x_1, y_1 = self.fwd.get_led_coords(led_num)
        self.assertEqual(int(x_1[0] - x_0[0]), self.fwd.M)
        self.assertEqual(int(y_1[0] - y_0[0]), self.fwd.N)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGetSubSpectrum(unittest.TestCase):
    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR, "get_sub_spectrum.npz"), allow_pickle=True
        )
        from src.preprocessing import (
            load_raw_data, load_metadata, compute_optical_params,
            compute_pupil_and_propagation,
        )
        from src.physics_model import FPMForwardModel
        from src.solvers import FullModel, load_model_with_required_grad

        device = "cuda:0"
        data_dir = os.path.join(TASK_DIR, "data")
        metadata = load_metadata(data_dir)
        raw_data = load_raw_data(data_dir)
        optical_params = compute_optical_params(raw_data, metadata)
        pupil_data_np = compute_pupil_and_propagation(optical_params)
        z_params_mod = {"z_min": -20.0, "z_max": 20.0}

        Pupil0 = (
            torch.from_numpy(pupil_data_np["Pupil0"])
            .view(1, 1, optical_params["M"], optical_params["N"])
            .to(device)
        )
        kzz = torch.from_numpy(pupil_data_np["kzz"]).to(device).unsqueeze(0)

        self.fwd = FPMForwardModel(
            Pupil0=Pupil0, kzz=kzz,
            ledpos_true=optical_params["ledpos_true"],
            M=optical_params["M"], N=optical_params["N"],
            MAGimg=optical_params["MAGimg"],
        )

        model = FullModel(
            w=optical_params["MM"], h=optical_params["MM"],
            num_feats=32, x_mode=metadata["num_modes"], y_mode=metadata["num_modes"],
            z_min=-20.0, z_max=20.0, ds_factor=1, use_layernorm=False,
        ).to(device)
        load_model_with_required_grad(
            model,
            os.path.join(TASK_DIR, "evaluation", "reference_outputs", "model_weights.pth"),
        )
        model.eval()

        dz = torch.tensor([0.0]).to(device)
        self.led_num = list(self.fixture["input_led_num"])
        self.spectrum_mask = self.fwd.compute_spectrum_mask(dz, self.led_num)
        with torch.no_grad():
            ampli, phase = model(dz)
            self.img_complex = ampli * torch.exp(1j * phase)

    def test_shape(self):
        result = self.fwd.get_sub_spectrum(self.img_complex, self.led_num, self.spectrum_mask)
        expected_shape = tuple(self.fixture["output_shape"])
        self.assertEqual(result.shape, torch.Size(expected_shape))

    def test_crop_values(self):
        result = self.fwd.get_sub_spectrum(self.img_complex, self.led_num, self.spectrum_mask)
        crop = result[0, 0, :64, :64].float().cpu().numpy()
        np.testing.assert_allclose(crop, self.fixture["output_crop"], rtol=1e-4, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGetMeasuredAmplitudes(unittest.TestCase):
    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR, "get_measured_amplitudes.npz"), allow_pickle=True
        )
        from src.preprocessing import (
            load_raw_data, load_metadata, compute_optical_params,
            compute_pupil_and_propagation,
        )
        from src.physics_model import FPMForwardModel

        device = "cuda:0"
        data_dir = os.path.join(TASK_DIR, "data")
        metadata = load_metadata(data_dir)
        raw_data = load_raw_data(data_dir)
        optical_params = compute_optical_params(raw_data, metadata)
        pupil_data_np = compute_pupil_and_propagation(optical_params)

        Pupil0 = (
            torch.from_numpy(pupil_data_np["Pupil0"])
            .view(1, 1, optical_params["M"], optical_params["N"])
            .to(device)
        )
        kzz = torch.from_numpy(pupil_data_np["kzz"]).to(device).unsqueeze(0)
        self.Isum = torch.from_numpy(optical_params["Isum"]).to(device)

        self.fwd = FPMForwardModel(
            Pupil0=Pupil0, kzz=kzz,
            ledpos_true=optical_params["ledpos_true"],
            M=optical_params["M"], N=optical_params["N"],
            MAGimg=optical_params["MAGimg"],
        )

    def test_shape(self):
        led_num = list(self.fixture["input_led_num"])
        n_z = int(self.fixture["input_n_z"])
        result = self.fwd.get_measured_amplitudes(self.Isum, led_num, n_z)
        expected_shape = tuple(self.fixture["output_shape"])
        self.assertEqual(result.shape, torch.Size(expected_shape))

    def test_crop_values(self):
        led_num = list(self.fixture["input_led_num"])
        n_z = int(self.fixture["input_n_z"])
        result = self.fwd.get_measured_amplitudes(self.Isum, led_num, n_z)
        crop = result[0, 0, :64, :64].float().cpu().numpy()
        np.testing.assert_allclose(crop, self.fixture["output_crop"], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
