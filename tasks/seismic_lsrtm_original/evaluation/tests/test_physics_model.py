"""Unit tests for src/physics_model.py."""

import numpy as np
import torch

from src.physics_model import (
    make_acquisition_geometry,
    make_ricker_wavelet,
    forward_model,
    born_forward_model,
)

FIXTURE = np.load("evaluation/fixtures/physics_model.npz")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestMakeAcquisitionGeometry:
    def test_source_shape(self):
        n_shots = int(FIXTURE["param_n_shots"])
        sl, _ = make_acquisition_geometry(n_shots=n_shots, d_source=20, first_source=5,
            n_receivers=int(FIXTURE["param_n_rec"]), d_receiver=3, first_receiver=0)
        assert sl.shape == (n_shots, 1, 2)

    def test_receiver_shape(self):
        n_shots = int(FIXTURE["param_n_shots"])
        n_rec = int(FIXTURE["param_n_rec"])
        _, rl = make_acquisition_geometry(n_shots=n_shots, d_source=20, first_source=5,
            n_receivers=n_rec, d_receiver=3, first_receiver=0)
        assert rl.shape == (n_shots, n_rec, 2)

    def test_source_values(self):
        n_shots = int(FIXTURE["param_n_shots"])
        n_rec = int(FIXTURE["param_n_rec"])
        sl, _ = make_acquisition_geometry(n_shots=n_shots, d_source=20, first_source=5,
            source_depth=2, n_receivers=n_rec, d_receiver=3, first_receiver=0, receiver_depth=2)
        expected = torch.from_numpy(FIXTURE["output_source_loc"])
        assert torch.equal(sl, expected)

    def test_receiver_values(self):
        n_shots = int(FIXTURE["param_n_shots"])
        n_rec = int(FIXTURE["param_n_rec"])
        _, rl = make_acquisition_geometry(n_shots=n_shots, d_source=20, first_source=5,
            source_depth=2, n_receivers=n_rec, d_receiver=3, first_receiver=0, receiver_depth=2)
        expected = torch.from_numpy(FIXTURE["output_receiver_loc"])
        assert torch.equal(rl, expected)


class TestMakeRickerWavelet:
    def test_shape(self):
        freq, nt, dt = float(FIXTURE["param_freq"]), int(FIXTURE["param_nt"]), float(FIXTURE["param_dt"])
        n_shots = int(FIXTURE["param_n_shots"])
        w = make_ricker_wavelet(freq, nt, dt, n_shots)
        assert w.shape == (n_shots, 1, nt)

    def test_values(self):
        freq, nt, dt = float(FIXTURE["param_freq"]), int(FIXTURE["param_nt"]), float(FIXTURE["param_dt"])
        n_shots = int(FIXTURE["param_n_shots"])
        w = make_ricker_wavelet(freq, nt, dt, n_shots)
        expected = torch.from_numpy(FIXTURE["output_wavelet"])
        torch.testing.assert_close(w, expected, rtol=1e-6, atol=1e-8)

    def test_all_shots_identical(self):
        freq, nt, dt = float(FIXTURE["param_freq"]), int(FIXTURE["param_nt"]), float(FIXTURE["param_dt"])
        n_shots = int(FIXTURE["param_n_shots"])
        w = make_ricker_wavelet(freq, nt, dt, n_shots)
        for i in range(1, n_shots):
            assert torch.equal(w[0], w[i])


class TestForwardModel:
    def test_output_shape(self):
        n_shots, n_rec, nt = int(FIXTURE["param_n_shots"]), int(FIXTURE["param_n_rec"]), int(FIXTURE["param_nt"])
        v = torch.from_numpy(FIXTURE["input_v_homo"]).to(DEVICE)
        sl = torch.from_numpy(FIXTURE["output_source_loc"]).to(DEVICE)
        rl = torch.from_numpy(FIXTURE["output_receiver_loc"]).to(DEVICE)
        w = torch.from_numpy(FIXTURE["output_wavelet"]).to(DEVICE)
        dx, dt, freq = float(FIXTURE["param_dx"]), float(FIXTURE["param_dt"]), float(FIXTURE["param_freq"])
        with torch.no_grad():
            rec = forward_model(v, dx, dt, w, sl, rl, freq)
        assert rec.shape == (n_shots, n_rec, nt)

    def test_output_values(self):
        v = torch.from_numpy(FIXTURE["input_v_homo"]).to(DEVICE)
        sl = torch.from_numpy(FIXTURE["output_source_loc"]).to(DEVICE)
        rl = torch.from_numpy(FIXTURE["output_receiver_loc"]).to(DEVICE)
        w = torch.from_numpy(FIXTURE["output_wavelet"]).to(DEVICE)
        dx, dt, freq = float(FIXTURE["param_dx"]), float(FIXTURE["param_dt"]), float(FIXTURE["param_freq"])
        with torch.no_grad():
            rec = forward_model(v, dx, dt, w, sl, rl, freq)
        expected = torch.from_numpy(FIXTURE["output_rec_forward"]).to(DEVICE)
        torch.testing.assert_close(rec, expected, rtol=1e-10, atol=0)


class TestBornForwardModel:
    def test_output_shape(self):
        n_shots, n_rec, nt = int(FIXTURE["param_n_shots"]), int(FIXTURE["param_n_rec"]), int(FIXTURE["param_nt"])
        v = torch.from_numpy(FIXTURE["input_v_homo"]).to(DEVICE)
        scatter = torch.from_numpy(FIXTURE["input_scatter"]).to(DEVICE)
        sl = torch.from_numpy(FIXTURE["output_source_loc"]).to(DEVICE)
        rl = torch.from_numpy(FIXTURE["output_receiver_loc"]).to(DEVICE)
        w = torch.from_numpy(FIXTURE["output_wavelet"]).to(DEVICE)
        dx, dt, freq = float(FIXTURE["param_dx"]), float(FIXTURE["param_dt"]), float(FIXTURE["param_freq"])
        with torch.no_grad():
            rec = born_forward_model(v, scatter, dx, dt, w, sl, rl, freq)
        assert rec.shape == (n_shots, n_rec, nt)

    def test_output_values(self):
        v = torch.from_numpy(FIXTURE["input_v_homo"]).to(DEVICE)
        scatter = torch.from_numpy(FIXTURE["input_scatter"]).to(DEVICE)
        sl = torch.from_numpy(FIXTURE["output_source_loc"]).to(DEVICE)
        rl = torch.from_numpy(FIXTURE["output_receiver_loc"]).to(DEVICE)
        w = torch.from_numpy(FIXTURE["output_wavelet"]).to(DEVICE)
        dx, dt, freq = float(FIXTURE["param_dx"]), float(FIXTURE["param_dt"]), float(FIXTURE["param_freq"])
        with torch.no_grad():
            rec = born_forward_model(v, scatter, dx, dt, w, sl, rl, freq)
        expected = torch.from_numpy(FIXTURE["output_rec_born"]).to(DEVICE)
        torch.testing.assert_close(rec, expected, rtol=1e-10, atol=0)

    def test_zero_scatter_gives_zero(self):
        v = torch.from_numpy(FIXTURE["input_v_homo"]).to(DEVICE)
        scatter = torch.zeros_like(v)
        sl = torch.from_numpy(FIXTURE["output_source_loc"]).to(DEVICE)
        rl = torch.from_numpy(FIXTURE["output_receiver_loc"]).to(DEVICE)
        w = torch.from_numpy(FIXTURE["output_wavelet"]).to(DEVICE)
        dx, dt, freq = float(FIXTURE["param_dx"]), float(FIXTURE["param_dt"]), float(FIXTURE["param_freq"])
        with torch.no_grad():
            rec = born_forward_model(v, scatter, dx, dt, w, sl, rl, freq)
        assert rec.abs().max() < 1e-10

    def test_linearity(self):
        v = torch.from_numpy(FIXTURE["input_v_homo"]).to(DEVICE)
        scatter = torch.from_numpy(FIXTURE["input_scatter"]).to(DEVICE)
        sl = torch.from_numpy(FIXTURE["output_source_loc"]).to(DEVICE)
        rl = torch.from_numpy(FIXTURE["output_receiver_loc"]).to(DEVICE)
        w = torch.from_numpy(FIXTURE["output_wavelet"]).to(DEVICE)
        dx, dt, freq = float(FIXTURE["param_dx"]), float(FIXTURE["param_dt"]), float(FIXTURE["param_freq"])
        with torch.no_grad():
            rec1 = born_forward_model(v, scatter, dx, dt, w, sl, rl, freq)
            rec2 = born_forward_model(v, 2 * scatter, dx, dt, w, sl, rl, freq)
        rel_err = (rec2 - 2 * rec1).norm() / (2 * rec1).norm()
        assert rel_err < 0.01, f"Born linearity violated: rel_err={rel_err:.4e}"
