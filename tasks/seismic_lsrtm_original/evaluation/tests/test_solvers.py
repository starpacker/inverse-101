"""Unit tests for src/solvers.py."""

import numpy as np
import torch

from src.solvers import subtract_direct_arrival, run_lsrtm
from src.physics_model import make_acquisition_geometry, make_ricker_wavelet, born_forward_model

FIXTURE_SOL = np.load("evaluation/fixtures/solvers.npz")
FIXTURE_PM = np.load("evaluation/fixtures/physics_model.npz")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSubtractDirectArrival:
    def test_homogeneous_residual_near_zero(self):
        observed = torch.from_numpy(FIXTURE_SOL["input_observed"]).to(DEVICE)
        v = torch.from_numpy(FIXTURE_SOL["input_v"]).to(DEVICE)
        dx = float(FIXTURE_PM["param_dx"])
        dt = float(FIXTURE_PM["param_dt"])
        freq = float(FIXTURE_PM["param_freq"])
        nt = int(FIXTURE_PM["param_nt"])
        n_shots = int(FIXTURE_PM["param_n_shots"])
        n_rec = int(FIXTURE_PM["param_n_rec"])

        sl, rl = make_acquisition_geometry(n_shots=n_shots, d_source=20, first_source=5,
            source_depth=2, n_receivers=n_rec, d_receiver=3, first_receiver=0,
            receiver_depth=2, device=DEVICE)
        w = make_ricker_wavelet(freq, nt, dt, n_shots, device=DEVICE)

        scattered = subtract_direct_arrival(observed, v, dx, dt, w, sl, rl, freq)
        rel_l2 = float(scattered.norm() / (observed.norm() + 1e-30))
        assert rel_l2 < 1e-5

    def test_output_shape(self):
        observed = torch.from_numpy(FIXTURE_SOL["input_observed"]).to(DEVICE)
        v = torch.from_numpy(FIXTURE_SOL["input_v"]).to(DEVICE)
        dx = float(FIXTURE_PM["param_dx"])
        dt = float(FIXTURE_PM["param_dt"])
        freq = float(FIXTURE_PM["param_freq"])
        nt = int(FIXTURE_PM["param_nt"])
        n_shots = int(FIXTURE_PM["param_n_shots"])
        n_rec = int(FIXTURE_PM["param_n_rec"])

        sl, rl = make_acquisition_geometry(n_shots=n_shots, d_source=20, first_source=5,
            source_depth=2, n_receivers=n_rec, d_receiver=3, first_receiver=0,
            receiver_depth=2, device=DEVICE)
        w = make_ricker_wavelet(freq, nt, dt, n_shots, device=DEVICE)

        scattered = subtract_direct_arrival(observed, v, dx, dt, w, sl, rl, freq)
        assert scattered.shape == observed.shape


class TestRunLsrtm:
    def test_loss_decreases(self):
        ny, nx = int(FIXTURE_PM["param_ny"]), int(FIXTURE_PM["param_nx"])
        n_shots = int(FIXTURE_PM["param_n_shots"])
        n_rec = int(FIXTURE_PM["param_n_rec"])
        nt = int(FIXTURE_PM["param_nt"])
        dx = float(FIXTURE_PM["param_dx"])
        dt = float(FIXTURE_PM["param_dt"])
        freq = float(FIXTURE_PM["param_freq"])

        v = torch.from_numpy(FIXTURE_PM["input_v_homo"]).to(DEVICE)
        scatter_true = torch.from_numpy(FIXTURE_PM["input_scatter"]).to(DEVICE)
        sl = torch.from_numpy(FIXTURE_PM["output_source_loc"]).to(DEVICE)
        rl = torch.from_numpy(FIXTURE_PM["output_receiver_loc"]).to(DEVICE)
        w = torch.from_numpy(FIXTURE_PM["output_wavelet"]).to(DEVICE)

        with torch.no_grad():
            target = born_forward_model(v, scatter_true, dx, dt, w, sl, rl, freq)

        _, losses = run_lsrtm(v, dx, dt, w, sl, rl, target, freq,
            n_epochs=3, device=DEVICE, print_every=99)
        assert losses[-1] < losses[0]

    def test_output_shape(self):
        ny, nx = int(FIXTURE_PM["param_ny"]), int(FIXTURE_PM["param_nx"])
        n_shots = int(FIXTURE_PM["param_n_shots"])
        n_rec = int(FIXTURE_PM["param_n_rec"])
        nt = int(FIXTURE_PM["param_nt"])
        dx = float(FIXTURE_PM["param_dx"])
        dt = float(FIXTURE_PM["param_dt"])
        freq = float(FIXTURE_PM["param_freq"])

        v = torch.from_numpy(FIXTURE_PM["input_v_homo"]).to(DEVICE)
        sl = torch.from_numpy(FIXTURE_PM["output_source_loc"]).to(DEVICE)
        rl = torch.from_numpy(FIXTURE_PM["output_receiver_loc"]).to(DEVICE)
        w = torch.from_numpy(FIXTURE_PM["output_wavelet"]).to(DEVICE)
        target = torch.randn(n_shots, n_rec, nt, device=DEVICE)

        scatter_inv, losses = run_lsrtm(v, dx, dt, w, sl, rl, target, freq,
            n_epochs=1, device=DEVICE, print_every=99)
        assert scatter_inv.shape == (ny, nx)
        assert len(losses) == 1
