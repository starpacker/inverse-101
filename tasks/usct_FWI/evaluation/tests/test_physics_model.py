"""Tests for physics_model (CBS solver) using external fixtures."""

import os
import sys
import pickle
import numpy as np
import torch
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXTURE_DIR, name), "rb") as f:
        return pickle.load(f)


from src.physics_model import setup_domain, cbs_solve, ensure_3d_shape


class TestEnsure3dShape:
    def test_2d_input(self):
        t = torch.ones(64, 64)
        sizes, roi, bw, flags = ensure_3d_shape(t, 40)
        assert flags == [True, True, False]
        assert sizes[2] == 1
        assert sizes[0] >= 64 + 2 * 40


class TestSetupDomain:
    def test_domain_keys(self):
        config = load_fixture("config_cbs_solve.pkl")
        velocity = torch.ones(
            config["velocity_shape"], device="cuda"
        ) * config["velocity_value"]
        domain = setup_domain(
            velocity, config["freq"], config["dh"], config["ppw"],
            config["lamb"], config["boundary_widths"],
            config["born_max"], config["energy_threshold"],
        )
        for key in ["grid", "wiggles", "n_wiggle", "gamma",
                     "mix_weight1", "mix_weight2", "roi", "flag_dim",
                     "epsilon", "Bl", "A", "dh"]:
            assert key in domain, f"Missing key: {key}"
        assert domain["n_wiggle"] == 4  # 2D -> 2^2 wiggles


class TestCBSSolve:
    def test_against_fixture(self):
        """CBS solve output should match precomputed fixture exactly."""
        config = load_fixture("config_cbs_solve.pkl")
        inp = load_fixture("input_cbs_solve.pkl")
        expected = load_fixture("output_cbs_solve.pkl")

        velocity = torch.ones(
            config["velocity_shape"], device="cuda"
        ) * config["velocity_value"]
        domain = setup_domain(
            velocity, config["freq"], config["dh"], config["ppw"],
            config["lamb"], config["boundary_widths"],
            config["born_max"], config["energy_threshold"],
        )
        field = cbs_solve(inp["src_ix"], inp["src_iy"], domain)

        assert field.shape == tuple(expected["field_shape"])
        np.testing.assert_allclose(
            field.cpu().numpy(), expected["field"],
            rtol=1e-5, atol=1e-10,
        )

    def test_nonzero_field(self):
        expected = load_fixture("output_cbs_solve.pkl")
        assert expected["field_abs_max"] > 0

    def test_approximate_reciprocity(self):
        """u(src_a, rec_b) should approximately equal u(src_b, rec_a)."""
        config = load_fixture("config_cbs_solve.pkl")
        velocity = torch.ones(
            config["velocity_shape"], device="cuda"
        ) * config["velocity_value"]
        domain = setup_domain(
            velocity, config["freq"], config["dh"], config["ppw"],
            config["lamb"], config["boundary_widths"],
            200, 1e-5,  # more iterations for better reciprocity
        )
        u_ab = cbs_solve(20, 40, domain)
        u_ba = cbs_solve(40, 20, domain)
        val_ab = u_ab[40, 20]
        val_ba = u_ba[20, 40]
        ratio = torch.abs(val_ab) / (torch.abs(val_ba) + 1e-30)
        assert 0.5 < ratio.item() < 2.0, f"Reciprocity ratio: {ratio.item()}"
