"""Tests for src/solvers.py."""

import numpy as np
import pytest
import os

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


class TestUniversalBackProjection:
    """Tests for universal back-projection reconstruction."""

    @pytest.fixture
    def fixture(self):
        return np.load(os.path.join(FIXTURES_DIR, "solver_ubp.npz"))

    def test_shape(self, fixture):
        from src.solvers import universal_back_projection
        recon, xf, yf, zf = universal_back_projection(
            fixture["input_signals"],
            fixture["param_xd"], fixture["param_yd"],
            fixture["param_t"], float(fixture["param_z_target"]),
            resolution=1e-3,
        )
        assert recon.shape[0] == len(xf)
        assert recon.shape[1] == len(yf)
        assert recon.shape[2] == 1

    def test_exact_parity(self, fixture):
        from src.solvers import universal_back_projection
        recon, xf, yf, zf = universal_back_projection(
            fixture["input_signals"],
            fixture["param_xd"], fixture["param_yd"],
            fixture["param_t"], float(fixture["param_z_target"]),
            resolution=1e-3,
        )
        np.testing.assert_allclose(
            np.squeeze(recon), np.squeeze(fixture["output_recon"]),
            rtol=1e-10)

    def test_normalisation(self, fixture):
        """Peak value should be 1.0 after normalisation."""
        from src.solvers import universal_back_projection
        recon, _, _, _ = universal_back_projection(
            fixture["input_signals"],
            fixture["param_xd"], fixture["param_yd"],
            fixture["param_t"], float(fixture["param_z_target"]),
            resolution=1e-3,
        )
        assert abs(np.max(np.abs(recon)) - 1.0) < 1e-10

    def test_grid_covers_aperture(self, fixture):
        """Reconstruction grid should span the detector aperture."""
        from src.solvers import universal_back_projection
        _, xf, yf, _ = universal_back_projection(
            fixture["input_signals"],
            fixture["param_xd"], fixture["param_yd"],
            fixture["param_t"], float(fixture["param_z_target"]),
            resolution=1e-3,
        )
        assert xf[0] <= fixture["param_xd"][0] + 1e-10
        assert xf[-1] >= fixture["param_xd"][-1] - 1e-10
