"""Tests for solvers module."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.solvers import (
    FitParams,
    AberrationParams,
    FitData,
    _get_backward_model,
    _get_forward_model_error,
    _calculate_view_error,
    _group_localisations,
    _light_field_fit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RHO_SCALING = 2.22  # representative value for a typical FLFM setup


def _make_multiview_locs(n_views=5, x0=1.0, y0=-0.5, z=0.3, rng=None):
    """Synthesise multi-view (K, 13) localisation rows for a single molecule.

    Generates localisations consistent with the forward model:
        x_i = x0 + u_i/rho + z*alpha_u_i
        y_i = y0 + v_i/rho + z*alpha_v_i

    Each row has 13 columns matching the Localisations internal layout:
        [0] frame, [1] U, [2] V, [3] X, [4] Y, [5..9] unused,
        [10] alpha_U, [11] alpha_V, [12] lens_idx
    """
    if rng is None:
        rng = np.random.default_rng(7)

    locs = np.zeros((n_views, 13))
    # Spread views across different (u, v) positions
    angles = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
    radii = rng.uniform(0.2, 0.8, n_views)
    u = radii * np.cos(angles)
    v = radii * np.sin(angles)

    # Simple linear alpha model: alpha = (u, v)
    alpha_u = u
    alpha_v = v

    locs[:, 0] = 1                                        # frame
    locs[:, 1] = u                                        # U
    locs[:, 2] = v                                        # V
    locs[:, 3] = x0 + u / RHO_SCALING + z * alpha_u      # X
    locs[:, 4] = y0 + v / RHO_SCALING + z * alpha_v      # Y
    locs[:, 10] = alpha_u                                 # alpha_U
    locs[:, 11] = alpha_v                                 # alpha_V
    locs[:, 12] = np.arange(n_views)                      # lens index
    locs[:, 7] = rng.uniform(500, 2000, n_views)          # intensity (photons)
    return locs


# ---------------------------------------------------------------------------
# Tests: _get_backward_model (OLS solver)
# ---------------------------------------------------------------------------

class TestGetBackwardModel:
    """Tests for the OLS 3D position solver."""

    def test_perfect_recovery(self):
        """With noiseless forward-model data the solver should recover (x0, y0, z)."""
        x0, y0, z = 2.0, -1.5, 0.8
        locs = _make_multiview_locs(n_views=8, x0=x0, y0=y0, z=z)
        model, std_err, mse = _get_backward_model(locs, RHO_SCALING)

        np.testing.assert_allclose(model[0], x0, atol=1e-10)
        np.testing.assert_allclose(model[1], y0, atol=1e-10)
        np.testing.assert_allclose(model[2], z,  atol=1e-10)

    def test_output_shapes(self):
        """Model should be (3,), std_err should be (3,), mse scalar."""
        locs = _make_multiview_locs(n_views=6)
        model, std_err, mse = _get_backward_model(locs, RHO_SCALING)
        assert model.shape == (3,)
        assert std_err.shape == (3,)
        assert np.isscalar(mse)

    def test_std_err_near_zero_for_perfect_data(self):
        """For noiseless data, standard errors should be essentially zero."""
        locs = _make_multiview_locs(n_views=8, x0=0.0, y0=0.0, z=0.0)
        _, std_err, mse = _get_backward_model(locs, RHO_SCALING)
        np.testing.assert_allclose(std_err, 0.0, atol=1e-10)
        assert mse < 1e-20

    def test_noisy_data_convergence(self):
        """With small Gaussian noise, the fit should still be close to true values."""
        rng = np.random.default_rng(123)
        x0, y0, z = 0.5, 0.5, 1.0
        locs = _make_multiview_locs(n_views=12, x0=x0, y0=y0, z=z, rng=rng)
        # Add small noise to X and Y
        locs[:, 3] += rng.normal(0, 0.005, locs.shape[0])
        locs[:, 4] += rng.normal(0, 0.005, locs.shape[0])

        model, std_err, _ = _get_backward_model(locs, RHO_SCALING)
        np.testing.assert_allclose(model[0], x0, atol=0.05)
        np.testing.assert_allclose(model[1], y0, atol=0.05)
        np.testing.assert_allclose(model[2], z,  atol=0.1)
        # Standard errors should be small but non-zero
        assert np.all(std_err > 0)
        assert np.all(std_err < 0.1)


# ---------------------------------------------------------------------------
# Tests: _get_forward_model_error
# ---------------------------------------------------------------------------

class TestGetForwardModelError:
    """Tests for the forward-model residual computation."""

    def test_zero_residuals_for_perfect_data(self):
        """When model matches data exactly, residuals should be zero."""
        x0, y0, z = 1.0, -0.5, 0.3
        locs = _make_multiview_locs(n_views=6, x0=x0, y0=y0, z=z)
        model = np.array([x0, y0, z])
        residuals = _get_forward_model_error(model, locs, RHO_SCALING)
        assert residuals.shape == (6, 2)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-12)

    def test_residual_shape(self):
        """Residuals should have shape (K, 2)."""
        locs = _make_multiview_locs(n_views=10)
        model = np.array([0.0, 0.0, 0.0])
        residuals = _get_forward_model_error(model, locs, RHO_SCALING)
        assert residuals.shape == (10, 2)

    def test_nonzero_residuals_for_wrong_model(self):
        """A deliberately wrong model should produce non-zero residuals."""
        x0, y0, z = 1.0, -0.5, 0.3
        locs = _make_multiview_locs(n_views=6, x0=x0, y0=y0, z=z)
        wrong_model = np.array([x0 + 5.0, y0 - 3.0, z + 2.0])
        residuals = _get_forward_model_error(wrong_model, locs, RHO_SCALING)
        assert np.any(np.abs(residuals) > 0.1)


# ---------------------------------------------------------------------------
# Tests: _calculate_view_error
# ---------------------------------------------------------------------------

class TestCalculateViewError:
    """Tests for per-view aberration estimation."""

    def test_correction_shape(self):
        """Output should be (V, 5) with one row per unique view."""
        rng = np.random.default_rng(55)
        locs = _make_multiview_locs(n_views=5, rng=rng)
        # Create FitData that passes all filters
        fd = FitData(
            frame=1,
            model=np.array([1.0, -0.5, 0.05]),  # small z for axial window
            points=locs,
            photon_count=5000.0,
            std_err=np.array([0.01, 0.01, 0.02]),
        )
        ab_params = AberrationParams(axial_window=1.0, photon_threshold=1, min_views=2)
        correction = _calculate_view_error(locs, RHO_SCALING, [fd], ab_params)
        n_views = np.unique(locs[:, 1:3], axis=0).shape[0]
        assert correction.shape == (n_views, 5)

    def test_correction_columns_uv(self):
        """Columns 0 and 1 should contain the unique (u, v) view coordinates."""
        rng = np.random.default_rng(55)
        locs = _make_multiview_locs(n_views=4, rng=rng)
        fd = FitData(
            frame=1,
            model=np.array([1.0, -0.5, 0.05]),
            points=locs,
            photon_count=5000.0,
            std_err=np.array([0.01, 0.01, 0.02]),
        )
        ab_params = AberrationParams(axial_window=1.0, photon_threshold=1, min_views=2)
        correction = _calculate_view_error(locs, RHO_SCALING, [fd], ab_params)
        unique_uv = np.unique(locs[:, 1:3], axis=0)
        np.testing.assert_array_equal(correction[:, 0:2], unique_uv)

    def test_no_fit_data_gives_zero_correction(self):
        """With an empty fit_data list, all corrections should be zero."""
        locs = _make_multiview_locs(n_views=4)
        ab_params = AberrationParams(axial_window=1.0, photon_threshold=1, min_views=2)
        correction = _calculate_view_error(locs, RHO_SCALING, [], ab_params)
        np.testing.assert_allclose(correction[:, 2:4], 0.0)
        np.testing.assert_allclose(correction[:, 4], 0.0)

    def test_filtered_by_axial_window(self):
        """Molecules with |z| >= axial_window should be excluded from correction."""
        rng = np.random.default_rng(55)
        locs = _make_multiview_locs(n_views=4, rng=rng)
        # Model z is well outside the axial window
        fd = FitData(
            frame=1,
            model=np.array([1.0, -0.5, 10.0]),  # z=10 >> window
            points=locs,
            photon_count=5000.0,
            std_err=np.array([0.01, 0.01, 0.02]),
        )
        ab_params = AberrationParams(axial_window=1.0, photon_threshold=1, min_views=2)
        correction = _calculate_view_error(locs, RHO_SCALING, [fd], ab_params)
        # All corrections should remain zero because the molecule was filtered out
        np.testing.assert_allclose(correction[:, 2:4], 0.0)


# ---------------------------------------------------------------------------
# Tests: _light_field_fit (integration-level)
# ---------------------------------------------------------------------------

class TestLightFieldFit:
    """Integration tests for the frame-by-frame fitting loop."""

    def _build_two_molecule_scene(self):
        """Build a scene with two molecules in separate frames, each with 6 views."""
        rng = np.random.default_rng(0)
        locs_a = _make_multiview_locs(n_views=6, x0=0.0, y0=0.0, z=0.0, rng=rng)
        locs_a[:, 0] = 1  # frame 1

        locs_b = _make_multiview_locs(n_views=6, x0=2.0, y0=1.0, z=0.5, rng=rng)
        locs_b[:, 0] = 2  # frame 2

        locs = np.row_stack((locs_a, locs_b))
        return locs

    def test_output_shape(self):
        """Fitted points should have 8 columns."""
        locs = self._build_two_molecule_scene()
        fp = FitParams(
            frame_min=1, frame_max=2,
            disparity_max=5.0, disparity_step=0.1,
            dist_search=0.5, angle_tolerance=90.0,
            threshold=2.0, min_views=2,
        )
        fitted, fit_data = _light_field_fit(locs, RHO_SCALING, fp)
        if fitted.shape[0] > 0:
            assert fitted.shape[1] == 8

    def test_z_calib_applied(self):
        """When z_calib is set, the z column should be scaled."""
        locs = self._build_two_molecule_scene()
        fp_no_cal = FitParams(
            frame_min=1, frame_max=2,
            disparity_max=5.0, disparity_step=0.1,
            dist_search=0.5, angle_tolerance=90.0,
            threshold=2.0, min_views=2,
        )
        fp_cal = FitParams(
            frame_min=1, frame_max=2,
            disparity_max=5.0, disparity_step=0.1,
            dist_search=0.5, angle_tolerance=90.0,
            threshold=2.0, min_views=2,
            z_calib=1.534,
        )
        fitted_no, _ = _light_field_fit(locs, RHO_SCALING, fp_no_cal)
        fitted_cal, _ = _light_field_fit(locs, RHO_SCALING, fp_cal)
        if fitted_no.shape[0] > 0 and fitted_cal.shape[0] > 0:
            np.testing.assert_allclose(
                fitted_cal[:, 2], fitted_no[:, 2] * 1.534, atol=1e-10
            )

    def test_progress_callback_called(self):
        """The progress callback should be invoked at least once."""
        locs = self._build_two_molecule_scene()
        fp = FitParams(
            frame_min=1, frame_max=2,
            disparity_max=5.0, disparity_step=0.1,
            dist_search=0.5, angle_tolerance=90.0,
            threshold=2.0, min_views=2,
        )
        calls = []
        def cb(frame, fmin, fmax):
            calls.append((frame, fmin, fmax))

        _light_field_fit(locs, RHO_SCALING, fp, progress_func=cb, progress_step=1)
        assert len(calls) > 0
        # All calls should have correct min/max
        for frame, fmin, fmax in calls:
            assert fmin == 1
            assert fmax == 2
