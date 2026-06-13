"""Tests for physics_model module (FLFM optics, MLA, lens assignment, alpha)."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics_model import (
    FourierMicroscope,
    MicroLensArray,
    Localisations,
    build_microscope,
    build_mla,
    assign_to_lenses,
    compute_alpha_model,
    _generate_lattice,
    _rotate_lattice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_meta():
    """Minimal meta_data dict for building a microscope and MLA."""
    return {
        "num_aperture": 1.27,
        "mla_lens_pitch": 222.0,
        "focal_length_mla": 5.556,
        "focal_length_obj_lens": 3.333,
        "focal_length_tube_lens": 200.0,
        "focal_length_fourier_lens": 150.0,
        "pixel_size_camera": 6.5,
        "ref_idx_immersion": 1.406,
        "ref_idx_medium": 1.33,
        "mla_type": "HEXAGONAL",
        "mla_optic_size": 1500.0,
        "mla_centre": [0.0, 0.0],
        "mla_rotation": 30.0,
        "alpha_model": "SPHERE",
    }


def _make_locs_2d_csv(n=32, rng=None):
    """Create synthetic (N,8) localisation array."""
    if rng is None:
        rng = np.random.default_rng(0)
    locs = np.zeros((n, 8))
    locs[:, 0] = rng.integers(1, 5, n)       # frame
    locs[:, 1] = rng.uniform(-5, 5, n)       # X um
    locs[:, 2] = rng.uniform(-5, 5, n)       # Y um
    locs[:, 3] = rng.uniform(0.05, 0.2, n)   # sigma_X
    locs[:, 4] = rng.uniform(0.05, 0.2, n)   # sigma_Y
    locs[:, 5] = rng.uniform(100, 5000, n)   # intensity
    locs[:, 6] = rng.uniform(10, 100, n)     # background
    locs[:, 7] = rng.uniform(0.01, 0.1, n)   # precision
    return locs


# ---------------------------------------------------------------------------
# Tests: build_microscope
# ---------------------------------------------------------------------------

class TestBuildMicroscope:
    """Tests for build_microscope derived quantities."""

    def test_return_type(self):
        lfm = build_microscope(_default_meta())
        assert isinstance(lfm, FourierMicroscope)

    def test_bfp_radius_formula(self):
        meta = _default_meta()
        lfm = build_microscope(meta)
        expected = 1000.0 * meta["num_aperture"] * meta["focal_length_obj_lens"] * (
            meta["focal_length_fourier_lens"] / meta["focal_length_tube_lens"]
        )
        np.testing.assert_allclose(lfm.bfp_radius, expected, rtol=1e-12)

    def test_magnification_formula(self):
        meta = _default_meta()
        lfm = build_microscope(meta)
        expected = (meta["focal_length_tube_lens"] / meta["focal_length_obj_lens"]) * (
            meta["focal_length_mla"] / meta["focal_length_fourier_lens"]
        )
        np.testing.assert_allclose(lfm.magnification, expected, rtol=1e-12)

    def test_pixel_size_sample_positive(self):
        lfm = build_microscope(_default_meta())
        assert lfm.pixel_size_sample > 0

    def test_rho_scaling_identity(self):
        """rho_scaling should equal magnification / bfp_radius."""
        lfm = build_microscope(_default_meta())
        np.testing.assert_allclose(
            lfm.rho_scaling, lfm.magnification / lfm.bfp_radius, rtol=1e-12
        )

    def test_mla_to_uv_scale(self):
        """mla_to_uv_scale = 2 / bfp_lens_count."""
        lfm = build_microscope(_default_meta())
        np.testing.assert_allclose(
            lfm.mla_to_uv_scale, 2.0 / lfm.bfp_lens_count, rtol=1e-12
        )


# ---------------------------------------------------------------------------
# Tests: MLA lattice generation
# ---------------------------------------------------------------------------

class TestGenerateLattice:
    """Tests for _generate_lattice and build_mla."""

    def test_square_lattice_shape(self):
        pts = _generate_lattice("SQUARE", optic_size=500.0, lens_pitch=100.0)
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        assert pts.shape[0] > 0

    def test_hexagonal_lattice_shape(self):
        pts = _generate_lattice("HEXAGONAL", optic_size=500.0, lens_pitch=100.0)
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        assert pts.shape[0] > 0

    def test_invalid_lattice_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _generate_lattice("TRIANGULAR", optic_size=500.0, lens_pitch=100.0)

    def test_build_mla_returns_correct_type(self):
        mla = build_mla(_default_meta())
        assert isinstance(mla, MicroLensArray)
        assert mla.lens_centres.shape[1] == 2


# ---------------------------------------------------------------------------
# Tests: _rotate_lattice
# ---------------------------------------------------------------------------

class TestRotateLattice:
    """Tests for _rotate_lattice in-place rotation."""

    def test_zero_rotation_identity(self):
        mla = MicroLensArray(
            lattice_type="HEXAGONAL", focal_length=5.0, lens_pitch=200.0,
            optic_size=1000.0, centre=np.array([0.0, 0.0]),
            lens_centres=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        )
        original = mla.lens_centres.copy()
        _rotate_lattice(mla, 0.0)
        np.testing.assert_allclose(mla.lens_centres, original, atol=1e-14)

    def test_90_degree_rotation(self):
        mla = MicroLensArray(
            lattice_type="SQUARE", focal_length=5.0, lens_pitch=200.0,
            optic_size=1000.0, centre=np.array([0.0, 0.0]),
            lens_centres=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        _rotate_lattice(mla, np.pi / 2)
        np.testing.assert_allclose(mla.lens_centres[0], [0.0, 1.0], atol=1e-14)
        np.testing.assert_allclose(mla.lens_centres[1], [-1.0, 0.0], atol=1e-14)

    def test_rotation_preserves_distance(self):
        mla = MicroLensArray(
            lattice_type="HEXAGONAL", focal_length=5.0, lens_pitch=200.0,
            optic_size=1000.0, centre=np.array([0.0, 0.0]),
            lens_centres=np.array([[3.0, 4.0], [-2.0, 1.0]]),
        )
        dist_before = np.sqrt(np.sum(mla.lens_centres ** 2, axis=1))
        _rotate_lattice(mla, np.deg2rad(45))
        dist_after = np.sqrt(np.sum(mla.lens_centres ** 2, axis=1))
        np.testing.assert_allclose(dist_after, dist_before, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: Localisations and assign_to_lenses
# ---------------------------------------------------------------------------

class TestLocalisationsAndAssign:
    """Tests for the Localisations class and lens assignment."""

    def test_localisations_shape(self):
        locs_csv = _make_locs_2d_csv(16)
        lfl = Localisations(locs_csv)
        assert lfl.locs_2d.shape == (16, 13)
        assert lfl.filtered_locs_2d.shape == (16, 13)

    def test_assign_populates_uv(self):
        meta = _default_meta()
        lfm = build_microscope(meta)
        mla = build_mla(meta)
        locs_csv = _make_locs_2d_csv(32)
        lfl = assign_to_lenses(locs_csv, mla, lfm)
        # U and V columns should be non-trivially filled (at least some nonzero)
        assert np.any(lfl.locs_2d[:, 1] != 0) or np.any(lfl.locs_2d[:, 2] != 0)
        # Lens index column should have valid integer indices
        assert np.all(lfl.locs_2d[:, 12] >= 0)
        assert np.all(lfl.locs_2d[:, 12] < mla.lens_centres.shape[0])

    def test_filter_lenses_reduces_count(self):
        """Filtering by BFP radius should keep only interior lenses."""
        meta = _default_meta()
        lfm = build_microscope(meta)
        mla = build_mla(meta)
        locs_csv = _make_locs_2d_csv(64)
        lfl = assign_to_lenses(locs_csv, mla, lfm)
        n_before = lfl.filtered_locs_2d.shape[0]
        lfl.filter_lenses(mla, lfm)
        n_after = lfl.filtered_locs_2d.shape[0]
        assert n_after <= n_before

    def test_filter_rhos(self):
        meta = _default_meta()
        lfm = build_microscope(meta)
        mla = build_mla(meta)
        locs_csv = _make_locs_2d_csv(64)
        lfl = assign_to_lenses(locs_csv, mla, lfm)
        lfl.filter_rhos((0.0, 0.5))
        uv = lfl.filtered_locs_2d[:, 1:3]
        rhos = np.sqrt(np.sum(uv ** 2, axis=1))
        assert np.all(rhos <= 0.5 + 1e-12)

    def test_filter_photons(self):
        locs_csv = _make_locs_2d_csv(32)
        lfl = Localisations(locs_csv)
        lfl.filter_photons((500.0, 3000.0))
        ph = lfl.filtered_locs_2d[:, 7]
        assert np.all(ph >= 500.0)
        assert np.all(ph <= 3000.0)

    def test_filter_spot_sizes(self):
        locs_csv = _make_locs_2d_csv(32)
        lfl = Localisations(locs_csv)
        lfl.filter_spot_sizes((0.08, 0.15))
        s = lfl.filtered_locs_2d[:, 5]
        assert np.all(s >= 0.08)
        assert np.all(s <= 0.15)


# ---------------------------------------------------------------------------
# Tests: Alpha model
# ---------------------------------------------------------------------------

class TestAlphaModel:
    """Tests for compute_alpha_model (LINEAR, SPHERE, INTEGRATE_SPHERE)."""

    def _setup_lfl(self):
        meta = _default_meta()
        lfm = build_microscope(meta)
        mla = build_mla(meta)
        locs_csv = _make_locs_2d_csv(32)
        lfl = assign_to_lenses(locs_csv, mla, lfm)
        lfl.filter_lenses(mla, lfm)
        return lfl, lfm

    def test_linear_alpha_equals_uv(self):
        lfl, lfm = self._setup_lfl()
        compute_alpha_model(lfl, lfm, model="LINEAR")
        uv = lfl.filtered_locs_2d[:, 1:3]
        alpha_uv = lfl.filtered_locs_2d[:, 10:12]
        np.testing.assert_allclose(alpha_uv, uv, atol=1e-14)

    def test_sphere_alpha_shape(self):
        lfl, lfm = self._setup_lfl()
        compute_alpha_model(lfl, lfm, model="SPHERE")
        alpha_uv = lfl.filtered_locs_2d[:, 10:12]
        assert alpha_uv.shape[1] == 2

    def test_integrate_sphere_alpha_finite(self):
        lfl, lfm = self._setup_lfl()
        compute_alpha_model(lfl, lfm, model="INTEGRATE_SPHERE")
        alpha_uv = lfl.filtered_locs_2d[:, 10:12]
        # Alpha values should be finite where views are inside the pupil
        assert np.all(np.isfinite(alpha_uv))

    def test_invalid_alpha_model_raises(self):
        lfl, lfm = self._setup_lfl()
        with pytest.raises(ValueError, match="Unknown alpha model"):
            lfl.compute_alpha_model(lfm, model="BOGUS")

    def test_sphere_alpha_sign(self):
        """Sphere alpha should have the same sign pattern as u,v (negative prefactor)."""
        lfl, lfm = self._setup_lfl()
        compute_alpha_model(lfl, lfm, model="SPHERE")
        uv = lfl.filtered_locs_2d[:, 1:3]
        alpha_uv = lfl.filtered_locs_2d[:, 10:12]
        # The prefactor -(NA/n)/sqrt(...) is negative, so alpha and uv have opposite signs
        # where uv != 0
        nonzero = np.abs(uv) > 1e-8
        if np.any(nonzero):
            # For components where u>0, alpha_u should be <0 (and vice versa)
            signs_uv = np.sign(uv[nonzero])
            signs_alpha = np.sign(alpha_uv[nonzero])
            assert np.all(signs_uv * signs_alpha < 0), (
                "SPHERE alpha should have opposite sign to (u,v) due to negative prefactor"
            )

    def test_correct_xy(self):
        """correct_xy should subtract per-view offsets from X,Y."""
        locs_csv = _make_locs_2d_csv(8)
        lfl = Localisations(locs_csv)
        # All locs have U=0.5, V=0.3
        lfl.filtered_locs_2d[:, 1] = 0.5
        lfl.filtered_locs_2d[:, 2] = 0.3
        x_before = lfl.filtered_locs_2d[:, 3].copy()
        y_before = lfl.filtered_locs_2d[:, 4].copy()
        correction = np.array([[0.5, 0.3, 0.1, -0.2, 8]])
        lfl.correct_xy(correction)
        np.testing.assert_allclose(lfl.corrected_locs_2d[:, 3], x_before - 0.1, atol=1e-14)
        np.testing.assert_allclose(lfl.corrected_locs_2d[:, 4], y_before + 0.2, atol=1e-14)
