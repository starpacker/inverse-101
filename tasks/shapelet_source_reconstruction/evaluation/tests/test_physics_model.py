"""Unit tests for physics_model.py"""

import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_model import (
    make_grid, image2array, array2image, re_size,
    pre_calc_shapelets, iterate_n1_n2, shapelet_function, shapelet_decomposition,
    shapelet_basis_list, ellipticity2phi_q, spep_deflection, shear_deflection,
    ray_shoot, fwhm2sigma, gaussian_convolve, simulate_image,
    add_poisson_noise, add_background_noise,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures')


class TestGridUtilities:
    def test_make_grid_center(self):
        x, y = make_grid(3, 1.0)
        assert len(x) == 9
        assert np.isclose(np.mean(x), 0.0, atol=1e-14)
        assert np.isclose(np.mean(y), 0.0, atol=1e-14)

    def test_make_grid_shape(self):
        x, y = make_grid(10, 0.5)
        assert len(x) == 100
        assert len(y) == 100

    def test_image2array_array2image_roundtrip(self):
        img = np.random.rand(5, 5)
        recovered = array2image(image2array(img))
        np.testing.assert_array_equal(img, recovered)

    def test_re_size_conservation(self):
        """Block-averaging should preserve total sum scaled by factor^2."""
        img = np.random.rand(10, 10)
        small = re_size(img, 2)
        assert small.shape == (5, 5)
        np.testing.assert_allclose(small.sum() * 4, img.sum(), rtol=1e-10)

    def test_re_size_fixture(self):
        inp = np.load(os.path.join(FIXTURE_DIR, 'input_resize.npz'))
        par = np.load(os.path.join(FIXTURE_DIR, 'param_resize.npz'))
        out = np.load(os.path.join(FIXTURE_DIR, 'output_resize.npz'))
        result = re_size(inp['image'], int(par['factor']))
        np.testing.assert_allclose(result, out['image'], rtol=1e-10)

    def test_make_grid_fixture(self):
        out = np.load(os.path.join(FIXTURE_DIR, 'output_grid_4x4.npz'))
        x, y = make_grid(4, 0.5)
        np.testing.assert_allclose(x, out['x'], rtol=1e-10)
        np.testing.assert_allclose(y, out['y'], rtol=1e-10)


class TestShapelets:
    def test_iterate_n1_n2_count(self):
        indices = list(iterate_n1_n2(5))
        assert len(indices) == 21  # (5+1)*(5+2)/2

    def test_iterate_n1_n2_constraint(self):
        for _, n1, n2 in iterate_n1_n2(10):
            assert n1 + n2 <= 10

    def test_shapelet_orthogonality(self):
        """Shapelet decomposition should recover known coefficients on large grid."""
        x, y = make_grid(64, 0.5)
        n_max = 3
        beta = 3.0
        num_param = (n_max + 1) * (n_max + 2) // 2
        amp = np.array([1.0, -0.5, 0.3, 0.8, -0.2, 0.6, -0.4, 0.1, 0.9, -0.7])
        flux = shapelet_function(x, y, amp, n_max, beta)
        coeff = shapelet_decomposition(flux, x, y, n_max, beta, 0.5)
        np.testing.assert_allclose(coeff, amp, rtol=1e-2, atol=1e-3)

    def test_shapelet_roundtrip(self):
        """Decompose then reconstruct should approximate the original."""
        x, y = make_grid(64, 0.5)
        n_max = 3
        beta = 3.0
        num_param = (n_max + 1) * (n_max + 2) // 2
        amp = np.random.randn(num_param)
        flux = shapelet_function(x, y, amp, n_max, beta)
        coeff = shapelet_decomposition(flux, x, y, n_max, beta, 0.5)
        np.testing.assert_allclose(coeff, amp, rtol=1e-2, atol=1e-3)

    def test_shapelet_basis_list_length(self):
        x, y = make_grid(8, 1.0)
        basis = shapelet_basis_list(x, y, 5, 2.0)
        assert len(basis) == 21

    def test_shapelet_fixture(self):
        inp = np.load(os.path.join(FIXTURE_DIR, 'input_shapelet.npz'))
        par = np.load(os.path.join(FIXTURE_DIR, 'param_shapelet.npz'))
        out = np.load(os.path.join(FIXTURE_DIR, 'output_shapelet.npz'))
        flux = shapelet_function(inp['x'], inp['y'], inp['amp'],
                                  int(par['n_max']), float(par['beta']))
        np.testing.assert_allclose(flux, out['flux'], rtol=1e-10)


class TestLensModels:
    def test_spep_sis_radial(self):
        """SIS: deflection = theta_E * (x,y)/r."""
        x = np.array([1.0, 0.0, 0.6])
        y = np.array([0.0, 1.0, 0.8])
        r = np.sqrt(x**2 + y**2)
        theta_E = 0.5
        ax, ay = spep_deflection(x, y, theta_E, 2.0, 0, 0)
        np.testing.assert_allclose(ax, theta_E * x / r, rtol=1e-5)
        np.testing.assert_allclose(ay, theta_E * y / r, rtol=1e-5)

    def test_spep_fixture(self):
        inp = np.load(os.path.join(FIXTURE_DIR, 'input_spep_sis.npz'))
        par = np.load(os.path.join(FIXTURE_DIR, 'param_spep_sis.npz'))
        out = np.load(os.path.join(FIXTURE_DIR, 'output_spep_sis.npz'))
        ax, ay = spep_deflection(inp['x'], inp['y'],
                                  float(par['theta_E']), float(par['gamma']),
                                  0, 0)
        np.testing.assert_allclose(ax, out['alpha_x'], rtol=1e-10)
        np.testing.assert_allclose(ay, out['alpha_y'], rtol=1e-10)

    def test_shear_zero(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        ax, ay = shear_deflection(x, y, 0.0, 0.0)
        np.testing.assert_array_equal(ax, 0.0)
        np.testing.assert_array_equal(ay, 0.0)

    def test_shear_fixture(self):
        inp = np.load(os.path.join(FIXTURE_DIR, 'input_shear.npz'))
        par = np.load(os.path.join(FIXTURE_DIR, 'param_shear.npz'))
        out = np.load(os.path.join(FIXTURE_DIR, 'output_shear.npz'))
        ax, ay = shear_deflection(inp['x'], inp['y'],
                                   float(par['gamma1']), float(par['gamma2']))
        np.testing.assert_allclose(ax, out['alpha_x'], rtol=1e-10)
        np.testing.assert_allclose(ay, out['alpha_y'], rtol=1e-10)

    def test_ellipticity_spherical(self):
        phi, q = ellipticity2phi_q(0, 0)
        assert np.isclose(q, 1.0)

    def test_ray_shoot_no_deflection(self):
        """Zero lens parameters should give identity mapping."""
        x = np.array([0.5, -0.3])
        y = np.array([0.2, 0.7])
        xs, ys = ray_shoot(x, y,
                           {'theta_E': 0, 'gamma': 2., 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0},
                           {'gamma1': 0, 'gamma2': 0})
        np.testing.assert_allclose(xs, x, atol=1e-6)
        np.testing.assert_allclose(ys, y, atol=1e-6)


class TestPSF:
    def test_fwhm2sigma(self):
        assert np.isclose(fwhm2sigma(1.0), 1.0 / (2 * np.sqrt(2 * np.log(2))))

    def test_gaussian_convolve_preserves_sum(self):
        """Sum is preserved for images large enough relative to the kernel."""
        img = np.zeros((64, 64))
        img[25:40, 25:40] = np.random.rand(15, 15)
        conv = gaussian_convolve(img, 0.1, 0.05)
        np.testing.assert_allclose(conv.sum(), img.sum(), rtol=0.01)

    def test_gaussian_convolve_fixture(self):
        inp = np.load(os.path.join(FIXTURE_DIR, 'input_gaussian_conv.npz'))
        par = np.load(os.path.join(FIXTURE_DIR, 'param_gaussian_conv.npz'))
        out = np.load(os.path.join(FIXTURE_DIR, 'output_gaussian_conv.npz'))
        result = gaussian_convolve(inp['image'], float(par['fwhm']), float(par['pixel_size']))
        np.testing.assert_allclose(result, out['image'], rtol=1e-10)

    def test_no_psf_identity(self):
        img = np.random.rand(8, 8)
        result = gaussian_convolve(img, 0.0, 0.05)
        np.testing.assert_array_equal(result, img)


class TestNoise:
    def test_poisson_noise_shape(self):
        img = np.ones((10, 10)) * 100
        noise = add_poisson_noise(img, 100.0)
        assert noise.shape == (10, 10)

    def test_background_noise_statistics(self):
        np.random.seed(0)
        img = np.zeros((100, 100))
        noise = add_background_noise(img, 5.0)
        assert np.isclose(np.std(noise), 5.0, rtol=0.1)


class TestSimulateImage:
    def test_output_shape(self):
        n_max = 3
        num_p = (n_max + 1) * (n_max + 2) // 2
        amp = np.zeros(num_p)
        amp[0] = 1.0
        kwargs_src = {'n_max': n_max, 'beta': 0.1, 'amp': amp, 'center_x': 0, 'center_y': 0}
        img = simulate_image(8, 0.1, 1, 0, kwargs_src, apply_lens=False, apply_psf=False)
        assert img.shape == (8, 8)

    def test_no_lens_no_psf_positive(self):
        n_max = 0
        amp = np.array([1.0])
        kwargs_src = {'n_max': n_max, 'beta': 0.5, 'amp': amp, 'center_x': 0, 'center_y': 0}
        img = simulate_image(8, 0.1, 1, 0, kwargs_src, apply_lens=False, apply_psf=False)
        assert np.all(img >= 0)
