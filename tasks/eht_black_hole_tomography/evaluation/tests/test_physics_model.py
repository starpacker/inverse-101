"""
Tests for src/physics_model.py
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics_model import (
    rotation_matrix, rotation_matrix_torch, keplerian_omega,
    velocity_warp_coords, fill_unsupervised, trilinear_interpolate,
    volume_render, dft_matrix,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'physics_model')


class TestRotationMatrix(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'rotation_matrix.npz'))

    def test_output_shape(self):
        R = rotation_matrix(self.f['input_axis'], float(self.f['input_angle']))
        self.assertEqual(R.shape, (3, 3))

    def test_output_values(self):
        R = rotation_matrix(self.f['input_axis'], float(self.f['input_angle']))
        np.testing.assert_allclose(R, self.f['output_matrix'], rtol=1e-10)

    def test_identity(self):
        R = rotation_matrix([0, 0, 1], 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_orthogonal(self):
        R = rotation_matrix([1, 1, 0], 1.23)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_determinant(self):
        R = rotation_matrix([0.3, 0.5, 0.8], 2.1)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)


class TestRotationMatrixBatched(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'rotation_matrix_batched.npz'))

    def test_output_shape(self):
        R = rotation_matrix(self.f['input_axis'], self.f['input_angles'])
        self.assertEqual(R.shape, (3, 3, 3))

    def test_output_values(self):
        R = rotation_matrix(self.f['input_axis'], self.f['input_angles'])
        np.testing.assert_allclose(R, self.f['output_matrix'], rtol=1e-10)


class TestKeplerianOmega(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'keplerian_omega.npz'))

    def test_output_values(self):
        Omega = keplerian_omega(self.f['input_r'], spin=float(self.f['param_spin']))
        np.testing.assert_allclose(Omega, self.f['output_Omega'], rtol=1e-10)

    def test_decreasing_with_radius(self):
        r = np.array([6.0, 8.0, 10.0, 12.0])
        Omega = keplerian_omega(r)
        self.assertTrue(np.all(np.diff(Omega) < 0))

    def test_analytical_schwarzschild(self):
        r = np.array([10.0])
        Omega = keplerian_omega(r, spin=0.0, M=1.0)
        expected = 1.0 / (10.0 ** 1.5)
        np.testing.assert_allclose(Omega, expected, rtol=1e-10)


class TestVelocityWarpCoords(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'velocity_warp.npz'))

    def test_output_shape(self):
        coords = torch.tensor(self.f['input_coords'])
        Omega = torch.tensor(self.f['input_Omega'])
        t_geo = torch.tensor(self.f['input_t_geo'])
        warped = velocity_warp_coords(
            coords, Omega, float(self.f['param_t_frame']),
            float(self.f['param_t_start_obs']), t_geo,
            float(self.f['param_t_injection'])
        )
        self.assertEqual(warped.shape, (4, 4, 5, 3))

    def test_output_values(self):
        coords = torch.tensor(self.f['input_coords'])
        Omega = torch.tensor(self.f['input_Omega'])
        t_geo = torch.tensor(self.f['input_t_geo'])
        warped = velocity_warp_coords(
            coords, Omega, float(self.f['param_t_frame']),
            float(self.f['param_t_start_obs']), t_geo,
            float(self.f['param_t_injection'])
        )
        np.testing.assert_allclose(warped.numpy(), self.f['output_warped'],
                                   rtol=1e-5, atol=1e-6)

    def test_zero_time_identity(self):
        coords = torch.randn(3, 4, 4, 5)
        Omega = torch.ones(4, 4, 5) * 0.01
        warped = velocity_warp_coords(
            coords, Omega, t_frame=0.0, t_start_obs=0.0,
            t_geo=torch.zeros(4, 4, 5), t_injection=0.0
        )
        np.testing.assert_allclose(
            warped.numpy(),
            coords.permute(1, 2, 3, 0).numpy(),
            atol=1e-6
        )


class TestFillUnsupervised(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'fill_unsupervised.npz'))

    def test_output_values(self):
        emission = torch.tensor(self.f['input_emission'])
        coords = torch.tensor(self.f['input_coords'])
        result = fill_unsupervised(emission, coords,
                                   rmin=float(self.f['param_rmin']),
                                   rmax=float(self.f['param_rmax']),
                                   z_width=float(self.f['param_z_width']))
        np.testing.assert_allclose(result.numpy(), self.f['output_emission'],
                                   rtol=1e-10)

    def test_zeros_inside_rmin(self):
        emission = torch.ones(5, 5, 5)
        coords = torch.zeros(3, 5, 5, 5)  # All at origin
        result = fill_unsupervised(emission, coords, rmin=1.0)
        self.assertEqual(result.sum().item(), 0.0)


class TestTrilinearInterpolate(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'trilinear_interpolate.npz'))

    def test_output_values(self):
        volume = torch.tensor(self.f['input_volume'])
        coords = torch.tensor(self.f['input_coords'])
        result = trilinear_interpolate(volume, coords,
                                       float(self.f['param_fov_min']),
                                       float(self.f['param_fov_max']))
        np.testing.assert_allclose(result.numpy(), self.f['output_values'],
                                   rtol=1e-5, atol=1e-6)

    def test_center_value(self):
        volume = torch.ones(8, 8, 8) * 3.0
        coords = torch.tensor([[0.0, 0.0, 0.0]])
        result = trilinear_interpolate(volume, coords, -4.0, 4.0)
        np.testing.assert_allclose(result.numpy(), [3.0], rtol=1e-5)


class TestVolumeRender(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'volume_render.npz'))

    def test_output_shape(self):
        emission = torch.tensor(self.f['input_emission'])
        g = torch.tensor(self.f['input_g'])
        dtau = torch.tensor(self.f['input_dtau'])
        Sigma = torch.tensor(self.f['input_Sigma'])
        result = volume_render(emission, g, dtau, Sigma)
        self.assertEqual(result.shape, (4, 4))

    def test_output_values(self):
        emission = torch.tensor(self.f['input_emission'])
        g = torch.tensor(self.f['input_g'])
        dtau = torch.tensor(self.f['input_dtau'])
        Sigma = torch.tensor(self.f['input_Sigma'])
        result = volume_render(emission, g, dtau, Sigma)
        np.testing.assert_allclose(result.numpy(), self.f['output_image'],
                                   rtol=1e-5)


class TestDFTMatrix(unittest.TestCase):

    def setUp(self):
        self.f = np.load(os.path.join(FIXTURE_DIR, 'dft_matrix.npz'))

    def test_output_shape(self):
        A = dft_matrix(self.f['input_uv_coords'],
                       float(self.f['param_fov_rad']),
                       int(self.f['param_npix']))
        self.assertEqual(A.shape, (3, 16))

    def test_output_values(self):
        A = dft_matrix(self.f['input_uv_coords'],
                       float(self.f['param_fov_rad']),
                       int(self.f['param_npix']))
        np.testing.assert_allclose(A, self.f['output_A'], rtol=1e-10)

    def test_unit_modulus(self):
        uv = np.array([[1e9, 0.0]])
        A = dft_matrix(uv, 1e-10, 4)
        np.testing.assert_allclose(np.abs(A), 1.0, rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
