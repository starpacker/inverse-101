"""Tests for physics_model module."""
import os

import numpy as np
import pytest
import torch

from src.physics_model import (
    Ray, Transformation, Material, Aspheric, normalize,
    rodrigues_rotation_matrix, get_nested_attr, set_nested_attr,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures')


class TestRay:
    def test_ray_evaluation(self):
        """Verify ray(t) = o + t * d."""
        fix = np.load(os.path.join(FIXTURES_DIR, 'input_ray_evaluation.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_ray_evaluation.npz'))

        o = torch.Tensor(fix['o'])
        d = torch.Tensor(fix['d'])
        ray = Ray(o, d, float(fix['wavelength']))
        t = torch.Tensor([float(fix['t'])])
        p = ray(t)
        torch.testing.assert_close(p.squeeze(), torch.Tensor(expected['expected_point']))


class TestTransformation:
    def test_identity(self):
        """Identity transform should not change points."""
        fix = np.load(os.path.join(FIXTURES_DIR, 'input_transformation.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_transformation.npz'))

        T = Transformation(torch.Tensor(fix['identity_R']), torch.Tensor(fix['identity_t']))
        p = torch.Tensor(fix['test_point'])
        torch.testing.assert_close(T.transform_point(p), torch.Tensor(expected['identity_expected']))

    def test_inverse(self):
        """T.inverse() should undo the transform."""
        fix = np.load(os.path.join(FIXTURES_DIR, 'input_transformation.npz'))

        R = rodrigues_rotation_matrix(
            torch.Tensor(fix['rotation_axis']), torch.Tensor([float(fix['rotation_angle'])])
        )
        t = torch.Tensor(fix['translation'])
        T = Transformation(R, t)
        T_inv = T.inverse()
        p = torch.Tensor(fix['inverse_test_point'])
        p_transformed = T.transform_point(p)
        p_recovered = T_inv.transform_point(p_transformed)
        torch.testing.assert_close(p_recovered, p, atol=1e-5, rtol=1e-5)


class TestMaterial:
    def test_nbk7_ior(self):
        """Verify N-BK7 IOR at 589.3nm is approximately 1.5168."""
        fix = np.load(os.path.join(FIXTURES_DIR, 'input_material.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_material.npz'))

        mat = Material('N-BK7')
        ior = mat.ior(torch.Tensor([float(fix['nbk7_wavelength'])]))
        assert abs(ior - float(expected['nbk7_ior_approx'])) < float(expected['nbk7_tolerance'])

    def test_air_ior(self):
        """Air IOR should be very close to 1."""
        fix = np.load(os.path.join(FIXTURES_DIR, 'input_material.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_material.npz'))

        mat = Material('air')
        ior = mat.ior(torch.Tensor([float(fix['air_wavelength'])]))
        assert abs(ior - float(expected['air_ior_approx'])) < float(expected['air_tolerance'])


class TestAspheric:
    def test_flat_surface(self):
        """Aspheric with c=0 should be flat (z=0)."""
        params = np.load(os.path.join(FIXTURES_DIR, 'param_aspheric_flat.npz'))
        inputs = np.load(os.path.join(FIXTURES_DIR, 'input_aspheric_flat.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_aspheric_flat.npz'))

        surface = Aspheric(
            r=float(params['r']), d=float(params['d']),
            c=float(params['c']), k=float(params['k'])
        )
        x = torch.Tensor(inputs['x'])
        y = torch.Tensor(inputs['y'])
        z = surface.surface(x, y)
        torch.testing.assert_close(z, torch.Tensor(expected['expected_z']), atol=1e-10, rtol=1e-10)

    def test_spherical_surface(self):
        """Aspheric with k=0 should be spherical."""
        params = np.load(os.path.join(FIXTURES_DIR, 'param_aspheric_spherical.npz'))
        inputs = np.load(os.path.join(FIXTURES_DIR, 'input_aspheric_spherical.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_aspheric_spherical.npz'))

        surface = Aspheric(
            r=float(params['r']), d=float(params['d']),
            c=float(params['c']), k=float(params['k'])
        )
        z0 = surface.surface(
            torch.Tensor([float(inputs['x_origin'])]),
            torch.Tensor([float(inputs['y_origin'])])
        )
        assert abs(z0.item()) < 1e-10

        z5 = surface.surface(
            torch.Tensor([float(inputs['x_test'])]),
            torch.Tensor([float(inputs['y_test'])])
        )
        assert abs(z5.item() - float(expected['z_at_x5'])) < 1e-6

    def test_ray_intersection(self):
        """Verify ray-surface intersection for a simple case."""
        params = np.load(os.path.join(FIXTURES_DIR, 'param_aspheric_intersection.npz'))
        inputs = np.load(os.path.join(FIXTURES_DIR, 'input_aspheric_intersection.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_aspheric_intersection.npz'))

        surface = Aspheric(
            r=float(params['r']), d=float(params['d']),
            c=float(params['c']), k=float(params['k'])
        )
        o = torch.Tensor(inputs['o'])
        d = torch.Tensor(inputs['d'])
        ray = Ray(o, d, 500.0)
        valid, p = surface.newtons_method(1e5, o, d)
        assert valid.all()
        assert abs(p[0, 2].item() - float(expected['expected_z_approx'])) < float(expected['z_tolerance'])


class TestNestedAttr:
    def test_get_set_simple(self):
        """Test simple nested attribute access."""
        class Inner:
            def __init__(self):
                self.val = 42

        class Outer:
            def __init__(self):
                self.inner = Inner()

        obj = Outer()
        assert get_nested_attr(obj, 'inner.val') == 42
        set_nested_attr(obj, 'inner.val', 99)
        assert get_nested_attr(obj, 'inner.val') == 99

    def test_get_set_indexed(self):
        """Test indexed attribute access like surfaces[0].c."""
        class Surface:
            def __init__(self, c):
                self.c = c

        class Group:
            def __init__(self):
                self.surfaces = [Surface(1.0), Surface(2.0)]

        obj = Group()
        assert get_nested_attr(obj, 'surfaces[0].c') == 1.0
        assert get_nested_attr(obj, 'surfaces[1].c') == 2.0
        set_nested_attr(obj, 'surfaces[0].c', 3.0)
        assert get_nested_attr(obj, 'surfaces[0].c') == 3.0
