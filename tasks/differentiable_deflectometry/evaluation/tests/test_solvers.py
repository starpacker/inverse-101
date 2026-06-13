"""Tests for solvers module."""
import os

import numpy as np
import pytest
import torch

from src.solvers import LMSolver, setup_diff_parameters, change_parameters
from src.physics_model import get_nested_attr, set_nested_attr

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'fixtures')


class TestLMJacobian:
    """Test Jacobian computation."""

    def test_jacobian_linear(self):
        """Jacobian of a linear function should be the coefficient matrix."""
        fix = np.load(os.path.join(FIXTURES_DIR, 'input_jacobian_linear.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_jacobian_linear.npz'))

        A = torch.Tensor(fix['A'])
        x = torch.Tensor(fix['x'])
        x.requires_grad = True

        def func():
            return A @ x

        solver = LMSolver()
        J = solver.jacobian(func, [x])
        torch.testing.assert_close(J, torch.Tensor(expected['expected_J']), atol=1e-5, rtol=1e-5)

    def test_jacobian_quadratic(self):
        """Jacobian of f(x) = x^2 at x=3 should be 2*3=6."""
        fix = np.load(os.path.join(FIXTURES_DIR, 'input_jacobian_quadratic.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_jacobian_quadratic.npz'))

        x = torch.Tensor(fix['x'])
        x.requires_grad = True

        def func():
            return x**2

        solver = LMSolver()
        J = solver.jacobian(func, [x])
        assert abs(J.item() - float(expected['expected_J'])) < 1e-5


class TestLMOptimization:
    """Test LM convergence on known problems."""

    def test_lm_quadratic_manual(self):
        """LM-style optimization on a simple quadratic using manual gradient descent."""
        fix = np.load(os.path.join(FIXTURES_DIR, 'input_lm_quadratic.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_lm_quadratic.npz'))

        x = torch.Tensor(fix['x_init'])
        target = torch.Tensor(fix['target'])
        lamb = float(fix['lamb'])
        n_iters = int(fix['n_iters'])
        tol = float(expected['tolerance'])

        for it in range(n_iters):
            residual = target - x
            L = torch.mean(residual**2).item()
            if L < 1e-10:
                break
            delta = residual / (1 + lamb)
            x = x + delta

        expected_x = expected['expected_x']
        assert abs(x[0].item() - expected_x[0]) < tol
        assert abs(x[1].item() - expected_x[1]) < tol


class TestParameterManagement:
    """Test parameter get/set/change utilities."""

    def test_change_parameters_add(self):
        fix = np.load(os.path.join(FIXTURES_DIR, 'input_change_params.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_change_params_add.npz'))

        class MockScene:
            def __init__(self):
                self.x = torch.Tensor(fix['x_init'])
                self.y = torch.Tensor(fix['y_init'])

        scene = MockScene()
        result = change_parameters(
            scene, ['x', 'y'],
            [torch.Tensor(fix['delta_x']), torch.Tensor(fix['delta_y'])],
            sign=True
        )
        assert abs(result[0].item() - float(expected['expected_x'])) < 1e-6
        assert abs(result[1].item() - float(expected['expected_y'])) < 1e-6

    def test_change_parameters_subtract(self):
        fix = np.load(os.path.join(FIXTURES_DIR, 'input_change_params.npz'))
        expected = np.load(os.path.join(FIXTURES_DIR, 'output_change_params_subtract.npz'))

        class MockScene:
            def __init__(self):
                self.x = torch.Tensor(fix['x_init'])

        scene = MockScene()
        change_parameters(scene, ['x'], [torch.Tensor(fix['delta_x'])], sign=False)
        assert abs(get_nested_attr(scene, 'x').item() - float(expected['expected_x'])) < 1e-6
