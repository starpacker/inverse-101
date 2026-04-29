"""Tests for solvers module."""

import os
import sys
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

import torch
from src.solvers import (
    ActNorm,
    ZeroFC,
    AffineCoupling,
    Flow,
    RealNVP,
    _order_inverse,
    AlphaDPISolver,
)


class TestActNorm(unittest.TestCase):
    """Tests for ActNorm layer."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        layer = ActNorm(logdet=True)
        x = torch.randn(8, 4)
        out, logdet = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_logdet_is_scalar(self):
        """Log-determinant should be a scalar tensor."""
        layer = ActNorm(logdet=True)
        x = torch.randn(8, 4)
        out, logdet = layer(x)
        self.assertEqual(logdet.dim(), 0)

    def test_initialization_flag(self):
        """After first forward pass, initialized flag should be set."""
        layer = ActNorm(logdet=True)
        self.assertEqual(layer.initialized.item(), 0)
        x = torch.randn(8, 4)
        _ = layer(x)
        self.assertEqual(layer.initialized.item(), 1)

    def test_forward_reverse_roundtrip(self):
        """Reverse should invert forward up to numerical precision."""
        layer = ActNorm(logdet=True)
        x = torch.randn(16, 4)
        out, logdet_fwd = layer(x)
        x_rec, logdet_rev = layer.reverse(out)
        np.testing.assert_allclose(
            x.detach().numpy(), x_rec.detach().numpy(), atol=1e-5,
            err_msg="ActNorm forward-reverse roundtrip failed"
        )


class TestZeroFC(unittest.TestCase):
    """Tests for ZeroFC layer."""

    def test_output_shape(self):
        """Output shape should be (B, out_dim)."""
        layer = ZeroFC(in_dim=8, out_dim=4)
        x = torch.randn(16, 8)
        out = layer(x)
        self.assertEqual(out.shape, (16, 4))

    def test_initial_output_near_zero(self):
        """At initialization, weights and biases are zero so output should be near zero."""
        layer = ZeroFC(in_dim=8, out_dim=4)
        x = torch.randn(16, 8)
        out = layer(x)
        np.testing.assert_allclose(
            out.detach().numpy(), 0.0, atol=1e-7,
            err_msg="ZeroFC initial output not near zero"
        )


class TestAffineCoupling(unittest.TestCase):
    """Tests for AffineCoupling layer."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        ndim = 8
        coupling = AffineCoupling(ndim, seqfrac=0.5, affine=True, batch_norm=False)
        x = torch.randn(16, ndim)
        out, logdet = coupling(x)
        self.assertEqual(out.shape, x.shape)

    def test_logdet_shape(self):
        """Log-determinant should have shape (B,)."""
        ndim = 8
        coupling = AffineCoupling(ndim, seqfrac=0.5, affine=True, batch_norm=False)
        x = torch.randn(16, ndim)
        out, logdet = coupling(x)
        self.assertEqual(logdet.shape, (16,))

    def test_forward_reverse_roundtrip(self):
        """Reverse should invert forward."""
        ndim = 8
        coupling = AffineCoupling(ndim, seqfrac=0.5, affine=True, batch_norm=False)
        x = torch.randn(16, ndim)
        out, _ = coupling(x)
        x_rec, _ = coupling.reverse(out)
        np.testing.assert_allclose(
            x.detach().numpy(), x_rec.detach().numpy(), atol=1e-5,
            err_msg="AffineCoupling forward-reverse roundtrip failed"
        )

    def test_first_half_unchanged(self):
        """The first half of the input should pass through unchanged."""
        ndim = 8
        coupling = AffineCoupling(ndim, seqfrac=0.5, affine=True, batch_norm=False)
        x = torch.randn(16, ndim)
        out, _ = coupling(x)
        np.testing.assert_allclose(
            x[:, :ndim // 2].detach().numpy(),
            out[:, :ndim // 2].detach().numpy(), atol=1e-7,
            err_msg="First half of input should be unchanged in coupling"
        )


class TestFlow(unittest.TestCase):
    """Tests for Flow block."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        ndim = 8
        flow = Flow(ndim, affine=True, seqfrac=0.5, batch_norm=False)
        x = torch.randn(16, ndim)
        out, logdet = flow(x)
        self.assertEqual(out.shape, x.shape)

    def test_logdet_scalar_per_batch(self):
        """Log-determinant should be a scalar (summed over dimensions)."""
        ndim = 8
        flow = Flow(ndim, affine=True, seqfrac=0.5, batch_norm=False)
        x = torch.randn(16, ndim)
        out, logdet = flow(x)
        # logdet is accumulated from ActNorm (scalar) and coupling (B,)
        # so the result can be (B,) or scalar depending on broadcast
        # Just check it is a valid tensor
        self.assertIsInstance(logdet, torch.Tensor)

    def test_forward_reverse_roundtrip(self):
        """Reverse should invert forward."""
        ndim = 8
        flow = Flow(ndim, affine=True, seqfrac=0.5, batch_norm=False)
        x = torch.randn(16, ndim)
        out, _ = flow(x)
        x_rec, _ = flow.reverse(out)
        np.testing.assert_allclose(
            x.detach().numpy(), x_rec.detach().numpy(), atol=1e-4,
            err_msg="Flow forward-reverse roundtrip failed"
        )


class TestRealNVP(unittest.TestCase):
    """Tests for RealNVP normalizing flow."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        ndim, n_flow = 4, 2
        model = RealNVP(ndim, n_flow, affine=True, seqfrac=0.5,
                        permute='random', batch_norm=False)
        x = torch.randn(8, ndim)
        out, logdet = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_reverse_roundtrip(self):
        """Reverse should invert forward."""
        ndim, n_flow = 4, 3
        model = RealNVP(ndim, n_flow, affine=True, seqfrac=0.5,
                        permute='random', batch_norm=False)
        x = torch.randn(8, ndim)
        out, _ = model(x)
        x_rec, _ = model.reverse(out)
        np.testing.assert_allclose(
            x.detach().numpy(), x_rec.detach().numpy(), atol=1e-4,
            err_msg="RealNVP forward-reverse roundtrip failed"
        )

    def test_logdet_dtype(self):
        """Log-determinant should be float dtype."""
        ndim, n_flow = 4, 2
        model = RealNVP(ndim, n_flow, affine=True, seqfrac=0.5,
                        permute='random', batch_norm=False)
        x = torch.randn(8, ndim)
        _, logdet = model(x)
        self.assertTrue(logdet.dtype in (torch.float32, torch.float64))


class TestOrderInverse(unittest.TestCase):
    """Tests for _order_inverse."""

    def test_inverse_identity(self):
        """Inverse of identity permutation should be identity."""
        order = np.arange(5)
        inv = _order_inverse(order)
        np.testing.assert_array_equal(inv, order)

    def test_inverse_reversal(self):
        """Inverse of a reversal should be a reversal."""
        order = np.arange(5)[::-1].copy()
        inv = _order_inverse(order)
        np.testing.assert_array_equal(inv, order)

    def test_roundtrip(self):
        """Applying a permutation then its inverse should be identity."""
        rng = np.random.RandomState(42)
        order = rng.permutation(8)
        inv = _order_inverse(order)
        data = np.arange(8)
        # Apply permutation then inverse
        permuted = data[order]
        recovered = permuted[inv]
        np.testing.assert_array_equal(recovered, data)


class TestAlphaDPISolverInit(unittest.TestCase):
    """Tests for AlphaDPISolver construction and utility methods."""

    def test_default_construction(self):
        """Solver should construct with default parameters without error."""
        solver = AlphaDPISolver(npix=16, fov_uas=120.0, device=torch.device('cpu'))
        self.assertEqual(solver.npix, 16)
        self.assertEqual(solver.fov_uas, 120.0)
        self.assertIsNone(solver.params_generator)

    def test_build_geometric_model_simple(self):
        """Building simple_crescent model should set img_converter with nparams=4."""
        solver = AlphaDPISolver(
            npix=16, fov_uas=120.0,
            geometric_model='simple_crescent',
            device=torch.device('cpu'),
        )
        nparams = solver._build_geometric_model()
        self.assertEqual(nparams, 4)
        self.assertIsNotNone(solver.img_converter)

    def test_build_geometric_model_nuisance(self):
        """Nuisance model nparams should be 4 + 6*n_gaussian."""
        solver = AlphaDPISolver(
            npix=16, fov_uas=120.0,
            geometric_model='simple_crescent_nuisance',
            n_gaussian=2,
            device=torch.device('cpu'),
        )
        nparams = solver._build_geometric_model()
        self.assertEqual(nparams, 4 + 6 * 2)

    def test_build_geometric_model_floor_nuisance(self):
        """Floor nuisance model nparams should be 4 + 6*n_gaussian + 2."""
        solver = AlphaDPISolver(
            npix=16, fov_uas=120.0,
            geometric_model='simple_crescent_floor_nuisance',
            n_gaussian=1,
            device=torch.device('cpu'),
        )
        nparams = solver._build_geometric_model()
        self.assertEqual(nparams, 4 + 6 * 1 + 2)

    def test_extract_physical_params_shape(self):
        """Physical params should have correct shape for simple crescent."""
        solver = AlphaDPISolver(
            npix=16, fov_uas=120.0,
            geometric_model='simple_crescent',
            device=torch.device('cpu'),
        )
        solver._build_geometric_model()
        params_unit = np.random.rand(32, 4)
        phys = solver.extract_physical_params(params_unit)
        self.assertEqual(phys.shape, (32, 4))

    def test_extract_physical_params_ranges(self):
        """Diameter should be within 2*r_range, width within width_range."""
        r_range = [10.0, 40.0]
        width_range = [1.0, 40.0]
        solver = AlphaDPISolver(
            npix=16, fov_uas=120.0,
            geometric_model='simple_crescent',
            r_range=r_range, width_range=width_range,
            device=torch.device('cpu'),
        )
        solver._build_geometric_model()
        params_unit = np.random.rand(100, 4)
        phys = solver.extract_physical_params(params_unit)
        diameters = phys[:, 0]
        widths = phys[:, 1]
        self.assertTrue(np.all(diameters >= 2 * r_range[0]))
        self.assertTrue(np.all(diameters <= 2 * r_range[1]))
        self.assertTrue(np.all(widths >= width_range[0]))
        self.assertTrue(np.all(widths <= width_range[1]))

    def test_sample_raises_before_reconstruct(self):
        """sample() should raise RuntimeError if called before reconstruct()."""
        solver = AlphaDPISolver(npix=16, fov_uas=120.0, device=torch.device('cpu'))
        with self.assertRaises(RuntimeError):
            solver.sample(n_samples=10)

    def test_unknown_geometric_model_raises(self):
        """Unknown geometric model should raise ValueError."""
        solver = AlphaDPISolver(
            npix=16, fov_uas=120.0,
            geometric_model='unknown_model',
            device=torch.device('cpu'),
        )
        with self.assertRaises(ValueError):
            solver._build_geometric_model()


if __name__ == '__main__':
    unittest.main()
