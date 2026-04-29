"""Tests for src/solvers.py (includes network architecture and solver)"""
import os
import unittest
import numpy as np
import torch

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR_SOLVERS = os.path.join(TASK_DIR, "evaluation", "fixtures", "solvers")
FIXTURE_DIR_NETWORK = os.path.join(TASK_DIR, "evaluation", "fixtures", "network")

import sys
sys.path.insert(0, TASK_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Network Architecture Tests
# ──────────────────────────────────────────────────────────────────────────────


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFullModelForward(unittest.TestCase):
    """
    Fixture: full_model_forward.npz
        output_amplitude_crop  : (64, 64) float32
        output_phase_crop      : (64, 64) float32
        output_amplitude_shape : (4,) int
        output_amplitude_mean  : float
        output_phase_mean      : float
    """

    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR_NETWORK, "full_model_forward.npz"),
            allow_pickle=True,
        )
        from src.solvers import FullModel, load_model_with_required_grad

        device = "cuda:0"
        self.model = FullModel(
            w=2048, h=2048, num_feats=32, x_mode=512, y_mode=512,
            z_min=-20.0, z_max=20.0, ds_factor=1, use_layernorm=False,
        ).to(device)
        load_model_with_required_grad(
            self.model,
            os.path.join(TASK_DIR, "evaluation", "reference_outputs", "model_weights.pth"),
        )
        self.model.eval()

        with torch.no_grad():
            self.ampli, self.phase = self.model(torch.tensor([0.0]).to(device))

    def test_output_shape(self):
        expected = tuple(self.fixture["output_amplitude_shape"])
        self.assertEqual(self.ampli.shape, torch.Size(expected))

    def test_amplitude_crop(self):
        crop = slice(512, 576)
        result = self.ampli[0, crop, crop].float().cpu().numpy()
        expected = self.fixture["output_amplitude_crop"]
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_phase_crop(self):
        crop = slice(512, 576)
        result = self.phase[0, crop, crop].float().cpu().numpy()
        expected = self.fixture["output_phase_crop"]
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_amplitude_mean(self):
        result = self.ampli.float().mean().cpu().item()
        expected = float(self.fixture["output_amplitude_mean"])
        self.assertAlmostEqual(result, expected, places=4)

    def test_phase_mean(self):
        result = self.phase.float().mean().cpu().item()
        expected = float(self.fixture["output_phase_mean"])
        self.assertAlmostEqual(result, expected, places=4)


class TestFullModelShapes(unittest.TestCase):
    """Test model output shapes for various batch sizes (CPU only)."""

    def test_single_z(self):
        from src.solvers import FullModel
        model = FullModel(
            w=64, h=64, num_feats=8, x_mode=32, y_mode=32,
            z_min=-20.0, z_max=20.0, ds_factor=1, use_layernorm=False,
        )
        dz = torch.tensor([0.0])
        ampli, phase = model(dz)
        self.assertEqual(ampli.shape, (1, 64, 64))
        self.assertEqual(phase.shape, (1, 64, 64))

    def test_batch_z(self):
        from src.solvers import FullModel
        model = FullModel(
            w=64, h=64, num_feats=8, x_mode=32, y_mode=32,
            z_min=-20.0, z_max=20.0, ds_factor=1, use_layernorm=False,
        )
        dz = torch.linspace(-20, 20, 5)
        ampli, phase = model(dz)
        self.assertEqual(ampli.shape, (5, 64, 64))
        self.assertEqual(phase.shape, (5, 64, 64))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSaveLoadModel(unittest.TestCase):
    """Test save/load roundtrip preserves model outputs exactly."""

    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR_NETWORK, "save_load_model.npz"), allow_pickle=True
        )

    def test_roundtrip_exact(self):
        from src.solvers import FullModel, save_model_with_required_grad, load_model_with_required_grad
        import tempfile

        device = "cuda:0"
        model = FullModel(
            w=2048, h=2048, num_feats=32, x_mode=512, y_mode=512,
            z_min=-20.0, z_max=20.0, ds_factor=1, use_layernorm=False,
        ).to(device)
        load_model_with_required_grad(
            model,
            os.path.join(TASK_DIR, "evaluation", "reference_outputs", "model_weights.pth"),
        )
        model.eval()

        dz = torch.tensor([0.0]).to(device)
        with torch.no_grad():
            ampli1, _ = model(dz)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            save_model_with_required_grad(model, tmp_path)

            model2 = FullModel(
                w=2048, h=2048, num_feats=32, x_mode=512, y_mode=512,
                z_min=-20.0, z_max=20.0, ds_factor=1, use_layernorm=False,
            ).to(device)
            load_model_with_required_grad(model2, tmp_path)
            model2.eval()

            with torch.no_grad():
                ampli2, _ = model2(dz)

            diff = (ampli1 - ampli2).abs().max().item()
            self.assertAlmostEqual(diff, 0.0, places=10)
        finally:
            os.remove(tmp_path)


class TestGRenderer(unittest.TestCase):
    """Test G_Renderer MLP component."""

    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR_NETWORK, "g_renderer.npz"), allow_pickle=True
        )

    def test_output_shape(self):
        from src.solvers import G_Renderer
        renderer = G_Renderer(in_dim=32, hidden_dim=32, num_layers=2, out_dim=1)
        renderer.load_state_dict(
            torch.load(os.path.join(FIXTURE_DIR_NETWORK, "g_renderer_state.pth"),
                       map_location="cpu", weights_only=False)
        )
        renderer.eval()

        x = torch.from_numpy(self.fixture["input_data"])
        with torch.no_grad():
            out = renderer(x)
        expected_shape = tuple(self.fixture["output_shape"])
        self.assertEqual(out.shape, torch.Size(expected_shape))

    def test_output_values(self):
        from src.solvers import G_Renderer
        renderer = G_Renderer(in_dim=32, hidden_dim=32, num_layers=2, out_dim=1)
        renderer.load_state_dict(
            torch.load(os.path.join(FIXTURE_DIR_NETWORK, "g_renderer_state.pth"),
                       map_location="cpu", weights_only=False)
        )
        renderer.eval()

        x = torch.from_numpy(self.fixture["input_data"])
        with torch.no_grad():
            out = renderer(x)
        np.testing.assert_allclose(
            out.numpy(), self.fixture["output_data"], rtol=1e-5, atol=1e-6
        )


class TestGTensor3D(unittest.TestCase):
    """Test G_Tensor3D 3D implicit representation."""

    def test_output_shape(self):
        from src.solvers import G_Tensor3D
        torch.manual_seed(42)
        t3d = G_Tensor3D(x_mode=32, y_mode=32, z_dim=5, z_min=-20.0, z_max=20.0, num_feats=8)
        t3d.create_coords(x_dim=64, y_dim=64, x_max=32, y_max=32)
        t3d.eval()
        dz = torch.tensor([0.0, 10.0])
        with torch.no_grad():
            out = t3d(dz)
        self.assertEqual(out.shape, (2, 1, 64, 64))

    def test_z_boundary(self):
        from src.solvers import G_Tensor3D
        torch.manual_seed(0)
        t3d = G_Tensor3D(x_mode=16, y_mode=16, z_dim=5, z_min=-20.0, z_max=20.0, num_feats=8)
        t3d.create_coords(x_dim=32, y_dim=32, x_max=16, y_max=16)
        t3d.eval()
        dz = torch.tensor([-20.0, 20.0])
        with torch.no_grad():
            out = t3d(dz)
        self.assertEqual(out.shape, (2, 1, 32, 32))
        self.assertFalse(torch.isnan(out).any())


class TestInitScaleGrids(unittest.TestCase):
    """Test FullModel.init_scale_grids rescales correctly."""

    def test_rescale(self):
        from src.solvers import FullModel
        model = FullModel(
            w=64, h=64, num_feats=8, x_mode=32, y_mode=32,
            z_min=-20.0, z_max=20.0, ds_factor=1, use_layernorm=False,
        )
        dz = torch.tensor([0.0])
        with torch.no_grad():
            a1, _ = model(dz)
        self.assertEqual(a1.shape, (1, 64, 64))

        model.init_scale_grids(ds_factor=2)
        with torch.no_grad():
            a2, _ = model(dz)
        self.assertEqual(a2.shape, (1, 64, 64))


# ──────────────────────────────────────────────────────────────────────────────
# Solver Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestZSampling(unittest.TestCase):
    """Test z-plane sampling strategy."""

    def setUp(self):
        from src.solvers import FPMINRSolver
        self.solver = FPMINRSolver()
        self.z_params = {"z_min": -20.0, "z_max": 20.0, "num_z": 14}

    def test_even_epoch_uniform(self):
        dzs = self.solver._sample_z_depths(0, self.z_params, "cpu")
        expected = torch.linspace(-20.0, 20.0, 14)
        torch.testing.assert_close(dzs, expected)

    def test_odd_epoch_random(self):
        dzs = self.solver._sample_z_depths(1, self.z_params, "cpu")
        # Random sampling: should have num_z // 2 elements
        self.assertEqual(len(dzs), 14 // 2)
        # All values should be within z range (with some tolerance)
        self.assertTrue(torch.all(dzs >= -25.0))
        self.assertTrue(torch.all(dzs <= 25.0))

    def test_even_odd_differ(self):
        dzs_even = self.solver._sample_z_depths(0, self.z_params, "cpu")
        dzs_odd = self.solver._sample_z_depths(1, self.z_params, "cpu")
        self.assertNotEqual(len(dzs_even), len(dzs_odd))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestEvaluate(unittest.TestCase):
    """Test inference on trained model."""

    def setUp(self):
        self.fixture = np.load(
            os.path.join(FIXTURE_DIR_SOLVERS, "evaluate.npz"), allow_pickle=True
        )
        from src.solvers import FullModel, load_model_with_required_grad, FPMINRSolver

        device = "cuda:0"
        self.model = FullModel(
            w=2048, h=2048, num_feats=32, x_mode=512, y_mode=512,
            z_min=-20.0, z_max=20.0, ds_factor=1, use_layernorm=False,
        ).to(device)
        load_model_with_required_grad(
            self.model,
            os.path.join(TASK_DIR, "evaluation", "reference_outputs", "model_weights.pth"),
        )
        self.solver = FPMINRSolver()

    def test_output_shapes(self):
        z_pos = self.fixture["input_z_positions"]
        ampli, phase = self.solver.evaluate(self.model, z_pos, device="cuda:0")
        expected_shape = tuple(self.fixture["output_amplitude_shape"])
        self.assertEqual(ampli.shape, expected_shape)
        self.assertEqual(phase.shape, tuple(self.fixture["output_phase_shape"]))

    def test_amplitude_crop(self):
        z_pos = self.fixture["input_z_positions"]
        ampli, _ = self.solver.evaluate(self.model, z_pos, device="cuda:0")
        crop = ampli[0, 512:576, 512:576]
        np.testing.assert_allclose(
            crop, self.fixture["output_amplitude_crop"], rtol=1e-4, atol=1e-5
        )

    def test_amplitude_mean(self):
        z_pos = self.fixture["input_z_positions"]
        ampli, _ = self.solver.evaluate(self.model, z_pos, device="cuda:0")
        np.testing.assert_allclose(
            ampli.mean(), float(self.fixture["output_amplitude_mean"]), rtol=1e-3
        )


class TestTrainInterface(unittest.TestCase):
    """Test solver.train returns expected structure (tiny model, 1 epoch)."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_returns_expected_keys(self):
        from src.solvers import FullModel, FPMINRSolver
        from src.physics_model import FPMForwardModel
        from src.preprocessing import prepare_data

        device = "cuda:0"
        data = prepare_data(os.path.join(TASK_DIR, "data"), device=device)

        model = FullModel(
            w=data["optical_params"]["MM"], h=data["optical_params"]["MM"],
            num_feats=8, x_mode=data["optical_params"]["M"],
            y_mode=data["optical_params"]["N"],
            z_min=data["z_params"]["z_min"], z_max=data["z_params"]["z_max"],
            ds_factor=1, use_layernorm=False,
        ).to(device)

        forward_model = FPMForwardModel(
            Pupil0=data["pupil_data"]["Pupil0"],
            kzz=data["pupil_data"]["kzz"],
            ledpos_true=data["optical_params"]["ledpos_true"],
            M=data["optical_params"]["M"], N=data["optical_params"]["N"],
            MAGimg=data["optical_params"]["MAGimg"],
        )

        solver = FPMINRSolver(num_epochs=1, use_compile=False)
        result = solver.train(
            model, forward_model, data["Isum"], data["z_params"], device
        )

        for key in ["final_loss", "final_psnr", "loss_history"]:
            self.assertIn(key, result)
        self.assertEqual(len(result["loss_history"]), 1)
        self.assertIsInstance(result["final_loss"], float)


if __name__ == "__main__":
    unittest.main()
