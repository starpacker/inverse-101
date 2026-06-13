"""Unit tests for src/physics_model.py.

These tests both verify shape contracts on randomly initialised modules and,
when the released checkpoints are available, perform a per-layer parity
check against the captured encoder skip-connection feature maps.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

TASK_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

from src.physics_model import (
    ERA5InverseModel,
    ERA5_K_O,
    ERA5_K_S,
    ERA5_K_S_preimage,
    ERA5_SETTINGS,
    ERA5ForwardModel,
)
from src.preprocessing import download_pretrained_weights, load_pretrained_models

PARITY_DIR = TASK_DIR / "evaluation" / "fixtures" / "parity"
WEIGHTS_DIR = TASK_DIR / "evaluation" / "checkpoints"


class TestSettings(unittest.TestCase):
    def test_state_feature_dim_consistency(self):
        # state_feature_dim[0] = filter_dims[-1] * (H*W) / 256 ; filter_dims[-1]=256, H*W=2048
        self.assertEqual(ERA5_SETTINGS["state_feature_dim"][0], 2048)
        self.assertEqual(ERA5_SETTINGS["state_feature_dim"][1], 512)


class TestKSShapes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.K_S = ERA5_K_S()

    def test_forward_no_skip(self):
        x = torch.randn(2, 5, 64, 32)
        z = self.K_S(x)
        self.assertEqual(z.shape, (2, 512))

    def test_forward_with_skip(self):
        x = torch.randn(2, 5, 64, 32)
        z, encode_list = self.K_S(x, return_encode_list=True)
        self.assertEqual(z.shape, (2, 512))
        # 1 input clone + 5 conv outputs
        self.assertEqual(len(encode_list), 6)
        expected_shapes = [
            (2, 5, 64, 32),
            (2, 32, 64, 32),
            (2, 32, 32, 16),
            (2, 64, 16, 8),
            (2, 128, 8, 4),
            (2, 256, 4, 2),
        ]
        for actual, expected in zip(encode_list, expected_shapes):
            self.assertEqual(tuple(actual.shape), expected)


class TestKSPreimageShapes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.K_S = ERA5_K_S()
        self.K_S_pre = ERA5_K_S_preimage()

    def test_decode_with_skip(self):
        x = torch.randn(2, 5, 64, 32)
        z, encode_list = self.K_S(x, return_encode_list=True)
        recon = self.K_S_pre(z, encode_list)
        self.assertEqual(recon.shape, (2, 5, 64, 32))


class TestKO(unittest.TestCase):
    def test_inverse_obs_forward(self):
        torch.manual_seed(0)
        K_O = ERA5_K_O()
        obs = torch.randn(2, 50, 64, 32)
        out = K_O(obs)
        self.assertEqual(out.shape, (2, 5, 64, 32))


class TestForwardModelWrapper(unittest.TestCase):
    def test_latent_forward_requires_C(self):
        m = ERA5ForwardModel()
        with self.assertRaises(RuntimeError):
            m.latent_forward(torch.randn(1, 512))

    def test_latent_forward_uses_C(self):
        torch.manual_seed(0)
        m = ERA5ForwardModel()
        m.C_forward = torch.eye(512)
        z = torch.randn(3, 512)
        out = m.latent_forward(z)
        torch.testing.assert_close(out, z)


class TestInverseModelWrapper(unittest.TestCase):
    def test_K_S_frozen(self):
        m = ERA5InverseModel()
        for p in m.K_S.parameters():
            self.assertFalse(p.requires_grad)
        for p in m.K_S_preimage.parameters():
            self.assertFalse(p.requires_grad)


def _have_fixtures() -> bool:
    return (PARITY_DIR / "encode_lists" / "encode_0.npy").exists()


@unittest.skipUnless(_have_fixtures(), "parity fixtures missing")
class TestEncoderSkipConnectionsParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        download_pretrained_weights(WEIGHTS_DIR)
        cls.fwd, cls.inv, _ = load_pretrained_models(WEIGHTS_DIR, "cpu")
        cls.fwd.eval()
        cls.inv.eval()
        cls.inv_obs = np.load(PARITY_DIR / "inv_obs_seq_z.npy")

    def test_encode_list_matches(self):
        with torch.no_grad():
            inv = torch.tensor(self.inv_obs, dtype=torch.float32)
            _, encode_list = self.fwd.encode(inv, return_encode_list=True)
        for i, item in enumerate(encode_list):
            expected = np.load(PARITY_DIR / "encode_lists" / f"encode_{i}.npy")
            np.testing.assert_allclose(
                item.detach().cpu().numpy(),
                expected,
                rtol=5e-5,
                atol=5e-6,
                err_msg=f"encoder skip layer {i} drifts from upstream",
            )


if __name__ == "__main__":
    unittest.main()
