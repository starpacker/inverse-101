"""Tests for src/generate_data.py."""
import os
import numpy as np
import pytest

TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(TASK_DIR, "evaluation", "fixtures")

import sys
sys.path.insert(0, TASK_DIR)

from src.generate_data import define_targets, define_detector_array, define_time_vector


class TestDefineTargets:
    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "generate_data_targets.npz"))
        tar_info, z_target = define_targets()
        np.testing.assert_allclose(tar_info, fix["output_tar_info"], rtol=1e-10)
        np.testing.assert_allclose(z_target, float(fix["output_z_target"]), rtol=1e-10)

    def test_shape(self):
        tar_info, z_target = define_targets()
        assert tar_info.shape == (4, 4)
        assert isinstance(z_target, float)

    def test_positive_radii(self):
        tar_info, _ = define_targets()
        assert np.all(tar_info[:, 3] > 0)


class TestDefineDetectorArray:
    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "generate_data_detector.npz"))
        xd, yd = define_detector_array()
        np.testing.assert_allclose(xd, fix["output_xd"], rtol=1e-10)
        np.testing.assert_allclose(yd, fix["output_yd"], rtol=1e-10)

    def test_shape(self):
        xd, yd = define_detector_array()
        assert xd.shape == yd.shape
        assert len(xd) == 31

    def test_centered(self):
        xd, _ = define_detector_array()
        assert abs(np.mean(xd)) < 1e-10


class TestDefineTimeVector:
    def test_fixture_match(self):
        fix = np.load(os.path.join(FIXTURE_DIR, "generate_data_time.npz"))
        t, fs = define_time_vector()
        np.testing.assert_allclose(t, fix["output_t"], rtol=1e-10)
        np.testing.assert_allclose(fs, float(fix["output_fs"]), rtol=1e-10)

    def test_positive_time(self):
        t, fs = define_time_vector()
        assert t[0] == 0
        assert np.all(np.diff(t) > 0)
        assert fs > 0
