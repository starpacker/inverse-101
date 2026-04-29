"""Tests for solvers module."""

import pathlib
import sys

import numpy as np
import pytest

TASK_DIR = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_DIR))

from src.solvers import ConstraintSingleGauss, build_method_configs, match_components


class TestConstraintSingleGauss:
    def test_alpha_range(self):
        with pytest.raises(ValueError):
            ConstraintSingleGauss(alpha=1.5)
        with pytest.raises(ValueError):
            ConstraintSingleGauss(alpha=-0.1)

    def test_output_shape(self):
        csg = ConstraintSingleGauss(alpha=1.0, axis=-1)
        x = np.arange(200, dtype=float)
        A = np.array([1000 + 5000 * np.exp(-(x - 100) ** 2 / (2 * 20 ** 2))])
        result = csg.transform(A)
        assert result.shape == A.shape

    def test_gaussian_recovery(self):
        csg = ConstraintSingleGauss(alpha=1.0, axis=-1)
        x = np.arange(200, dtype=float)
        # Pure Gaussian + baseline should be recovered well
        A = np.array([1000 + 8000 * np.exp(-(x - 100) ** 2 / (2 * 30 ** 2))])
        result = csg.transform(A)
        # Fit should be close to the input
        np.testing.assert_allclose(result, A, rtol=0.05)


class TestBuildMethodConfigs:
    def test_returns_five_methods(self):
        configs = build_method_configs()
        assert len(configs) == 5

    def test_config_keys(self):
        configs = build_method_configs()
        for c in configs:
            assert "name" in c
            assert "c_regr" in c
            assert "st_regr" in c
            assert "c_constraints" in c
            assert "st_constraints" in c

    def test_method_names(self):
        configs = build_method_configs()
        names = [c["name"] for c in configs]
        assert "MCR-ALS" in names
        assert "MCR-NNLS" in names
        assert "MCR-AR Gauss" in names
        assert "MCR-AR Ridge" in names
        assert "MCR-AR Lasso" in names


class TestMatchComponents:
    def test_perfect_match(self):
        conc = np.eye(3)
        # Estimated is same but permuted: columns [2, 0, 1]
        C_est = conc[:, [2, 0, 1]]
        select = match_components(C_est, conc, 3)
        # select[k] should map true column k to estimated column
        assert select[0] == 1  # true col 0 matches est col 1
        assert select[1] == 2  # true col 1 matches est col 2
        assert select[2] == 0  # true col 2 matches est col 0

    def test_identity(self):
        conc = np.eye(4)
        select = match_components(conc, conc, 4)
        assert select == [0, 1, 2, 3]
