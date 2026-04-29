"""Shared fixtures for tests."""

import os
import numpy as np
import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
REF_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_outputs")


@pytest.fixture(scope="session")
def fixtures():
    return np.load(os.path.join(FIXTURE_DIR, "test_fixtures.npz"))


@pytest.fixture(scope="session")
def raw_data():
    return np.load(os.path.join(DATA_DIR, "raw_data.npz"))


@pytest.fixture(scope="session")
def reference_outputs():
    return np.load(os.path.join(REF_DIR, "unwrapped_phase.npz"))
