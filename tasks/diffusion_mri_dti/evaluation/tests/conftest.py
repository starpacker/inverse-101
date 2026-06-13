"""Shared test fixtures for diffusion_mri_dti."""

import os
import sys
import pytest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)


@pytest.fixture
def task_dir():
    return TASK_DIR


@pytest.fixture
def data_dir():
    return os.path.join(TASK_DIR, 'data')


@pytest.fixture
def fixtures_dir():
    return os.path.join(TASK_DIR, 'evaluation', 'fixtures')


@pytest.fixture
def ref_dir():
    return os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')
