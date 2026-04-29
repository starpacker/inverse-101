"""Shared test configuration for MRI T2 mapping tests."""

import os
import sys

import pytest

# Add task root to path so `src` is importable
TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK_DIR)

FIXTURES_DIR = os.path.join(TASK_DIR, 'evaluation', 'fixtures')
DATA_DIR = os.path.join(TASK_DIR, 'data')
REF_DIR = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')


@pytest.fixture
def task_dir():
    return TASK_DIR


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def data_dir():
    return DATA_DIR
