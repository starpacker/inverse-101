"""Shared test fixtures for pet_mlem."""
import os, sys, pytest

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, TASK_DIR)

@pytest.fixture
def task_dir():
    return TASK_DIR
