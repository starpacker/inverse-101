"""Shared pytest fixtures for FPM evaluation tests."""
import numpy as np
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"
FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


@pytest.fixture(scope="session")
def fpm_data():
    """Load the real FPM dataset once per test session."""
    from src.preprocessing import load_experimental_data
    return load_experimental_data(DATA_DIR)


@pytest.fixture(scope="session")
def fpm_state(fpm_data):
    """Initialized reconstruction state (seed=42)."""
    from src.preprocessing import setup_reconstruction
    return setup_reconstruction(fpm_data, seed=42)
