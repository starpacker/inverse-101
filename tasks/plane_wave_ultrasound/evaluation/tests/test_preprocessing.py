from pathlib import Path

import numpy as np

from src.preprocessing import load_dataset, remove_dc


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_remove_dc_matches_fixture():
    rf = np.load(FIXTURES / "input_remove_dc.npy")
    expected = np.load(FIXTURES / "output_remove_dc.npy")
    actual = remove_dc(rf)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_load_dataset_returns_dc_removed_array_and_parameters():
    rf, params = load_dataset(
        str(FIXTURES / "input_toy_raw_data.npz"),
        str(FIXTURES / "input_toy_meta_data.json"),
        dataset="fibers",
    )

    expected_rf = np.load(FIXTURES / "output_toy_rf_fibers.npy")
    np.testing.assert_allclose(rf, expected_rf, rtol=1e-10, atol=1e-10)

    assert params == {
        "c": 1540.0,
        "fs": 20_000_000.0,
        "pitch": 2.98e-4,
        "TXangle_rad": [-0.1, 0.0, 0.1],
        "t0": 0.0,
    }
