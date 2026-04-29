from pathlib import Path

import numpy as np

from src.visualization import envelope_bmode, measure_cnr, measure_psf_fwhm


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_envelope_bmode_matches_fixture():
    mig = np.load(FIXTURES / "input_envelope_bmode.npy")
    expected = np.load(FIXTURES / "output_envelope_bmode.npy")
    actual = envelope_bmode(mig, gamma=0.5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_measure_psf_fwhm_matches_fixture():
    inputs = np.load(FIXTURES / "input_psf_case.npz")
    expected = np.load(FIXTURES / "output_psf_fwhm.npy")
    actual = measure_psf_fwhm(inputs["bmode"], inputs["x"], inputs["z"], inputs["z_targets"].tolist())
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_measure_cnr_matches_fixture():
    inputs = np.load(FIXTURES / "input_cnr_case.npz")
    expected = np.load(FIXTURES / "output_cnr.npy")
    centers = [tuple(row) for row in inputs["cyst_centers"]]
    actual = measure_cnr(
        inputs["bmode"],
        inputs["x"],
        inputs["z"],
        centers,
        cyst_radius=float(inputs["cyst_radius"]),
        shell_inner=float(inputs["shell_inner"]),
        shell_outer=float(inputs["shell_outer"]),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
