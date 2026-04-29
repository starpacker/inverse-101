from pathlib import Path

import numpy as np

from src.solvers import coherent_compound, fkmig


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_fkmig_matches_fixture():
    inputs = np.load(FIXTURES / "input_fkmig_case.npz")
    outputs = np.load(FIXTURES / "output_fkmig_case.npz")

    x, z, mig = fkmig(
        inputs["SIG"],
        float(inputs["fs"]),
        float(inputs["pitch"]),
        TXangle=float(inputs["txangle"]),
        c=float(inputs["c"]),
        t0=float(inputs["t0"]),
    )

    np.testing.assert_allclose(x, outputs["x"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(z, outputs["z"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(mig, outputs["mig"], rtol=1e-10, atol=1e-10)


def test_coherent_compound_matches_fixture():
    inputs = np.load(FIXTURES / "input_compound_case.npz")
    outputs = np.load(FIXTURES / "output_compound_case.npz")

    x, z, compound = coherent_compound(
        inputs["RF"],
        float(inputs["fs"]),
        float(inputs["pitch"]),
        inputs["TXangles"],
        c=float(inputs["c"]),
        t0=float(inputs["t0"]),
    )

    np.testing.assert_allclose(x, outputs["x"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(z, outputs["z"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(compound, outputs["compound"], rtol=1e-10, atol=1e-10)
