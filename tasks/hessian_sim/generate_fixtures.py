"""Generate basic fixtures for hessian_sim.

Runs the physics model functions on small synthetic data.
"""
import sys
import pathlib
import numpy as np

TASK_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from src.physics import generate_otf, pad_to_size, dft_conv, shift_otf

FIXTURES_DIR = TASK_DIR / "evaluation" / "fixtures"


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate OTF
    n = 32
    na = 1.2
    wavelength = 525.0  # nm
    pixel_size = 65.0   # nm
    otf = generate_otf(n, na, wavelength, pixel_size)

    # Test pad_to_size
    small = np.random.default_rng(42).standard_normal((16, 16))
    padded = pad_to_size(small, (32, 32))

    # Test dft_conv
    h = np.random.default_rng(42).standard_normal((8, 8))
    g = np.random.default_rng(43).standard_normal((8, 8))
    conv_result = dft_conv(h, g)

    # Test shift_otf
    H_2n = np.zeros((2 * n, 2 * n), dtype=complex)
    H_2n[n - n // 2:n + n // 2, n - n // 2:n + n // 2] = otf
    shifted = shift_otf(H_2n, 0.1, 0.2, n)

    np.savez(
        FIXTURES_DIR / "basic_pipeline.npz",
        otf=otf,
        padded=padded,
        conv_result=conv_result,
        shifted_real=shifted.real,
        shifted_imag=shifted.imag,
    )
    print(f"Saved fixtures to {FIXTURES_DIR / 'basic_pipeline.npz'}")


if __name__ == "__main__":
    main()
