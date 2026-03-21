"""
VLBI Forward Model for Radio Interferometric Imaging
=====================================================

Physical model (van Cittert–Zernike theorem):

    V(u, v) = ∬ I(l, m) exp(-2πi (ul + vm)) dl dm

Discretized as a linear measurement:

    y = A x + n

where:
    x ∈ R^{N²}    vectorized sky brightness image (non-negative)
    A ∈ C^{M×N²}  measurement matrix; A[k,j] = exp(-2πi(u_k l_j + v_k m_j)) Δθ²
    y ∈ C^M       measured complex visibilities
    n             thermal noise, n ~ CN(0, σ²I)

The system is severely underdetermined: M ≪ N².
"""

import numpy as np


class VLBIForwardModel:
    """
    Linear forward model for VLBI (Very Long Baseline Interferometry) imaging.

    Builds the full measurement matrix A and supports forward projection,
    adjoint (back-projection), dirty image, and PSF computation.

    For images larger than ~128×128 consider an NUFFT-based implementation
    instead of the direct matrix approach used here.

    Parameters
    ----------
    uv_coords : ndarray, shape (M, 2)
        Measured (u, v) baseline coordinates in wavelengths.
    image_size : int
        Side length N of the N×N image grid.
    pixel_size_rad : float
        Angular pixel size in radians.
    """

    def __init__(self, uv_coords: np.ndarray, image_size: int, pixel_size_rad: float):
        self.uv = uv_coords                 # (M, 2)
        self.N = image_size
        self.pixel_size = pixel_size_rad    # Δθ  [rad]
        self.M = len(uv_coords)

        # ── Pixel coordinate grids (centred at zero) ──────────────────────
        idx = np.arange(self.N) - self.N // 2
        l, m = np.meshgrid(idx * pixel_size_rad, idx * pixel_size_rad)

        # ── Build measurement matrix A ─────────────────────────────────────
        # A[k, j] = exp(-2πi (u_k l_j + v_k m_j)) · Δθ²
        # Shape: (M, N²)
        l_flat = l.ravel()   # (N²,)
        m_flat = m.ravel()   # (N²,)

        # Exponent: -2πi (u·l + v·m);  broadcasting (M,1) × (1,N²) → (M,N²)
        phase = -2j * np.pi * (
            uv_coords[:, 0:1] * l_flat[np.newaxis, :] +
            uv_coords[:, 1:2] * m_flat[np.newaxis, :]
        )
        # Note: we omit the Δθ² solid-angle factor from A.
        # For normalised images (x sums to 1), Δθ² is a global scale that
        # cancels in CLEAN and only shifts the absolute χ² scale in RML.
        self.A = np.exp(phase)   # (M, N²), complex128

    # ──────────────────────────────────────────────────────────────────────
    # Core operators
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Compute complex visibilities from a sky brightness image.

        Parameters
        ----------
        image : ndarray, shape (N, N)
            Sky brightness distribution [Jy/sr or normalised].

        Returns
        -------
        vis : ndarray, shape (M,), complex
            Measured complex visibilities.
        """
        return self.A @ image.ravel()

    def adjoint(self, vis: np.ndarray) -> np.ndarray:
        """
        Back-project visibilities to image space (un-normalised dirty image).

        This is the Hermitian adjoint of the forward operator:
            x̃ = Aᴴ y

        Parameters
        ----------
        vis : ndarray, shape (M,), complex

        Returns
        -------
        image : ndarray, shape (N, N), real
        """
        return (self.A.conj().T @ vis).real.reshape(self.N, self.N)

    def dirty_image(self, vis: np.ndarray) -> np.ndarray:
        """
        Compute the normalised dirty image (matched-filter reconstruction).

        I_dirty = Aᴴ y / max(Aᴴ · 1)

        The dirty image is the sky brightness convolved with the PSF (dirty beam).
        It is the baseline reconstruction before any deconvolution.
        """
        raw = self.adjoint(vis)
        psf_peak = self._psf_peak()
        return raw / psf_peak

    def psf(self) -> np.ndarray:
        """
        Compute the Point Spread Function (dirty beam).

        PSF = Aᴴ 1 / max(Aᴴ 1)

        The PSF encodes the imaging artefacts introduced by incomplete
        uv-coverage. A filled aperture would give a PSF equal to a delta
        function; EHT's sparse coverage produces a complex sidelobe pattern.
        """
        ones = np.ones(self.M, dtype=complex)
        raw = self.adjoint(ones)
        return raw / raw.max()

    def _psf_peak(self) -> float:
        ones = np.ones(self.M, dtype=complex)
        return self.adjoint(ones).max()

    # ──────────────────────────────────────────────────────────────────────
    # Noise
    # ──────────────────────────────────────────────────────────────────────

    def add_noise(self, vis: np.ndarray, snr: float = 20.0, rng=None):
        """
        Add complex Gaussian thermal noise to visibilities.

        Noise variance σ² is set so that RMS(signal) / σ = snr.

        Parameters
        ----------
        vis : ndarray, shape (M,), complex
        snr : float
            Signal-to-noise ratio (per visibility).
        rng : numpy.random.Generator or None
            Random generator for reproducibility.

        Returns
        -------
        vis_noisy : ndarray, shape (M,), complex
        noise_std : float
            Standard deviation σ of the noise (real part and imag part each).
        """
        if rng is None:
            rng = np.random.default_rng()

        signal_rms = np.sqrt(np.mean(np.abs(vis) ** 2))
        noise_std = signal_rms / snr

        noise = noise_std * (
            rng.standard_normal(self.M) + 1j * rng.standard_normal(self.M)
        ) / np.sqrt(2)

        return vis + noise, noise_std

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    @property
    def shape(self):
        """(M, N, N) — number of measurements and image dimensions."""
        return self.M, self.N, self.N

    def __repr__(self):
        return (
            f"VLBIForwardModel("
            f"M={self.M} baselines, "
            f"N={self.N}×{self.N} image, "
            f"pixel={self.pixel_size * 180 / np.pi * 3600 * 1e6:.2f} μas)"
        )
