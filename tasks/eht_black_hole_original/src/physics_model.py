"""
Closure-Only Forward Model for VLBI Imaging
=============================================

Extends the standard VLBI forward model (image → complex visibilities) with
closure quantity operators:

    image → visibilities → closure phases (on triangles)
                        → closure amplitudes (on quadrangles)

The key property of closure quantities is that station-based gain errors
cancel exactly, making them robust calibration-independent observables.

Includes analytic gradients of closure chi-squared terms w.r.t. image pixels,
following Chael et al. (2018), ApJ 857, 23.

Physical model (van Cittert–Zernike theorem):
    V(u,v) = ∬ I(l,m) exp(-2πi(ul+vm)) dl dm

Discretized: y = A x, where A is the DFT measurement matrix.
"""

import numpy as np


class ClosureForwardModel:
    """
    Forward model with closure quantity operators for VLBI imaging.

    Builds the DFT measurement matrix A and provides:
    - Standard forward/adjoint operators
    - Closure phase computation and gradient
    - Closure amplitude computation and gradient
    - Closure-only chi-squared and gradient

    Parameters
    ----------
    uv_coords : ndarray, shape (M, 2)
        Measured (u, v) baseline coordinates in wavelengths.
    image_size : int
        Side length N of the N×N image grid.
    pixel_size_rad : float
        Angular pixel size in radians.
    station_ids : ndarray, shape (M, 2), int
        Station pair indices for each baseline.
    triangles : ndarray, shape (N_tri, 3), int
        Baseline indices forming each closure phase triangle.
    quadrangles : ndarray, shape (N_quad, 4), int
        Baseline indices [ij, kl, ik, jl] for each closure amplitude quadrangle.
    """

    def __init__(
        self,
        uv_coords: np.ndarray,
        image_size: int,
        pixel_size_rad: float,
        station_ids: np.ndarray,
        triangles: np.ndarray,
        quadrangles: np.ndarray,
    ):
        self.uv = uv_coords
        self.N = image_size
        self.pixel_size = pixel_size_rad
        self.M = len(uv_coords)
        self.station_ids = station_ids
        self.triangles = triangles
        self.quadrangles = quadrangles

        # ── Pixel coordinate grids (centred at zero) ────────────────────
        idx = np.arange(self.N) - self.N // 2
        l, m = np.meshgrid(idx * pixel_size_rad, idx * pixel_size_rad)
        l_flat = l.ravel()
        m_flat = m.ravel()

        # ── Build measurement matrix A ──────────────────────────────────
        phase = -2j * np.pi * (
            uv_coords[:, 0:1] * l_flat[np.newaxis, :] +
            uv_coords[:, 1:2] * m_flat[np.newaxis, :]
        )
        self.A = np.exp(phase)  # (M, N²), complex128

        # ── Pre-extract DFT rows for triangle/quadrangle baselines ──────
        # For triangles: A1, A2, A3 are DFT matrices for each leg
        self._tri_A = []
        for leg in range(3):
            bl_codes = triangles[:, leg]
            indices = np.where(bl_codes >= 0, bl_codes, -(bl_codes + 1))
            self._tri_A.append(self.A[indices])  # (N_tri, N²)

        self._tri_conj = []
        for leg in range(3):
            self._tri_conj.append(triangles[:, leg] < 0)  # boolean mask

        # For quadrangles: A for each of the 4 baselines
        self._quad_A = []
        for leg in range(4):
            self._quad_A.append(self.A[quadrangles[:, leg]])

    # ──────────────────────────────────────────────────────────────────────
    # Core operators
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Compute complex visibilities from a sky brightness image.

        Parameters
        ----------
        image : ndarray, shape (N, N)

        Returns
        -------
        vis : ndarray, shape (M,), complex
        """
        return self.A @ image.ravel()

    def adjoint(self, vis: np.ndarray) -> np.ndarray:
        """
        Back-project visibilities to image space: x̃ = Aᴴ y.

        Parameters
        ----------
        vis : ndarray, shape (M,), complex

        Returns
        -------
        image : ndarray, shape (N, N), real
        """
        return (self.A.conj().T @ vis).real.reshape(self.N, self.N)

    def dirty_image(self, vis: np.ndarray) -> np.ndarray:
        """Normalized dirty image: Aᴴ y / max(Aᴴ 1)."""
        raw = self.adjoint(vis)
        return raw / self._psf_peak()

    def psf(self) -> np.ndarray:
        """Point Spread Function (dirty beam): Aᴴ 1 / max(Aᴴ 1)."""
        ones = np.ones(self.M, dtype=complex)
        raw = self.adjoint(ones)
        return raw / raw.max()

    def _psf_peak(self) -> float:
        ones = np.ones(self.M, dtype=complex)
        return self.adjoint(ones).max()

    # ──────────────────────────────────────────────────────────────────────
    # Closure phase operators
    # ──────────────────────────────────────────────────────────────────────

    def model_closure_phases(self, image: np.ndarray) -> np.ndarray:
        """
        Compute model closure phases from an image.

        Parameters
        ----------
        image : (N, N) ndarray

        Returns
        -------
        cphases : (N_tri,) ndarray — closure phases in radians
        """
        x = image.ravel()
        bispec = np.ones(len(self.triangles), dtype=np.complex128)
        for leg in range(3):
            v_leg = self._tri_A[leg] @ x
            if self._tri_conj[leg].any():
                v_leg = np.where(self._tri_conj[leg], np.conj(v_leg), v_leg)
            bispec *= v_leg
        return np.angle(bispec)

    def closure_phase_chisq(self, image: np.ndarray, cphases_obs: np.ndarray,
                            sigma_cp: np.ndarray) -> float:
        """
        Closure phase chi-squared (Eq. 11, Chael 2018).

        χ²_CP = (2/N_CP) Σ_k (1 - cos(φ_k^obs - φ_k^model)) / σ_k²

        Uses von Mises-like form for periodic closure phases.

        Parameters
        ----------
        image : (N, N)
        cphases_obs : (N_tri,) observed closure phases
        sigma_cp : (N_tri,) closure phase noise

        Returns
        -------
        chisq : float
        """
        cp_model = self.model_closure_phases(image)
        N_cp = len(cphases_obs)
        chisq = (2.0 / N_cp) * np.sum(
            (1.0 - np.cos(cphases_obs - cp_model)) / sigma_cp ** 2
        )
        return float(chisq)

    def closure_phase_chisq_grad(self, image: np.ndarray, cphases_obs: np.ndarray,
                                 sigma_cp: np.ndarray) -> np.ndarray:
        """
        Gradient of closure phase chi-squared w.r.t. image pixels.

        d(χ²_CP)/dx = (-2/N_CP) Im[Σ_j (sin(φ^obs - φ^model)/σ²) / V_j · A_j]

        Parameters
        ----------
        image : (N, N)
        cphases_obs : (N_tri,)
        sigma_cp : (N_tri,)

        Returns
        -------
        grad : (N, N) ndarray
        """
        x = image.ravel()
        N_cp = len(cphases_obs)

        # Compute model visibilities for each triangle leg
        v_legs = []
        for leg in range(3):
            v = self._tri_A[leg] @ x
            if self._tri_conj[leg].any():
                v = np.where(self._tri_conj[leg], np.conj(v), v)
            v_legs.append(v)

        # Model closure phase
        bispec = v_legs[0] * v_legs[1] * v_legs[2]
        cp_model = np.angle(bispec)

        pref = np.sin(cphases_obs - cp_model) / sigma_cp ** 2

        grad = np.zeros(self.N * self.N)
        for leg in range(3):
            v_leg = v_legs[leg]
            pt = pref / np.conj(v_leg)  # (N_tri,)
            if self._tri_conj[leg].any():
                pt = np.where(self._tri_conj[leg], np.conj(pt), pt)
            grad += np.imag(pt @ self._tri_A[leg])  # sum over triangles → (N²,)

        grad *= (-2.0 / N_cp)
        return grad.reshape(self.N, self.N)

    # ──────────────────────────────────────────────────────────────────────
    # Log closure amplitude operators
    # ──────────────────────────────────────────────────────────────────────

    def model_log_closure_amplitudes(self, image: np.ndarray) -> np.ndarray:
        """
        Compute model log closure amplitudes from an image.

        log CA = log|V_ij| + log|V_kl| - log|V_ik| - log|V_jl|

        Parameters
        ----------
        image : (N, N)

        Returns
        -------
        log_camps : (N_quad,) ndarray
        """
        x = image.ravel()
        signs = [1.0, 1.0, -1.0, -1.0]
        log_ca = np.zeros(len(self.quadrangles))
        for leg in range(4):
            v = self._quad_A[leg] @ x
            log_ca += signs[leg] * np.log(np.maximum(np.abs(v), 1e-30))
        return log_ca

    def log_closure_amp_chisq(self, image: np.ndarray, log_camps_obs: np.ndarray,
                              sigma_logca: np.ndarray) -> float:
        """
        Log closure amplitude chi-squared (Eq. 12, Chael 2018).

        χ²_logCA = (1/N_CA) Σ_k (logCA_k^obs - logCA_k^model)² / σ_k²

        Parameters
        ----------
        image : (N, N)
        log_camps_obs : (N_quad,)
        sigma_logca : (N_quad,)

        Returns
        -------
        chisq : float
        """
        lca_model = self.model_log_closure_amplitudes(image)
        N_ca = len(log_camps_obs)
        chisq = (1.0 / N_ca) * np.sum(
            (log_camps_obs - lca_model) ** 2 / sigma_logca ** 2
        )
        return float(chisq)

    def log_closure_amp_chisq_grad(self, image: np.ndarray, log_camps_obs: np.ndarray,
                                   sigma_logca: np.ndarray) -> np.ndarray:
        """
        Gradient of log closure amplitude chi-squared w.r.t. image pixels.

        d(χ²_logCA)/dx = (-2/N_CA) Re[Σ_leg s_leg * pp/V_leg · A_leg]

        where pp = (logCA^obs - logCA^model)/σ², s_leg = +1 for numerator, -1 for denominator.

        Parameters
        ----------
        image : (N, N)
        log_camps_obs : (N_quad,)
        sigma_logca : (N_quad,)

        Returns
        -------
        grad : (N, N) ndarray
        """
        x = image.ravel()
        N_ca = len(log_camps_obs)
        signs = [1.0, 1.0, -1.0, -1.0]

        v_legs = [self._quad_A[leg] @ x for leg in range(4)]
        lca_model = np.zeros(N_ca)
        for leg in range(4):
            lca_model += signs[leg] * np.log(np.maximum(np.abs(v_legs[leg]), 1e-30))

        pp = (log_camps_obs - lca_model) / sigma_logca ** 2

        grad = np.zeros(self.N * self.N)
        for leg in range(4):
            pt = signs[leg] * pp / np.conj(v_legs[leg])
            grad += np.real(pt @ self._quad_A[leg])

        grad *= (-2.0 / N_ca)
        return grad.reshape(self.N, self.N)

    # ──────────────────────────────────────────────────────────────────────
    # Standard visibility chi-squared (for comparison)
    # ──────────────────────────────────────────────────────────────────────

    def visibility_chisq(self, image: np.ndarray, vis_obs: np.ndarray,
                         noise_std: float) -> float:
        """Standard visibility chi-squared: ‖Ax - y‖² / (2σ²M)."""
        residual = self.forward(image) - vis_obs
        return float(0.5 * np.sum(np.abs(residual) ** 2) / (noise_std ** 2 * self.M))

    def visibility_chisq_grad(self, image: np.ndarray, vis_obs: np.ndarray,
                              noise_std: float) -> np.ndarray:
        """Gradient of visibility chi-squared w.r.t. image."""
        residual = self.forward(image) - vis_obs
        grad = (self.A.conj().T @ residual).real / (noise_std ** 2 * self.M)
        return grad.reshape(self.N, self.N)

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    @property
    def shape(self):
        """(M, N, N) — number of measurements and image dimensions."""
        return self.M, self.N, self.N

    def __repr__(self):
        return (
            f"ClosureForwardModel("
            f"M={self.M} baselines, "
            f"N={self.N}×{self.N} image, "
            f"tri={len(self.triangles)}, "
            f"quad={len(self.quadrangles)}, "
            f"pixel={self.pixel_size * 180 / np.pi * 3600 * 1e6:.2f} μas)"
        )
