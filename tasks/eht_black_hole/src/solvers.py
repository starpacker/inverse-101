"""
Inverse Problem Solvers for VLBI Imaging
=========================================

All solvers share a common interface::

    reconstructor = SolverClass(**hyperparams)
    image = reconstructor.reconstruct(model, vis, noise_std)

Implemented methods
-------------------
DirtyImageReconstructor
    Baseline: matched-filter back-projection. No deconvolution.

CLEANReconstructor
    Högbom CLEAN (1974). Standard algorithm in radio astronomy.
    Assumes sky = sum of point sources; iteratively deconvolves the PSF.

RMLSolver
    Regularized Maximum Likelihood via L-BFGS-B.
    Minimises: χ²(x) + Σ_r λ_r R_r(x),  subject to x ≥ 0.
    Plug in any combination of regularizers from below.

Regularizers (used with RMLSolver)
------------------------------------
TVRegularizer          Total Variation  ‖∇x‖₁  (isotropic, Huber-smoothed)
MaxEntropyRegularizer  -H(x)  relative entropy w.r.t. a flat prior
L1SparsityRegularizer  ‖x‖₁  (Huber-smoothed)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter


# ═══════════════════════════════════════════════════════════════════════════
# Reconstructors
# ═══════════════════════════════════════════════════════════════════════════

class DirtyImageReconstructor:
    """
    Baseline reconstructor: normalised back-projection (dirty image).

    No deconvolution is performed. The result is the sky brightness
    convolved with the PSF (dirty beam) of the array.
    """

    def reconstruct(self, model, vis: np.ndarray, noise_std: float = 1.0) -> np.ndarray:
        """
        Parameters
        ----------
        model : VLBIForwardModel
        vis   : (M,) complex visibilities
        noise_std : unused (kept for API consistency)

        Returns
        -------
        image : (N, N) dirty image
        """
        return model.dirty_image(vis)


# ───────────────────────────────────────────────────────────────────────────

class CLEANReconstructor:
    """
    Högbom CLEAN Algorithm.

    The sky brightness is modelled as a superposition of point sources
    (delta functions). CLEAN iteratively finds and subtracts the brightest
    source from the dirty (residual) image, accumulating "CLEAN components".
    Finally, the components are convolved with an idealised Gaussian beam
    (the "restoring beam") and the remaining residual is added back.

    Algorithm
    ---------
    1. residual ← dirty_image(vis)
    2. Repeat up to n_iter times:
         a. peak ← argmax |residual|
         b. components[peak] += gain × residual[peak]
         c. residual -= gain × residual[peak] × PSF(shifted to peak)
         d. Stop if max|residual| < threshold
    3. Return:  gaussian_filter(components, σ_beam) + residual

    Parameters
    ----------
    gain : float
        Loop gain (fraction of peak subtracted per iteration). Typical: 0.05–0.2.
    n_iter : int
        Maximum number of minor-cycle iterations.
    threshold : float
        Stop when max|residual| falls below this fraction of the initial peak.
    clean_beam_fwhm : float or None
        FWHM of the restoring Gaussian beam in pixels.
        If None, estimated automatically from the PSF main lobe.
    support_radius : float or None
        Radius (in pixels from image centre) within which CLEAN may place
        components. This CLEAN window is critical for sparse arrays like EHT
        whose PSF has sidelobes comparable in height to the main lobe
        (~98% for EHT/M87*). Without a support constraint the algorithm
        diverges because every subtraction creates new false peaks in the
        sidelobe region. Set to the expected source size + a margin.
        If None, the full image is searched (only safe for dense uv-coverage).

    Reference
    ---------
    Högbom, J.A. (1974). A&AS 15, 417.
    """

    def __init__(
        self,
        gain: float = 0.05,
        n_iter: int = 50,
        threshold: float = 1e-4,
        clean_beam_fwhm: float = None,
        support_radius: float = None,
    ):
        self.gain = gain
        self.n_iter = n_iter
        self.threshold = threshold
        self.clean_beam_fwhm = clean_beam_fwhm
        self.support_radius = support_radius

    def reconstruct(self, model, vis: np.ndarray, noise_std: float = 1.0) -> np.ndarray:
        N = model.N
        dirty = model.dirty_image(vis)
        psf = model.psf()

        components = np.zeros((N, N))
        residual = dirty.copy()
        psf_center = np.array([N // 2, N // 2])
        initial_peak = residual.max()
        stop_level = self.threshold * initial_peak

        # ── Build support mask ────────────────────────────────────────────
        # Only search for CLEAN components within the support region.
        # For EHT-like sparse arrays this is mandatory: PSF sidelobes can
        # reach ~98% of the main lobe, causing divergence without a window.
        if self.support_radius is not None:
            ii, jj = np.ogrid[:N, :N]
            dist = np.sqrt((ii - N // 2) ** 2 + (jj - N // 2) ** 2)
            support = dist <= self.support_radius
        else:
            support = np.ones((N, N), dtype=bool)

        masked_residual = np.where(support, residual, -np.inf)

        for _ in range(self.n_iter):
            if masked_residual.max() < stop_level:
                break

            # Only subtract from positive peaks within the support window.
            peak_idx = np.unravel_index(masked_residual.argmax(), (N, N))
            peak_val = residual[peak_idx]

            if peak_val <= 0:
                break

            # Accumulate delta-function component
            components[peak_idx] += self.gain * peak_val

            # Subtract PSF centred on peak from residual
            shift = (int(peak_idx[0]) - psf_center[0],
                     int(peak_idx[1]) - psf_center[1])
            psf_shifted = np.roll(psf, shift, axis=(0, 1))
            residual -= self.gain * peak_val * psf_shifted

            masked_residual = np.where(support, residual, -np.inf)

        # ── Restore ──────────────────────────────────────────────────────
        fwhm = self.clean_beam_fwhm if self.clean_beam_fwhm is not None \
               else self._estimate_beam_fwhm(psf)
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        clean_image = gaussian_filter(components, sigma=sigma) + residual

        return clean_image

    @staticmethod
    def _estimate_beam_fwhm(psf: np.ndarray) -> float:
        """Estimate restoring beam FWHM from PSF main lobe (pixels)."""
        N = psf.shape[0]
        row = psf[N // 2, :]
        half_max = 0.5 * row.max()
        above = np.where(row >= half_max)[0]
        if len(above) >= 2:
            return float(above[-1] - above[0])
        return 2.0


# ───────────────────────────────────────────────────────────────────────────

class RMLSolver:
    """
    Regularized Maximum Likelihood (RML) Solver.

    Minimises the penalised negative log-likelihood::

        L(x) = χ²(x) + Σ_r λ_r R_r(x)

    where::

        χ²(x) = ‖Ax − y‖² / (2σ²)   [data fidelity / chi-squared]

    subject to x ≥ 0 (positivity of sky brightness).

    Solved with L-BFGS-B (limited-memory quasi-Newton with box constraints).

    Parameters
    ----------
    regularizers : list of (weight, regularizer) tuples
        Each regularizer must implement ``value_and_grad(x) → (float, ndarray)``.
        Example::

            regularizers = [
                (1e3, TVRegularizer()),
                (1e2, MaxEntropyRegularizer()),
            ]
    n_iter : int
        Maximum L-BFGS-B iterations.
    positivity : bool
        If True, enforce x ≥ 0 via box constraints.

    Reference
    ---------
    EHT Collaboration et al. (2019). ApJL 875, L3.
    Chael et al. (2018). ApJ 857, 23.
    """

    def __init__(
        self,
        regularizers=None,
        n_iter: int = 500,
        positivity: bool = True,
    ):
        self.regularizers = regularizers or []
        self.n_iter = n_iter
        self.positivity = positivity

    def reconstruct(
        self,
        model,
        vis: np.ndarray,
        noise_std: float = 1.0,
        x0: np.ndarray = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        model     : VLBIForwardModel
        vis       : (M,) complex visibilities
        noise_std : noise standard deviation σ
        x0        : (N, N) initial image; defaults to clipped dirty image

        Returns
        -------
        image : (N, N) reconstructed image
        """
        N = model.N

        if x0 is None:
            x0 = model.dirty_image(vis).clip(0)

        def objective_and_grad(x_flat):
            x = x_flat.reshape(N, N)

            # ── Data fidelity ───────────────────────────────────────────
            residual = model.A @ x_flat - vis          # (M,) complex
            chi2 = 0.5 * np.sum(np.abs(residual) ** 2) / noise_std ** 2
            grad_chi2 = (model.A.conj().T @ residual).real / noise_std ** 2

            # ── Regularization ──────────────────────────────────────────
            reg_val = 0.0
            reg_grad = np.zeros(N * N)
            for weight, reg in self.regularizers:
                rv, rg = reg.value_and_grad(x)
                reg_val += weight * rv
                reg_grad += weight * rg.ravel()

            return chi2 + reg_val, grad_chi2 + reg_grad

        bounds = [(0.0, None)] * (N * N) if self.positivity else None

        result = minimize(
            objective_and_grad,
            x0.ravel().astype(np.float64),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self.n_iter, "ftol": 1e-14, "gtol": 1e-9},
        )

        return result.x.reshape(N, N)


# ═══════════════════════════════════════════════════════════════════════════
# Regularizers
# ═══════════════════════════════════════════════════════════════════════════

class TVRegularizer:
    """
    Isotropic Total Variation regularizer.

    TV(x) = Σ_{i,j} √( (∂x/∂l)²_{i,j} + (∂x/∂m)²_{i,j} + ε² ) − ε

    Huber smoothing (parameter ε) makes the function differentiable at zero,
    enabling gradient-based optimisation.

    Parameters
    ----------
    epsilon : float
        Smoothing parameter. Smaller values approximate ‖·‖₁ more closely
        but make the gradient stiffer near zero.
    """

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def value_and_grad(self, x: np.ndarray):
        """
        Parameters
        ----------
        x : (N, N) image

        Returns
        -------
        val  : float, TV(x)
        grad : (N, N) array, ∂TV/∂x
        """
        eps = self.epsilon

        # Forward finite differences (periodic boundary)
        dx = np.roll(x, -1, axis=1) - x   # ∂x/∂l  (horizontal)
        dy = np.roll(x, -1, axis=0) - x   # ∂x/∂m  (vertical)

        # Huber-smoothed magnitude
        mag = np.sqrt(dx ** 2 + dy ** 2 + eps ** 2)
        val = float(np.sum(mag - eps))

        # Gradient via chain rule (divergence of dx/mag, dy/mag)
        dv_dx = dx / mag - np.roll(dx / mag, 1, axis=1)
        dv_dy = dy / mag - np.roll(dy / mag, 1, axis=0)
        grad = dv_dx + dv_dy

        return val, grad


class MaxEntropyRegularizer:
    """
    Relative entropy (Kullback–Leibler divergence from a prior).

    R(x) = Σ_i x_i log(x_i / p_i)   (negative entropy w.r.t. prior p)

    Entering with a positive weight λ in the RML objective this penalises
    deviations from the prior distribution, favouring smooth, diffuse images.

    Parameters
    ----------
    prior : ndarray (N, N) or None
        Prior image p. If None, uses a uniform (flat) prior proportional
        to the image total flux.
    epsilon : float
        Small floor to avoid log(0).
    """

    def __init__(self, prior: np.ndarray = None, epsilon: float = 1e-12):
        self.prior = prior
        self.epsilon = epsilon

    def value_and_grad(self, x: np.ndarray):
        """
        Returns
        -------
        val  : float, R(x) = Σ x_i log(x_i / p_i)
        grad : (N, N) array, ∂R/∂x = log(x/p) + 1
        """
        if self.prior is None:
            total = x.sum() + self.epsilon
            prior = np.full_like(x, total / x.size)
        else:
            prior = self.prior

        x_s = np.maximum(x, self.epsilon)
        p_s = np.maximum(prior, self.epsilon)

        val = float(np.sum(x_s * np.log(x_s / p_s)))
        grad = np.log(x_s / p_s) + 1.0

        return val, grad


class L1SparsityRegularizer:
    """
    Smoothed L1 (sparsity) regularizer.

    R(x) = Σ_i (√(x_i² + ε²) − ε)  ≈  ‖x‖₁

    Promotes solutions with compact, point-like emission.

    Parameters
    ----------
    epsilon : float
        Huber smoothing parameter.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def value_and_grad(self, x: np.ndarray):
        eps = self.epsilon
        mag = np.sqrt(x ** 2 + eps ** 2)
        val = float(np.sum(mag - eps))
        grad = x / mag
        return val, grad
