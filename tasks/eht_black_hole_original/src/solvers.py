"""
Inverse Problem Solvers for Closure-Only VLBI Imaging
======================================================

All solvers share a common interface::

    solver = SolverClass(**hyperparams)
    image = solver.reconstruct(model, closure_data, ...)

Implemented methods
-------------------
ClosurePhaseOnlySolver
    RML imaging using only closure phase chi-squared.
    Robust to both amplitude and phase gain errors.

ClosurePhasePlusAmpSolver
    RML imaging using closure phase + log closure amplitude chi-squared.
    Robust to all station-based calibration errors (Chael et al. 2018).

VisibilityRMLSolver
    Traditional RML using calibrated complex visibilities.
    NOT robust to gain errors — used as a comparison baseline.

Regularizers (used with all solvers)
-------------------------------------
TVRegularizer          Total Variation  ‖∇x‖₁  (isotropic, Huber-smoothed)
MaxEntropyRegularizer  -H(x)  relative entropy w.r.t. a flat prior
L1SparsityRegularizer  ‖x‖₁  (Huber-smoothed)

Reference
---------
Chael et al. (2018). ApJ, 857, 23.
"""

import numpy as np
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════════════════
# Closure-Only Solvers
# ═══════════════════════════════════════════════════════════════════════════

class ClosurePhaseOnlySolver:
    """
    RML solver using only closure phase chi-squared.

    Minimises::

        L(x) = α_cp · χ²_CP(x) + Σ_r λ_r R_r(x)

    where χ²_CP uses the von Mises form (Eq. 11, Chael 2018):

        χ²_CP = (2/N_CP) Σ (1 - cos(φ^obs - φ^model)) / σ²

    This solver is robust to arbitrary station-based gain errors
    (both amplitude and phase) because closure phases are gain-invariant.

    Parameters
    ----------
    regularizers : list of (weight, regularizer) tuples
    alpha_cp : float
        Weight for the closure phase data term.
    n_iter : int
        Maximum L-BFGS-B iterations.
    positivity : bool
        Enforce x ≥ 0 via box constraints.
    """

    def __init__(
        self,
        regularizers=None,
        alpha_cp: float = 100.0,
        n_iter: int = 500,
        positivity: bool = True,
    ):
        self.regularizers = regularizers or []
        self.alpha_cp = alpha_cp
        self.n_iter = n_iter
        self.positivity = positivity

    def reconstruct(
        self,
        model,
        closure_data: dict,
        x0: np.ndarray = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        model : ClosureForwardModel
        closure_data : dict with 'cphases', 'sigma_cp'
        x0 : (N, N) initial image; defaults to uniform positive image

        Returns
        -------
        image : (N, N) reconstructed image
        """
        N = model.N
        cphases_obs = closure_data["cphases"]
        sigma_cp = closure_data["sigma_cp"]

        if x0 is None:
            x0 = np.ones((N, N)) / (N * N)

        def objective_and_grad(x_flat):
            x = x_flat.reshape(N, N)

            # Closure phase data term
            chisq = self.alpha_cp * model.closure_phase_chisq(x, cphases_obs, sigma_cp)
            grad = self.alpha_cp * model.closure_phase_chisq_grad(x, cphases_obs, sigma_cp)

            # Regularization
            for weight, reg in self.regularizers:
                rv, rg = reg.value_and_grad(x)
                chisq += weight * rv
                grad = grad + weight * rg

            return chisq, grad.ravel()

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


# ───────────────────────────────────────────────────────────────────────────

class ClosurePhasePlusAmpSolver:
    """
    RML solver using closure phase + log closure amplitude chi-squared.

    Minimises::

        L(x) = α_cp · χ²_CP(x) + α_ca · χ²_logCA(x) + Σ_r λ_r R_r(x)

    Uses both closure phases (Eq. 11) and log closure amplitudes (Eq. 12)
    from Chael et al. 2018.

    Parameters
    ----------
    regularizers : list of (weight, regularizer) tuples
    alpha_cp : float
        Weight for closure phase data term.
    alpha_ca : float
        Weight for log closure amplitude data term.
    n_iter : int
    positivity : bool
    """

    def __init__(
        self,
        regularizers=None,
        alpha_cp: float = 100.0,
        alpha_ca: float = 100.0,
        n_iter: int = 500,
        positivity: bool = True,
    ):
        self.regularizers = regularizers or []
        self.alpha_cp = alpha_cp
        self.alpha_ca = alpha_ca
        self.n_iter = n_iter
        self.positivity = positivity

    def reconstruct(
        self,
        model,
        closure_data: dict,
        x0: np.ndarray = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        model : ClosureForwardModel
        closure_data : dict with 'cphases', 'sigma_cp', 'log_camps', 'sigma_logca'
        x0 : (N, N) initial image

        Returns
        -------
        image : (N, N) reconstructed image
        """
        N = model.N
        cphases_obs = closure_data["cphases"]
        sigma_cp = closure_data["sigma_cp"]
        log_camps_obs = closure_data["log_camps"]
        sigma_logca = closure_data["sigma_logca"]

        if x0 is None:
            x0 = np.ones((N, N)) / (N * N)

        def objective_and_grad(x_flat):
            x = x_flat.reshape(N, N)

            # Closure phase data term
            chisq = self.alpha_cp * model.closure_phase_chisq(x, cphases_obs, sigma_cp)
            grad = self.alpha_cp * model.closure_phase_chisq_grad(x, cphases_obs, sigma_cp)

            # Log closure amplitude data term
            chisq += self.alpha_ca * model.log_closure_amp_chisq(x, log_camps_obs, sigma_logca)
            grad = grad + self.alpha_ca * model.log_closure_amp_chisq_grad(x, log_camps_obs, sigma_logca)

            # Regularization
            for weight, reg in self.regularizers:
                rv, rg = reg.value_and_grad(x)
                chisq += weight * rv
                grad = grad + weight * rg

            return chisq, grad.ravel()

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


# ───────────────────────────────────────────────────────────────────────────

class VisibilityRMLSolver:
    """
    Traditional RML solver using calibrated complex visibilities.

    Minimises::

        L(x) = ‖Ax − y‖² / (2σ²) + Σ_r λ_r R_r(x)

    NOT robust to station-based gain errors. Used as a comparison to
    demonstrate that closure-only imaging is superior when gains are corrupt.

    Parameters
    ----------
    regularizers : list of (weight, regularizer) tuples
    n_iter : int
    positivity : bool
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
        model : ClosureForwardModel
        vis : (M,) complex visibilities
        noise_std : float
        x0 : (N, N) initial image

        Returns
        -------
        image : (N, N) reconstructed image
        """
        N = model.N

        if x0 is None:
            x0 = model.dirty_image(vis).clip(0)

        def objective_and_grad(x_flat):
            x = x_flat.reshape(N, N)

            residual = model.A @ x_flat - vis
            chi2 = 0.5 * np.sum(np.abs(residual) ** 2) / noise_std ** 2
            grad_chi2 = (model.A.conj().T @ residual).real / noise_std ** 2

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

    Huber smoothing (parameter ε) makes the function differentiable at zero.

    Parameters
    ----------
    epsilon : float
        Smoothing parameter.
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

        dx = np.roll(x, -1, axis=1) - x
        dy = np.roll(x, -1, axis=0) - x

        mag = np.sqrt(dx ** 2 + dy ** 2 + eps ** 2)
        val = float(np.sum(mag - eps))

        dv_dx = dx / mag - np.roll(dx / mag, 1, axis=1)
        dv_dy = dy / mag - np.roll(dy / mag, 1, axis=0)
        grad = dv_dx + dv_dy

        return val, grad


class MaxEntropyRegularizer:
    """
    Relative entropy (KL divergence from a prior).

    R(x) = Σ_i x_i log(x_i / p_i)

    Parameters
    ----------
    prior : ndarray (N, N) or None
        Prior image. If None, uses a uniform flat prior.
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
        val  : float
        grad : (N, N) array
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
