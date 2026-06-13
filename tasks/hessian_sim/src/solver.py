"""
Solver module: y → x_hat (inverse problem solvers).

Contains Hessian denoising and TV denoising via Split-Bregman iteration.
These take the Wiener-SIM reconstruction as input and produce denoised output.
"""

import numpy as np
from numpy.fft import fftn, ifftn


def _fdiff(x, dim):
    """Forward finite difference along axis *dim* with periodic BC."""
    return np.roll(x, -1, axis=dim) - x


def _bdiff(x, dim):
    """Backward finite difference along axis *dim* with periodic BC."""
    return x - np.roll(x, 1, axis=dim)


def hessian_denoise(stack, mu=150.0, sigma_z=1.0, n_iter=100, lamda=1.0):
    """Hessian-regularized denoising via Split Bregman iteration.

    Solves: min_x (mu/2)||x - y||^2 + ||D_xx x||_1 + ||D_yy x||_1
            + sigma^2 ||D_zz x||_1 + 2||D_xy x||_1
            + 2*sigma||D_xz x||_1 + 2*sigma||D_yz x||_1

    Parameters
    ----------
    stack : ndarray (nz, sy, sx) or (sy, sx)
    mu : float – data fidelity weight (siranu)
    sigma_z : float – z-axis weight (zbei)
    n_iter : int
    lamda : float – Bregman splitting parameter (0.5 in Bregman_Hessian_Denoise.m, 1.0 in LowRam version)

    Returns
    -------
    x : ndarray, same shape as input.
    """
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]

    nz, sy_s, sx_s = stack.shape
    nz_orig = nz

    if nz < 3:
        sigma_z = 0.0
        pad_z = 3 - nz
        stack = np.concatenate([stack, np.tile(stack[-1:], (pad_z, 1, 1))], axis=0)
        nz = 3

    y = stack.astype(np.float32)
    ymax = y.max()
    if ymax > 0:
        y = y / ymax

    sizex = (nz, sy_s, sx_s)
    siranu = mu
    zbei = sigma_z

    # FFT of second-order difference operators
    kernel = np.zeros(sizex)
    kernel[0, 0, :3] = [1, -2, 1]
    Frefft = fftn(kernel) * np.conj(fftn(kernel))

    kernel = np.zeros(sizex)
    kernel[0, :3, 0] = [1, -2, 1]
    Frefft = Frefft + fftn(kernel) * np.conj(fftn(kernel))

    kernel = np.zeros(sizex)
    kernel[:3, 0, 0] = [1, -2, 1]
    Frefft = Frefft + (zbei ** 2) * fftn(kernel) * np.conj(fftn(kernel))

    kernel = np.zeros(sizex)
    kernel[0, :2, :2] = [[1, -1], [-1, 1]]
    Frefft = Frefft + 2 * fftn(kernel) * np.conj(fftn(kernel))

    kernel = np.zeros(sizex)
    kernel[:2, 0, :2] = [[1, -1], [-1, 1]]
    Frefft = Frefft + 2 * zbei * fftn(kernel) * np.conj(fftn(kernel))

    kernel = np.zeros(sizex)
    kernel[:2, :2, 0] = [[1, -1], [-1, 1]]
    Frefft = Frefft + 2 * zbei * fftn(kernel) * np.conj(fftn(kernel))

    divide = np.real((siranu / lamda) + Frefft).astype(np.float32)

    b = [np.zeros(sizex, dtype=np.float32) for _ in range(6)]
    x = np.zeros(sizex, dtype=np.float32)

    # MATLAB structure: frac is set BEFORE the loop, then carried across iterations.
    # At the top of each iteration, frac (which includes adjoint terms from prev iter)
    # is FFT'd for the x-update. Then frac is reset to data_term and adjoint terms
    # are added again for the NEXT iteration's x-update.
    frac = (siranu / lamda) * y
    for it in range(n_iter):
        # x-update: uses frac from previous iteration (data + adjoint terms)
        frac_f = fftn(frac)
        if it > 0:
            x = np.real(ifftn(frac_f / divide)).astype(np.float32)
        else:
            x = np.real(ifftn(frac_f / (siranu / lamda))).astype(np.float32)

        # Reset frac to data fidelity term, then accumulate adjoint terms
        frac = (siranu / lamda) * y

        # d_xx (axis=2)
        u = _bdiff(_fdiff(x, 2), 2)
        s = np.abs(u + b[0]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[0])
        b[0] += u - d
        frac += _bdiff(_fdiff(d - b[0], 2), 2)

        # d_yy (axis=1)
        u = _bdiff(_fdiff(x, 1), 1)
        s = np.abs(u + b[1]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[1])
        b[1] += u - d
        frac += _bdiff(_fdiff(d - b[1], 1), 1)

        # d_zz (axis=0)
        u = _bdiff(_fdiff(x, 0), 0)
        s = np.abs(u + b[2]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[2])
        b[2] += u - d
        frac += (zbei ** 2) * _bdiff(_fdiff(d - b[2], 0), 0)

        # d_xy
        u = _fdiff(_fdiff(x, 2), 1)
        s = np.abs(u + b[3]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[3])
        b[3] += u - d
        frac += 2 * _bdiff(_bdiff(d - b[3], 1), 2)

        # d_xz
        u = _fdiff(_fdiff(x, 2), 0)
        s = np.abs(u + b[4]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[4])
        b[4] += u - d
        frac += 2 * zbei * _bdiff(_bdiff(d - b[4], 0), 2)

        # d_yz
        u = _fdiff(_fdiff(x, 1), 0)
        s = np.abs(u + b[5]) - 1.0 / lamda
        s[s < 0] = 0
        d = s * np.sign(u + b[5])
        b[5] += u - d
        frac += 2 * zbei * _bdiff(_bdiff(d - b[5], 0), 1)

    x[x < 0] = 0
    x = x[:nz_orig] * ymax
    return x


def tv_denoise(stack, mu=150.0, n_iter=100):
    """TV denoising via Split Bregman iteration.

    Solves: min_x (mu/2)||x - y||^2 + ||D_x x||_1 + ||D_y x||_1

    Parameters
    ----------
    stack : ndarray (nz, sy, sx) or (sy, sx)
    mu : float – data fidelity weight
    n_iter : int

    Returns
    -------
    x : ndarray, same shape as input.
    """
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]

    nz, sy_s, sx_s = stack.shape
    y = stack.astype(np.float64)
    ymax = y.max()
    if ymax > 0:
        y = y / ymax

    sizex = (nz, sy_s, sx_s)
    lamda = 1.0
    siranu = mu

    kernel_x = np.zeros(sizex)
    kernel_x[0, 0, :2] = [1, -1]
    xfft = fftn(kernel_x) * np.conj(fftn(kernel_x))

    kernel_y = np.zeros(sizex)
    kernel_y[0, :2, 0] = [1, -1]
    yfft = fftn(kernel_y) * np.conj(fftn(kernel_y))

    b7 = np.zeros(sizex)
    b8 = np.zeros(sizex)
    d7 = np.zeros(sizex)
    d8 = np.zeros(sizex)
    x = np.zeros(sizex)

    for it in range(n_iter):
        frac = (siranu / lamda) * y
        frac -= _bdiff(d7 - b7,2)
        frac -= _bdiff(d8 - b8,1)
        frac_f = fftn(frac)
        divide = (siranu / lamda) + xfft + yfft
        x = np.real(ifftn(frac_f / divide))

        u_x = _fdiff(x,2)
        u_y = _fdiff(x,1)

        s7 = np.abs(u_x + b7) - 1.0 / lamda
        s7[s7 < 0] = 0
        d7 = s7 * np.sign(u_x + b7)

        s8 = np.abs(u_y + b8) - 1.0 / lamda
        s8[s8 < 0] = 0
        d8 = s8 * np.sign(u_y + b8)

        b7 += u_x - d7
        b8 += u_y - d8
        x[x < 0] = 0

    x = x * ymax
    return x.astype(np.float32)
