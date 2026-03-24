"""
Physics module: forward model operators and physical constants.

Contains OTF generation, finite difference operators, Fourier shift operators,
and other physics-related utilities used by preprocessing and solver modules.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PyEMD import EMD as PyEMD_EMD


def generate_otf(n, na, wavelength, pixel_size):
    """Generate a synthetic 2D OTF (optical transfer function).

    Parameters
    ----------
    n : int
        Size of the OTF array (n x n).
    na : float
        Numerical aperture.
    wavelength : float
        Emission wavelength in nm.
    pixel_size : float
        Pixel size in nm.

    Returns
    -------
    otf : ndarray (n, n), normalized to [0, 1].
    """
    cutoff_freq = 2 * na / (wavelength * 1e-3)  # in 1/um
    freq_per_pixel = 1.0 / (n * pixel_size * 1e-3)  # in 1/um
    fc_pixels = cutoff_freq / freq_per_pixel

    ky, kx = np.mgrid[-n // 2:n // 2, -n // 2:n // 2]
    kr = np.sqrt(kx ** 2 + ky ** 2)

    otf = np.clip(1.0 - kr / fc_pixels, 0, 1)
    otf = otf / otf.max()
    return otf.astype(np.float64)


def pad_to_size(arr, target_shape):
    """Zero-pad a 2D array to *target_shape*, centred."""
    sy, sx = arr.shape
    ty, tx = target_shape
    py = (ty - sy) // 2
    px = (tx - sx) // 2
    out = np.zeros(target_shape, dtype=arr.dtype)
    out[py:py + sy, px:px + sx] = arr
    return out


def dft_conv(h, g):
    """FFT-based linear convolution of 2D arrays. Port of MATLAB dft.m."""
    K = np.array(h.shape)
    L = np.array(g.shape)
    N = K + L
    H = fftshift(fft2(ifftshift(h), s=tuple(N)))
    G = fftshift(fft2(ifftshift(g), s=tuple(N)))
    result = fftshift(ifft2(ifftshift(H * G)))
    out_size = K + L - 1
    start = ((N - out_size) // 2).astype(int)
    return result[start[0]:start[0] + out_size[0], start[1]:start[1] + out_size[1]]


def shift_otf(H_2n, kx, ky, n):
    """Shift OTF by (kx, ky) via Fourier shift theorem.

    Computes: F{ F^{-1}[H_2n] * exp(j*(kx*xx + ky*yy)) }
    """
    xx, yy = np.meshgrid(np.arange(2 * n), np.arange(2 * n))
    Ir = np.exp(1j * (kx * xx + ky * yy))
    h_real = fftshift(ifft2(ifftshift(H_2n)))
    return fftshift(fft2(ifftshift(h_real * Ir)))


def emd_decompose(signal):
    """Compute EMD of a 1D real signal. Returns IMFs as rows (like MATLAB emd)."""
    emd_obj = PyEMD_EMD()
    emd_obj.FIXE_H = 0
    t = np.arange(len(signal))
    try:
        imfs = emd_obj.emd(signal.astype(np.float64), t)
    except Exception:
        return signal.reshape(1, -1)
    return imfs


def compute_merit(H, H1, sp_center, sp_shifted, kx, ky, n):
    """Compute cross-correlation merit function for pattern frequency (kx, ky)."""
    replcHtest = shift_otf(H1, kx, ky, n)
    mask = np.zeros_like(replcHtest, dtype=np.float64)
    mask[np.abs(replcHtest) > 0.9] = 1.0
    replcHtest = mask

    replch = shift_otf(H, kx, ky, n)
    replch = replch * replcHtest

    sp_shifted_real = fftshift(ifft2(ifftshift(sp_shifted)))
    xx, yy = np.meshgrid(np.arange(2 * n), np.arange(2 * n))
    Ir = np.exp(1j * (kx * xx + ky * yy))
    replctest = fftshift(fft2(ifftshift(sp_shifted_real * Ir)))

    youhua = replctest * replcHtest * H
    he_num = np.abs(np.sum(np.conj(youhua) * (sp_center * replch)))
    he_den = np.sum(H1 * replcHtest)
    return he_num / (he_den + 1e-12)
