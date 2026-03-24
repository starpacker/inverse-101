"""
Preprocessing module: raw SIM data → super-resolved image y.

Includes SIM parameter estimation, modulation/phase estimation,
Wiener-SIM reconstruction (phase separation + frequency shifting via
the Moiré effect + Wiener recombination), and temporal running average.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import zoom as ndizoom

from .physics import (
    pad_to_size, shift_otf, dft_conv,
    compute_merit, emd_decompose,
)


# ---------------------------------------------------------------------------
# Stage 1: SIM parameter estimation (port of mysim3_512_fast_21.m)
# ---------------------------------------------------------------------------

def estimate_sim_parameters(raw_frames, otf, nangles, nphases, n,
                            pg, fanwei, regul, spjg, beishu_an=1):
    """Estimate SIM pattern frequencies via cross-correlation + sub-pixel refinement.

    Returns
    -------
    zuobiaox, zuobiaoy : ndarray (nangles*nphases, 1)
        Pattern frequency coordinates in the 2n grid.
    """
    nframes = nangles * nphases
    sy, sx = raw_frames.shape[1], raw_frames.shape[2]

    fd = np.zeros((sy, sx, nframes), dtype=np.float64)
    for i in range(nframes):
        indices = list(range(i, nframes * beishu_an, nframes))
        fd[:, :, i] = raw_frames[indices].mean(axis=0)

    total_spjg = sum(spjg)
    phase_matrix = np.zeros((nphases, nphases), dtype=np.complex128)
    for j in range(nphases):
        cum = sum(spjg[:j]) / total_spjg if j > 0 else 0.0
        phase_matrix[j, 0] = 1.0
        phase_matrix[j, 1] = np.exp(1j * regul * cum)
        phase_matrix[j, 2] = np.exp(-1j * regul * cum)
    inv_pm = np.linalg.inv(phase_matrix)

    H = otf.copy().astype(np.float64)
    fc = int(np.ceil(220 * (n / 512)))
    ky_grid, kx_grid = np.mgrid[-n // 2:n // 2, -n // 2:n // 2]
    kr = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    H[kr > fc] = 0
    H = np.abs(H)
    H1 = (H != 0).astype(np.float64)
    H2 = H1.copy()

    fd_n = np.zeros((n, n, nframes), dtype=np.float64)
    py = (n - sy) // 2
    px = (n - sx) // 2
    for i in range(nframes):
        fd_n[py:py + sy, px:px + sx, i] = fd[:, :, i]
    DIbars = np.zeros((n, n, nframes), dtype=np.complex128)
    for i in range(nframes):
        DIbars[:, :, i] = fftshift(fft2(ifftshift(fd_n[:, :, i])))

    H9 = np.tile(H1[:, :, np.newaxis], (1, 1, nframes))
    DIbars = H9 * DIbars

    sp = np.zeros((n, n, nframes), dtype=np.complex128)
    for itheta in range(nangles):
        for j in range(nphases):
            for k in range(nphases):
                idx_in = itheta * nphases + k
                idx_out = itheta * nphases + j
                sp[:, :, idx_out] += inv_pm[j, k] * DIbars[:, :, idx_in]

    sp = sp / (np.abs(sp) + 1e-12)

    H_2n = pad_to_size(H, (2 * n, 2 * n))
    H1_2n = pad_to_size(H1, (2 * n, 2 * n))

    zuobiaox = np.full((nframes, 1), float(n))
    zuobiaoy = np.full((nframes, 1), float(n))

    for spi in range(0, nangles * (nphases - 1), 2):
        spi_m = spi + 1
        idx_center = int(np.ceil(spi_m / 2) * nphases - 1) - 1
        idx_shifted = int(np.ceil(spi_m / 2) + 2 * np.floor(spi_m / 2)) - 1

        sp_center = sp[:, :, idx_center]
        sp_shifted = sp[:, :, idx_shifted]

        H2_flip = H2[::-1, ::-1]
        cishu = dft_conv(H2, H2_flip)
        ci = cishu.copy()
        ci[np.abs(ci) < 0.9] = 0
        ci[ci != 0] = 1

        sp_conj_flip = np.conj(sp_shifted)[::-1, ::-1]
        jieguo = dft_conv(sp_center, sp_conj_flip)
        jieguo = jieguo * ci
        lihe = np.abs(jieguo / (cishu + 1e-12))

        lh, lw = lihe.shape
        ky_bp = np.arange(-lh // 2, lh // 2 + 1)[:lh]
        kx_bp = np.arange(-lw // 2, lw // 2 + 1)[:lw]
        KX, KY = np.meshgrid(kx_bp, ky_bp)
        kr_bp = np.sqrt(KX ** 2 + KY ** 2)
        mask_bp = (kr_bp >= (pg - fanwei)) & (kr_bp <= (pg + fanwei))
        lihe[~mask_bp] = 0

        peak = np.unravel_index(np.argmax(np.abs(lihe)), lihe.shape)
        maxx = float(peak[0])
        maxy = float(peak[1])

        sp_center_2n = pad_to_size(sp_center, (2 * n, 2 * n))
        sp_shifted_2n = pad_to_size(sp_shifted, (2 * n, 2 * n))

        kx0 = 2 * np.pi * (maxy - n) / (2 * n)
        ky0 = 2 * np.pi * (maxx - n) / (2 * n)
        he = compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx0, ky0, n)

        eps_sub = 1e-5
        step_x = 1.0
        step_y = 1.0

        def _grad_dir_maxx():
            ky_m = 2 * np.pi * (maxx - eps_sub - n) / (2 * n)
            ky_p = 2 * np.pi * (maxx + eps_sub - n) / (2 * n)
            kx_c = 2 * np.pi * (maxy - n) / (2 * n)
            hm = compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx_c, ky_m, n)
            hp = compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx_c, ky_p, n)
            return -1 if hm > hp else (1 if hp > hm else None), hm, hp

        def _grad_dir_maxy():
            kx_m = 2 * np.pi * (maxy - eps_sub - n) / (2 * n)
            kx_p = 2 * np.pi * (maxy + eps_sub - n) / (2 * n)
            ky_c = 2 * np.pi * (maxx - n) / (2 * n)
            hm = compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx_m, ky_c, n)
            hp = compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n, kx_p, ky_c, n)
            return -1 if hm > hp else (1 if hp > hm else None), hm, hp

        d, _, _ = _grad_dir_maxx()
        flag_maxx = d if d is not None else 1
        d, _, _ = _grad_dir_maxy()
        flag_maxy = d if d is not None else 1

        while step_x > 1e-4 or step_y > 1e-4:
            if step_x > 1e-4:
                d, _, _ = _grad_dir_maxx()
                if d is not None:
                    flag_maxx = d
                else:
                    flag_maxx = -flag_maxx
                while step_x > 1e-4:
                    maxx_try = maxx + flag_maxx * step_x
                    ky_try = 2 * np.pi * (maxx_try - n) / (2 * n)
                    kx_try = 2 * np.pi * (maxy - n) / (2 * n)
                    he_try = compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n,
                                           kx_try, ky_try, n)
                    if he_try > he:
                        he = he_try
                        maxx = maxx_try
                        break
                    else:
                        step_x *= 0.5

            if step_y > 1e-4:
                d, _, _ = _grad_dir_maxy()
                if d is not None:
                    flag_maxy = d
                else:
                    flag_maxy = -flag_maxy
                while step_y > 1e-4:
                    maxy_try = maxy + flag_maxy * step_y
                    ky_try = 2 * np.pi * (maxx - n) / (2 * n)
                    kx_try = 2 * np.pi * (maxy_try - n) / (2 * n)
                    he_try = compute_merit(H_2n, H1_2n, sp_center_2n, sp_shifted_2n,
                                           kx_try, ky_try, n)
                    if he_try > he:
                        he = he_try
                        maxy = maxy_try
                        break
                    else:
                        step_y *= 0.5

        idx1 = spi_m + 1 + int(np.floor((spi_m - 1) / 2)) - 1
        idx2 = idx1 + 1
        zuobiaox[idx1, 0] = maxx
        zuobiaoy[idx1, 0] = maxy
        zuobiaox[idx2, 0] = 2 * n - maxx
        zuobiaoy[idx2, 0] = 2 * n - maxy
        print(f"  Angle {spi // 2}: pattern freq at ({maxx:.4f}, {maxy:.4f})")

    return zuobiaox, zuobiaoy


# ---------------------------------------------------------------------------
# Stage 2: Modulation depth & phase estimation
#           (port of SIM_512_fast_22_wl9.m lines 1-199)
# ---------------------------------------------------------------------------

def estimate_modulation_and_phase(raw_frames, otf, zuobiaox, zuobiaoy,
                                  nangles, nphases, n, regul, spjg,
                                  wavelength, beishu_an=1, bg=None):
    """Estimate modulation depth (c6) and phase offsets (angle6) via EMD.

    Returns
    -------
    c6 : ndarray (nangles*(nphases-1),) – modulation depths
    angle6 : ndarray (nangles*(nphases-1),) – phase offsets
    """
    n_512 = n
    nframes = nangles * nphases
    sy, sx = raw_frames.shape[1], raw_frames.shape[2]

    fc_ang = int(np.ceil(120 * (n_512 / 512)))
    fc_con = int(np.ceil(80 * (n_512 / 512))) if wavelength == 647 else int(np.ceil(105 * (n_512 / 512)))

    scale = n_512 / np.floor((zuobiaox[0, 0] + zuobiaoy[0, 0]) / 2)
    zuobiaox_512 = zuobiaox * scale
    zuobiaoy_512 = zuobiaoy * scale

    ky_grid, kx_grid = np.mgrid[-n_512 // 2:n_512 // 2, -n_512 // 2:n_512 // 2]
    k_r = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    H_ang = np.abs(otf.copy().astype(np.float64))
    H_con = np.abs(otf.copy().astype(np.float64))
    H_ang[k_r > fc_ang] = 0
    H_con[k_r > fc_con] = 0
    H1_ang = (H_ang != 0).astype(np.float64)
    H1_con = (H_con != 0).astype(np.float64)

    total_spjg = sum(spjg)
    phase_matrix = np.zeros((nphases, nphases), dtype=np.complex128)
    for j in range(nphases):
        cum = sum(spjg[:j]) / total_spjg if j > 0 else 0.0
        phase_matrix[j, 0] = 1.0
        phase_matrix[j, 1] = np.exp(1j * regul * cum)
        phase_matrix[j, 2] = np.exp(-1j * regul * cum)
    inv_pm = np.linalg.inv(phase_matrix)

    fd = np.zeros((sy, sx, nframes), dtype=np.float64)
    for i in range(nframes):
        indices = list(range(i, nframes * beishu_an, nframes))
        fd[:, :, i] = raw_frames[indices].mean(axis=0)

    if bg is not None:
        bgy, bgx = bg.shape
        cy, cx = bgy // 2, bgx // 2
        bg_crop = bg[cy - sy // 2:cy + sy // 2, cx - sx // 2:cx + sx // 2]
        if bg_crop.shape != (sy, sx):
            bg_crop = np.zeros((sy, sx))
    else:
        bg_crop = np.zeros((sy, sx))

    py = (n_512 - sy) // 2
    px = (n_512 - sx) // 2
    DIbars = np.zeros((n_512, n_512, nframes), dtype=np.complex128)
    for i in range(nframes):
        frame = fd[:, :, i] - bg_crop
        tmp = np.zeros((n_512, n_512), dtype=np.float64)
        tmp[py:py + sy, px:px + sx] = frame
        DIbars[:, :, i] = fftshift(fft2(ifftshift(tmp)))

    sp = np.zeros((n_512, n_512, nframes), dtype=np.complex128)
    for itheta in range(nangles):
        for j in range(nphases):
            for k in range(nphases):
                idx_in = itheta * nphases + k
                idx_out = itheta * nphases + j
                sp[:, :, idx_out] += inv_pm[j, k] * DIbars[:, :, idx_in]

    H_ang_2n = pad_to_size(H_ang, (2 * n_512, 2 * n_512))
    H_con_2n = pad_to_size(H_con, (2 * n_512, 2 * n_512))
    H1_ang_2n = pad_to_size(H1_ang, (2 * n_512, 2 * n_512))
    H1_con_2n = pad_to_size(H1_con, (2 * n_512, 2 * n_512))

    n_orders = nangles * (nphases - 1)
    c6 = np.zeros(n_orders)
    angle6 = np.zeros(n_orders)
    xx, yy = np.meshgrid(np.arange(2 * n_512), np.arange(2 * n_512))

    for spi in range(n_orders):
        spi_m = spi + 1
        idx_center = int(np.ceil(spi_m / 2)) * 3 - 1 - 1
        idx_shifted = int(np.ceil(spi_m / 2)) + 2 * int(np.floor(spi_m / 2)) - 1

        spzhongxin = pad_to_size(sp[:, :, idx_center], (2 * n_512, 2 * n_512))
        spzhongyiwei = pad_to_size(sp[:, :, idx_shifted], (2 * n_512, 2 * n_512))

        coord_idx = spi_m + 1 + int(np.floor((spi_m - 1) / 2)) - 1
        kytest = 2 * np.pi * (zuobiaox_512[coord_idx, 0] - n_512) / (2 * n_512)
        kxtest = 2 * np.pi * (zuobiaoy_512[coord_idx, 0] - n_512) / (2 * n_512)

        Ir = np.exp(1j * (kxtest * xx + kytest * yy))

        replcHtest_ang = shift_otf(H1_ang_2n, kxtest, kytest, n_512)
        mask_ang = np.zeros_like(replcHtest_ang, dtype=np.float64)
        mask_ang[np.abs(replcHtest_ang) > 0.9] = 1.0
        replcHtest_ang = mask_ang

        replcHtest_con = shift_otf(H1_con_2n, kxtest, kytest, n_512)
        mask_con = np.zeros_like(replcHtest_con, dtype=np.float64)
        mask_con[np.abs(replcHtest_con) > 0.9] = 1.0
        replcHtest_con = mask_con

        replch_ang = shift_otf(H_ang_2n, kxtest, kytest, n_512) * replcHtest_ang
        replch_con = shift_otf(H_con_2n, kxtest, kytest, n_512) * replcHtest_con

        sp_real = fftshift(ifft2(ifftshift(spzhongyiwei)))
        replctest = fftshift(fft2(ifftshift(sp_real * Ir)))

        youhua_ang = replctest * replcHtest_ang * H_ang_2n
        youhua_con = replctest * replcHtest_con * H_con_2n
        chongdie_ang = spzhongxin * replch_ang
        chongdie_con = spzhongxin * replch_con

        cm_ang = youhua_ang / (chongdie_ang + 1e-12)
        cm_con = youhua_con / (chongdie_con + 1e-12)

        angcm = np.angle(cm_ang)
        abscm = np.abs(cm_con)

        replc6_ang = replch_ang * H_ang_2n
        replc6_con = replch_con * H_con_2n

        mask_nz_ang = (replc6_ang.ravel() != 0)
        phase_vals = angcm.ravel()[mask_nz_ang]
        mask_nz_con = (replc6_con.ravel() != 0)
        mod_vals = abscm.ravel()[mask_nz_con]

        jdjd = 0.02
        jdc = 0.02
        f_bins = np.arange(-np.pi, np.pi + jdjd, jdjd)
        ff_bins = np.arange(0, 0.6 + jdc, jdc)

        g_hist = np.histogram(np.real(phase_vals), bins=f_bins)[0].astype(np.float64)
        gg_hist = np.histogram(np.real(mod_vals), bins=ff_bins)[0].astype(np.float64)

        if len(g_hist) > 10:
            try:
                imf = emd_decompose(g_hist)
                if imf.shape[0] >= 5 and np.max(g_hist) < 50:
                    g_smooth = np.sum(imf[4:], axis=0)
                elif imf.shape[0] >= 4:
                    g_smooth = np.sum(imf[3:], axis=0)
                else:
                    g_smooth = g_hist
            except Exception:
                g_smooth = g_hist
        else:
            g_smooth = g_hist

        if len(gg_hist) > 10:
            try:
                imf2 = emd_decompose(gg_hist)
                gg_smooth = np.sum(imf2, axis=0)
            except Exception:
                gg_smooth = gg_hist
        else:
            gg_smooth = gg_hist

        h_idx = np.argmax(g_smooth)
        angle6[spi] = -np.pi + (h_idx + 1) * jdjd

        hh_idx = np.where(gg_smooth == np.max(gg_smooth))[0]
        c6[spi] = jdc * np.mean(hh_idx + 1)

        print(f"  Order {spi}: angle6={angle6[spi]:.4f}, c6={c6[spi]:.4f}")

    return c6, angle6


# ---------------------------------------------------------------------------
# Stage 3: Wiener SIM reconstruction
#           (port of SIM_512_fast_22_wl9.m lines 200-430)
# ---------------------------------------------------------------------------

def wiener_sim_reconstruct(raw_frames, otf, zuobiaox, zuobiaoy,
                           c6, angle6,
                           nangles, nphases, n, weilac, regul, spjg,
                           beishu_re=1, starframe=0, bg=None):
    """Wiener-filtered SIM reconstruction with Moiré frequency unmixing.

    Performs phase separation via matrix inversion, shifts frequency bands
    to their correct positions using the Fourier shift theorem (Moiré effect),
    and recombines with Wiener filtering.

    Returns
    -------
    result_stack : ndarray (n_timepoints, 2*sy, 2*sx)
    """
    nframes = nangles * nphases
    sy, sx = raw_frames.shape[1], raw_frames.shape[2]
    total_frames = raw_frames.shape[0]

    image_size = max(sy, sx)
    if image_size <= 256:
        otfx = 256
    elif image_size <= 512:
        otfx = 512
    else:
        otfx = image_size

    zuobiaox_r = zuobiaox * (otfx / n)
    zuobiaoy_r = zuobiaoy * (otfx / n)
    n_work = otfx

    num_images = total_frames - starframe
    n_timepoints = num_images // (nframes * beishu_re)

    xishu = np.ones(nframes)
    for itheta in range(nangles):
        c6_pair = c6[itheta * (nphases - 1):(itheta + 1) * (nphases - 1)]
        avg_inv = np.mean(1.0 / (c6_pair + 1e-12))
        xishu[itheta * nphases + 1] = avg_inv
        xishu[itheta * nphases + 2] = avg_inv
    print(f"  Modulation depth (xishu): {xishu}")

    deph = np.zeros(nangles)
    for itheta in range(nangles):
        a6_pair = angle6[itheta * (nphases - 1):(itheta + 1) * (nphases - 1)]
        deph[itheta] = np.sign(a6_pair[0]) * np.mean(np.abs(a6_pair))

    fc = int(np.ceil(200 * (n_work / 512)))
    if otf.shape[0] != n_work or otf.shape[1] != n_work:
        otf_r = ndizoom(otf, n_work / otf.shape[0], order=1)
    else:
        otf_r = otf.copy()

    ky_grid, kx_grid = np.mgrid[-n_work // 2:n_work // 2, -n_work // 2:n_work // 2]
    kr = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    jiequ = np.ones((n_work, n_work))
    jiequ[kr > fc] = 0

    H = otf_r.astype(np.float64) * jiequ
    H = H / (H.max() + 1e-12)
    H1 = (H != 0).astype(np.float64)

    Hk = pad_to_size(H, (2 * n_work, 2 * n_work))
    H1_2n = pad_to_size(H1, (2 * n_work, 2 * n_work))

    xx, yy = np.meshgrid(np.arange(2 * n_work), np.arange(2 * n_work))
    Irtest = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.complex128)
    replcHtest = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.float64)
    replch = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.complex128)

    for ii in range(nframes):
        kytest = 2 * np.pi * (zuobiaox_r[ii, 0] - n_work) / (2 * n_work)
        kxtest = 2 * np.pi * (zuobiaoy_r[ii, 0] - n_work) / (2 * n_work)
        Irtest[:, :, ii] = np.exp(1j * (kxtest * xx + kytest * yy))
        replcHtest[:, :, ii] = np.real(shift_otf(H1_2n, kxtest, kytest, n_work))
        replch[:, :, ii] = shift_otf(Hk, kxtest, kytest, n_work)

    replcHtest[np.abs(replcHtest) > 0.9] = 1.0
    replcHtest[np.abs(replcHtest) != 1.0] = 0.0
    replch = replch * replcHtest

    re = replch.copy()
    absre = np.abs(re)
    absre_mask = (absre != 0).astype(np.float64)
    re = re * absre_mask
    re[re == 0] = 1e19
    re = re / np.abs(re)

    reh = np.abs(replch)
    hs = np.sum(reh ** 2, axis=2)

    plong = int(np.floor(
        np.sum(np.sqrt((zuobiaox_r[:, 0] - zuobiaox_r[0, 0]) ** 2 +
                       (zuobiaoy_r[:, 0] - zuobiaoy_r[0, 0]) ** 2)) / (2 * nangles)
    ))
    k_max = plong + fc
    ky_2n, kx_2n = np.mgrid[-n_work:n_work, -n_work:n_work]
    kr_2n = np.sqrt(kx_2n ** 2 + ky_2n ** 2)
    bhs = np.cos(np.pi * kr_2n / (2 * k_max))
    bhs[kr_2n > k_max] = 0

    total_spjg = sum(spjg)
    inv_phase_matrices = np.zeros((nphases, nphases, nangles), dtype=np.complex128)
    for itheta in range(nangles):
        phi_tmp = np.linspace(0 + deph[itheta], regul + deph[itheta], total_spjg + 1)
        phi = np.zeros(nphases)
        phi[0] = phi_tmp[0]
        phi[1] = phi_tmp[int(spjg[0])]
        phi[2] = phi_tmp[int(spjg[0] + spjg[1])]
        pm = np.zeros((nphases, nphases), dtype=np.complex128)
        for j in range(nphases):
            pm[j, 0] = 1.0
            pm[j, 1] = np.exp(1j * phi[j])
            pm[j, 2] = np.exp(-1j * phi[j])
        inv_phase_matrices[:, :, itheta] = np.linalg.inv(pm)

    sig = 0.25
    x_mask = np.arange(1, sx + 1)
    y_mask = np.arange(1, sy + 1)[:, np.newaxis]
    _sig = lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))
    mask = (_sig(sig * x_mask) - _sig(sig * (x_mask - sx - 1))) * \
           (_sig(sig * y_mask) - _sig(sig * (y_mask - sy - 1)))
    mask9 = np.tile(mask[:, :, np.newaxis] ** 3, (1, 1, nframes))

    if bg is not None:
        bgy, bgx = bg.shape
        cy, cx = bgy // 2, bgx // 2
        bg_crop = bg[cy - sy // 2:cy + sy // 2, cx - sx // 2:cx + sx // 2]
        if bg_crop.shape != (sy, sx):
            bg_crop = np.zeros((sy, sx))
    else:
        bg_crop = np.zeros((sy, sx))

    result_stack = []
    zhen = starframe
    while zhen <= num_images - nframes * beishu_re + starframe:
        D = raw_frames[zhen:zhen + nframes * beishu_re].astype(np.float64)
        for i in range(D.shape[0]):
            D[i] -= bg_crop

        fd = np.zeros((sy, sx, nframes), dtype=np.float64)
        for i in range(nframes):
            indices = list(range(i, nframes * beishu_re, nframes))
            fd[:, :, i] = D[indices].mean(axis=0)

        fd *= mask9

        py = (n_work - sy) // 2
        px = (n_work - sx) // 2
        DIbar = np.zeros((n_work, n_work, nframes), dtype=np.complex128)
        for i in range(nframes):
            tmp = np.zeros((n_work, n_work), dtype=np.float64)
            tmp[py:py + sy, px:px + sx] = fd[:, :, i]
            DIbar[:, :, i] = fftshift(fft2(ifftshift(tmp)))

        sp = np.zeros((n_work, n_work, nframes), dtype=np.complex128)
        for itheta in range(nangles):
            inv_pm = inv_phase_matrices[:, :, itheta]
            for j in range(nphases):
                for k in range(nphases):
                    idx_in = itheta * nphases + k
                    idx_out = itheta * nphases + j
                    sp[:, :, idx_out] += inv_pm[j, k] * DIbar[:, :, idx_in]

        jiequ9 = np.tile(jiequ[:, :, np.newaxis], (1, 1, nframes))
        sp = sp * jiequ9

        retirff = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.complex128)
        for ii in range(nframes):
            sp_padded = pad_to_size(sp[:, :, ii], (2 * n_work, 2 * n_work))
            sp_real = fftshift(ifft2(ifftshift(sp_padded)))
            retirff[:, :, ii] = fftshift(fft2(ifftshift(sp_real * Irtest[:, :, ii])))

        retirff = retirff * replcHtest
        retirff = retirff / re

        tmprc = np.zeros((2 * n_work, 2 * n_work, nframes), dtype=np.complex128)
        for t in range(nframes):
            tmprc[:, :, t] = (xishu[t] * retirff[:, :, t] * np.conj(reh[:, :, t])) / \
                             (hs + 0.005 * nangles * weilac ** 2)

        dr = np.sum(tmprc, axis=2)
        drr = dr * bhs
        fimage = fftshift(ifft2(ifftshift(drr)))

        cy, cx = n_work, n_work
        result = np.real(fimage[cy - sy:cy + sy, cx - sx:cx + sx])
        result[result < 0] = 0
        result_stack.append(result.astype(np.float32))

        zhen += nframes * beishu_re

    return np.array(result_stack)


# ---------------------------------------------------------------------------
# Stage 6: Running average
# ---------------------------------------------------------------------------

def running_average(stack, window=3):
    """Temporal running average. Exact port of Running_average.m."""
    if stack.ndim < 3 or stack.shape[0] < 3:
        return stack

    nz = stack.shape[0]
    result = np.zeros_like(stack)
    half_before = (window - 1) // 2
    half_after = window // 2

    for i in range(half_before, nz - half_after):
        result[i] = np.mean(stack[i - half_before:i + half_after + 1], axis=0)

    return result
