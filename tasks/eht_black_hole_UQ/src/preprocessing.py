"""
Data Preprocessing for DPI (Deep Probabilistic Imaging)
========================================================

Handles loading EHT observation data (UVFITS), extracting closure quantity
indices, computing NUFFT parameters, and building the Gaussian prior image.

Pipeline: obs.uvfits + gt.fits + meta_data → preprocessed arrays for DPI training

Reference
---------
Sun & Bouman (2020), arXiv:2010.14462 — Deep Probabilistic Imaging
"""

import os
import json
import numpy as np
import torch

import ehtim as eh
import ehtim.const_def as ehc
from ehtim.observing.obs_helpers import NFFTInfo


def load_observation(data_dir: str = "data") -> dict:
    """
    Load EHT observation from UVFITS file.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing obs.uvfits.

    Returns
    -------
    dict with keys:
        'obs'       : ehtim Obsdata object
        'vis'       : (M,) complex ndarray — complex visibilities
        'vis_sigma' : (M,) ndarray — per-visibility noise standard deviation
        'uv_coords' : (M, 2) ndarray — (u, v) baseline coordinates in wavelengths
    """
    obs_path = os.path.join(data_dir, "obs.uvfits")
    obs = eh.obsdata.load_uvfits(obs_path)
    obs_data = obs.unpack(['u', 'v', 'vis', 'sigma'])
    return {
        "obs": obs,
        "vis": obs_data['vis'],
        "vis_sigma": obs_data['sigma'],
        "uv_coords": np.hstack((obs_data['u'].reshape(-1, 1),
                                 obs_data['v'].reshape(-1, 1))),
    }


def load_ground_truth(data_dir: str = "data", npix: int = 32,
                       fov_uas: float = 160.0) -> np.ndarray:
    """
    Load ground-truth image from FITS file and regrid to task resolution.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing gt.fits.
    npix : int
        Target image size in pixels.
    fov_uas : float
        Target field of view in microarcseconds.

    Returns
    -------
    (npix, npix) ndarray — ground-truth image.
    """
    gt_path = os.path.join(data_dir, "gt.fits")
    simim = eh.image.load_fits(gt_path)
    fov = fov_uas * eh.RADPERUAS
    simim = simim.regrid_image(fov, npix)
    return simim.imvec.reshape((npix, npix))


def load_metadata(data_dir: str = "data") -> dict:
    """
    Load imaging and training parameters from JSON metadata file.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing meta_data.

    Returns
    -------
    dict with all metadata fields (npix, fov_uas, n_flow, n_epoch, etc.)
    """
    path = os.path.join(data_dir, "meta_data")
    with open(path, "r") as f:
        return json.load(f)


def extract_closure_indices(obs, snrcut: float = 0.0) -> dict:
    """
    Extract closure phase triangle and closure amplitude quadrangle index maps.

    Maps each closure quantity to the corresponding visibility indices and signs,
    enabling efficient GPU computation of closure phases and log closure amplitudes.

    The index mapping uses a zero_symbol trick from the original DPI code to
    handle the case where a visibility index is 0 (which cannot be distinguished
    from "unset" when using signed indices).

    Parameters
    ----------
    obs : ehtim.Obsdata
        EHT observation object.
    snrcut : float
        SNR cutoff for closure quantity selection.

    Returns
    -------
    dict with keys:
        'cphase_ind_list'  : list of 3 int64 ndarrays — visibility indices for closure phase triangles
        'cphase_sign_list' : list of 3 float64 ndarrays — conjugation signs (+1 or -1)
        'camp_ind_list'    : list of 4 int64 ndarrays — visibility indices for closure amplitude quadrangles
        'cphase_data'      : structured array — closure phase data from ehtim
        'camp_data'        : structured array — closure amplitude data from ehtim
        'logcamp_data'     : structured array — log closure amplitude data from ehtim
    """
    # --- Closure phases ---
    if snrcut > 0:
        obs.add_cphase(count='min-cut0bl', uv_min=.1e9, snrcut=snrcut)
    else:
        obs.add_cphase(count='min-cut0bl', uv_min=.1e9)

    cphase_map = np.zeros((len(obs.cphase['time']), 3))
    zero_symbol = 100000

    for k1 in range(cphase_map.shape[0]):
        for k2 in list(np.where(obs.data['time'] == obs.cphase['time'][k1])[0]):
            if (obs.data['t1'][k2] == obs.cphase['t1'][k1] and
                    obs.data['t2'][k2] == obs.cphase['t2'][k1]):
                cphase_map[k1, 0] = zero_symbol if k2 == 0 else k2
            elif (obs.data['t2'][k2] == obs.cphase['t1'][k1] and
                  obs.data['t1'][k2] == obs.cphase['t2'][k1]):
                cphase_map[k1, 0] = -zero_symbol if k2 == 0 else -k2
            elif (obs.data['t1'][k2] == obs.cphase['t2'][k1] and
                  obs.data['t2'][k2] == obs.cphase['t3'][k1]):
                cphase_map[k1, 1] = zero_symbol if k2 == 0 else k2
            elif (obs.data['t2'][k2] == obs.cphase['t2'][k1] and
                  obs.data['t1'][k2] == obs.cphase['t3'][k1]):
                cphase_map[k1, 1] = -zero_symbol if k2 == 0 else -k2
            elif (obs.data['t1'][k2] == obs.cphase['t3'][k1] and
                  obs.data['t2'][k2] == obs.cphase['t1'][k1]):
                cphase_map[k1, 2] = zero_symbol if k2 == 0 else k2
            elif (obs.data['t2'][k2] == obs.cphase['t3'][k1] and
                  obs.data['t1'][k2] == obs.cphase['t1'][k1]):
                cphase_map[k1, 2] = -zero_symbol if k2 == 0 else -k2

    cphase_ind_list = []
    cphase_sign_list = []
    for col in range(3):
        raw = cphase_map[:, col]
        ind = np.abs(raw).astype(np.int64)
        ind[ind == zero_symbol] = 0
        cphase_ind_list.append(ind)
        cphase_sign_list.append(np.sign(raw))

    # --- Closure amplitudes ---
    if snrcut > 0:
        obs.add_camp(debias=True, count='min', snrcut=snrcut)
        obs.add_logcamp(debias=True, count='min', snrcut=snrcut)
    else:
        obs.add_camp(debias=True, count='min')
        obs.add_logcamp(debias=True, count='min')

    camp_map = np.zeros((len(obs.camp['time']), 6))
    zero_symbol = 10000

    for k1 in range(camp_map.shape[0]):
        for k2 in list(np.where(obs.data['time'] == obs.camp['time'][k1])[0]):
            if (obs.data['t1'][k2] == obs.camp['t1'][k1] and
                    obs.data['t2'][k2] == obs.camp['t2'][k1]):
                camp_map[k1, 0] = zero_symbol if k2 == 0 else k2
            elif (obs.data['t2'][k2] == obs.camp['t1'][k1] and
                  obs.data['t1'][k2] == obs.camp['t2'][k1]):
                camp_map[k1, 0] = -zero_symbol if k2 == 0 else -k2
            elif (obs.data['t1'][k2] == obs.camp['t1'][k1] and
                  obs.data['t2'][k2] == obs.camp['t3'][k1]):
                camp_map[k1, 1] = zero_symbol if k2 == 0 else k2
            elif (obs.data['t2'][k2] == obs.camp['t1'][k1] and
                  obs.data['t1'][k2] == obs.camp['t3'][k1]):
                camp_map[k1, 1] = -zero_symbol if k2 == 0 else -k2
            elif (obs.data['t1'][k2] == obs.camp['t1'][k1] and
                  obs.data['t2'][k2] == obs.camp['t4'][k1]):
                camp_map[k1, 2] = zero_symbol if k2 == 0 else k2
            elif (obs.data['t2'][k2] == obs.camp['t1'][k1] and
                  obs.data['t1'][k2] == obs.camp['t4'][k1]):
                camp_map[k1, 2] = -zero_symbol if k2 == 0 else -k2
            elif (obs.data['t1'][k2] == obs.camp['t2'][k1] and
                  obs.data['t2'][k2] == obs.camp['t3'][k1]):
                camp_map[k1, 3] = zero_symbol if k2 == 0 else k2
            elif (obs.data['t2'][k2] == obs.camp['t2'][k1] and
                  obs.data['t1'][k2] == obs.camp['t3'][k1]):
                camp_map[k1, 3] = -zero_symbol if k2 == 0 else -k2
            elif (obs.data['t1'][k2] == obs.camp['t2'][k1] and
                  obs.data['t2'][k2] == obs.camp['t4'][k1]):
                camp_map[k1, 4] = zero_symbol if k2 == 0 else k2
            elif (obs.data['t2'][k2] == obs.camp['t2'][k1] and
                  obs.data['t1'][k2] == obs.camp['t4'][k1]):
                camp_map[k1, 4] = -zero_symbol if k2 == 0 else -k2
            elif (obs.data['t1'][k2] == obs.camp['t3'][k1] and
                  obs.data['t2'][k2] == obs.camp['t4'][k1]):
                camp_map[k1, 5] = zero_symbol if k2 == 0 else k2
            elif (obs.data['t2'][k2] == obs.camp['t3'][k1] and
                  obs.data['t1'][k2] == obs.camp['t4'][k1]):
                camp_map[k1, 5] = -zero_symbol if k2 == 0 else -k2

    # Columns 0, 5, 2, 3 correspond to baselines (1-2), (3-4), (1-4), (2-3)
    # for log closure amplitude = log|V12| + log|V34| - log|V14| - log|V23|
    camp_ind_list = []
    for col in [0, 5, 2, 3]:
        raw = camp_map[:, col]
        ind = np.abs(raw).astype(np.int64)
        ind[ind == zero_symbol] = 0
        camp_ind_list.append(ind)

    return {
        "cphase_ind_list": cphase_ind_list,
        "cphase_sign_list": cphase_sign_list,
        "camp_ind_list": camp_ind_list,
        "cphase_data": obs.cphase,
        "camp_data": obs.camp,
        "logcamp_data": obs.logcamp,
    }


def compute_nufft_params(obs, npix: int, fov_uas: float) -> dict:
    """
    Compute NUFFT trajectory and pulse correction factor for torchkbnufft.

    Parameters
    ----------
    obs : ehtim.Obsdata
        EHT observation object.
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.

    Returns
    -------
    dict with keys:
        'ktraj_vis'     : (1, 2, M) torch.Tensor — scaled (v, u) trajectory
        'pulsefac_vis'  : (2, M) torch.Tensor — pulse function correction [real; imag]
    """
    fov = fov_uas * eh.RADPERUAS
    psize = fov / npix

    obs_data = obs.unpack(['u', 'v'])
    uv = np.hstack((obs_data['u'].reshape(-1, 1), obs_data['v'].reshape(-1, 1)))
    vu = np.hstack((obs_data['v'].reshape(-1, 1), obs_data['u'].reshape(-1, 1)))

    # Build a temporary ehtim image to get pulse function info
    simim = eh.image.make_square(obs, npix, fov)
    simim.ra = obs.ra
    simim.dec = obs.dec
    simim.rf = obs.rf

    fft_pad_factor = ehc.FFT_PAD_DEFAULT
    p_rad = ehc.GRIDDER_P_RAD_DEFAULT
    npad = int(fft_pad_factor * max(simim.xdim, simim.ydim))
    nfft_info_vis = NFFTInfo(simim.xdim, simim.ydim, simim.psize,
                              simim.pulse, npad, p_rad, uv)
    pulsefac_vis = nfft_info_vis.pulsefac

    # Scale trajectory for torchkbnufft: vu * psize * 2π
    vu_scaled = np.array(vu * simim.psize * 2 * np.pi)
    ktraj_vis = torch.tensor(vu_scaled.T, dtype=torch.float32).unsqueeze(0)
    pulsefac_vis_torch = torch.tensor(
        np.concatenate([np.expand_dims(pulsefac_vis.real, 0),
                        np.expand_dims(pulsefac_vis.imag, 0)], 0),
        dtype=torch.float32
    )

    return {
        "ktraj_vis": ktraj_vis,
        "pulsefac_vis": pulsefac_vis_torch,
    }


def build_prior_image(obs, npix: int, fov_uas: float,
                       prior_fwhm_uas: float = 50.0) -> tuple:
    """
    Build Gaussian prior image for the MEM regularizer.

    The prior is a circular Gaussian centered on the image, with flux set
    to the median APEX-ALMA baseline visibility amplitude.

    Parameters
    ----------
    obs : ehtim.Obsdata
        EHT observation object.
    npix : int
        Image size in pixels.
    fov_uas : float
        Field of view in microarcseconds.
    prior_fwhm_uas : float
        FWHM of the Gaussian prior in microarcseconds.

    Returns
    -------
    (prior_image, flux_const) : (ndarray, float)
        prior_image : (npix, npix) ndarray — prior image for MEM
        flux_const  : float — estimated total flux
    """
    fov = fov_uas * eh.RADPERUAS
    prior_fwhm = prior_fwhm_uas * eh.RADPERUAS

    flux_const = float(np.median(obs.unpack_bl('APEX', 'ALMA', 'amp')['amp']))

    prior = eh.image.make_square(obs, npix, fov)
    prior = prior.add_gauss(flux_const,
                             (prior_fwhm, prior_fwhm, 0, 0, 0))
    prior = prior.add_gauss(flux_const * 1e-6,
                             (prior_fwhm, prior_fwhm, 0, prior_fwhm, prior_fwhm))

    prior_image = prior.imvec.reshape((npix, npix))
    return prior_image, flux_const


def prepare_data(data_dir: str = "data") -> tuple:
    """
    Combined loader for all preprocessed DPI inputs.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    (obs, obs_data, closure_indices, nufft_params, prior_image, flux_const, metadata)
    """
    metadata = load_metadata(data_dir)
    npix = metadata["npix"]
    fov_uas = metadata["fov_uas"]
    prior_fwhm_uas = metadata["prior_fwhm_uas"]

    obs_data = load_observation(data_dir)
    obs = obs_data["obs"]

    closure_indices = extract_closure_indices(obs)
    nufft_params = compute_nufft_params(obs, npix, fov_uas)
    prior_image, flux_const = build_prior_image(obs, npix, fov_uas, prior_fwhm_uas)

    return obs, obs_data, closure_indices, nufft_params, prior_image, flux_const, metadata
