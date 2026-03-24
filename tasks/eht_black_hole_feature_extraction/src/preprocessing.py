"""
Data Preprocessing for α-DPI Feature Extraction
==================================================

Handles loading EHT observation data (UVFITS), extracting closure quantity
indices, computing NUFFT parameters, and estimating flux.

Pipeline: obs.uvfits + meta_data → preprocessed arrays for α-DPI training

Reference
---------
Sun et al. (2022), ApJ 932:99 — α-DPI
Original code: DPI/DPItorch/interferometry_helpers.py
"""

import os
import json
import numpy as np
import torch

import ehtim as eh


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


def load_ground_truth(data_dir: str = "data", npix: int = 64,
                       fov_uas: float = 120.0) -> np.ndarray:
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
    dict with all metadata fields.
    """
    path = os.path.join(data_dir, "meta_data")
    with open(path, "r") as f:
        return json.load(f)


def extract_closure_indices(obs, snrcut: float = 0.0) -> dict:
    """
    Extract closure phase triangle and closure amplitude quadrangle index maps.

    Maps each closure quantity to the corresponding visibility indices and signs,
    enabling efficient GPU computation of closure phases and log closure amplitudes.

    Parameters
    ----------
    obs : ehtim.Obsdata
        EHT observation object.
    snrcut : float
        SNR cutoff for closure quantity selection.

    Returns
    -------
    dict with keys:
        'cphase_ind_list'  : list of 3 int64 ndarrays
        'cphase_sign_list' : list of 3 float64 ndarrays
        'camp_ind_list'    : list of 4 int64 ndarrays
        'cphase_data'      : structured array — closure phase data
        'camp_data'        : structured array — closure amplitude data
        'logcamp_data'     : structured array — log closure amplitude data
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
        'pulsefac_vis'  : (2, M) torch.Tensor — pulse function correction
    """
    fov = fov_uas * eh.RADPERUAS
    psize = fov / npix

    obs_data = obs.unpack(['u', 'v'])
    uv = np.hstack((obs_data['u'].reshape(-1, 1), obs_data['v'].reshape(-1, 1)))
    vu = np.hstack((obs_data['v'].reshape(-1, 1), obs_data['u'].reshape(-1, 1)))

    # Compute pulse correction factor analytically for trianglePulse2D
    # pulsefac = sinc^2(u * psize) * sinc^2(v * psize) * exp(-i*pi*(u+v)*psize)
    # The phase factor accounts for the pixel grid offset
    pulsefac_amp = np.sinc(uv[:, 0] * psize) ** 2 * np.sinc(uv[:, 1] * psize) ** 2
    phase = -np.pi * (uv[:, 0] + uv[:, 1]) * psize
    pulsefac_real = pulsefac_amp * np.cos(phase)
    pulsefac_imag = pulsefac_amp * np.sin(phase)

    vu_scaled = np.array(vu * psize * 2 * np.pi)
    ktraj_vis = torch.tensor(vu_scaled.T, dtype=torch.float32).unsqueeze(0)
    pulsefac_vis_torch = torch.tensor(
        np.stack([pulsefac_real, pulsefac_imag], axis=0),
        dtype=torch.float32
    )

    return {
        "ktraj_vis": ktraj_vis,
        "pulsefac_vis": pulsefac_vis_torch,
    }


def estimate_flux(obs) -> float:
    """
    Estimate total flux from short-baseline visibility amplitudes.

    Parameters
    ----------
    obs : ehtim.Obsdata
        EHT observation object.

    Returns
    -------
    float — estimated total flux in Jy.
    """
    try:
        flux_const = float(np.median(obs.unpack_bl('AP', 'AA', 'amp')['amp']))
    except Exception:
        flux_const = float(np.median(obs.unpack_bl('APEX', 'ALMA', 'amp')['amp']))
    return flux_const


def prepare_data(data_dir: str = "data") -> tuple:
    """
    Combined loader for all preprocessed α-DPI inputs.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    (obs, obs_data, closure_indices, nufft_params, flux_const, metadata)
    """
    metadata = load_metadata(data_dir)
    npix = metadata["npix"]
    fov_uas = metadata["fov_uas"]

    obs_data = load_observation(data_dir)
    obs = obs_data["obs"]

    closure_indices = extract_closure_indices(obs)
    nufft_params = compute_nufft_params(obs, npix, fov_uas)
    flux_const = estimate_flux(obs)

    return obs, obs_data, closure_indices, nufft_params, flux_const, metadata
