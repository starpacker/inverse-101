"""
Generate parity reference fixtures using the CLEANED src/ code.

Uses the self-contained analytical implementations in src/preprocessing.py
and src/physics_model.py, which do NOT depend on pynfft/NFFTInfo.

Usage:
    cd tasks/eht_black_hole_UQ
    conda run -n eht_audit python evaluation/tests/generate_parity_refs.py
"""

import os
import sys
import numpy as np
import torch

TASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.join(TASK_DIR, "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "fixtures", "parity")

sys.path.insert(0, TASK_DIR)

import ehtim as eh
from src.preprocessing import extract_closure_indices, compute_nufft_params, build_prior_image
from src.physics_model import NUFFTForwardModel

os.makedirs(OUTPUT_DIR, exist_ok=True)

npix = 32
fov_uas = 160.0
prior_fwhm_uas = 50.0

# --- Preprocessing ---
obs = eh.obsdata.load_uvfits(os.path.join(DATA_DIR, "obs.uvfits"))

print("Extracting closure indices...")
closure = extract_closure_indices(obs)

print("Computing NUFFT params...")
obs2 = eh.obsdata.load_uvfits(os.path.join(DATA_DIR, "obs.uvfits"))
nufft = compute_nufft_params(obs2, npix, fov_uas)

print("Building prior image...")
obs3 = eh.obsdata.load_uvfits(os.path.join(DATA_DIR, "obs.uvfits"))
prior_image, flux_const = build_prior_image(obs3, npix, fov_uas, prior_fwhm_uas)

np.savez(
    os.path.join(OUTPUT_DIR, "orig_preproc.npz"),
    ktraj=nufft['ktraj_vis'].numpy(),
    pulsefac=nufft['pulsefac_vis'].numpy(),
    cp_ind0=np.array(closure['cphase_ind_list'][0]),
    cp_ind1=np.array(closure['cphase_ind_list'][1]),
    cp_ind2=np.array(closure['cphase_ind_list'][2]),
    cp_sign0=np.array(closure['cphase_sign_list'][0]),
    cp_sign1=np.array(closure['cphase_sign_list'][1]),
    cp_sign2=np.array(closure['cphase_sign_list'][2]),
    ca_ind0=np.array(closure['camp_ind_list'][0]),
    ca_ind1=np.array(closure['camp_ind_list'][1]),
    ca_ind2=np.array(closure['camp_ind_list'][2]),
    ca_ind3=np.array(closure['camp_ind_list'][3]),
    flux_const=np.array(flux_const),
    prior_image=prior_image,
)
print("Saved orig_preproc.npz")

# --- Forward model ---
device = torch.device("cpu")
cphase_ind_t = [torch.tensor(a, dtype=torch.long) for a in closure['cphase_ind_list']]
cphase_sign_t = [torch.tensor(a, dtype=torch.float32) for a in closure['cphase_sign_list']]
camp_ind_t = [torch.tensor(a, dtype=torch.long) for a in closure['camp_ind_list']]

fwd = NUFFTForwardModel(
    npix, nufft['ktraj_vis'], nufft['pulsefac_vis'],
    cphase_ind_t, cphase_sign_t, camp_ind_t, device)

np.random.seed(42)
test_img = torch.tensor(np.abs(np.random.randn(4, npix, npix)).astype(np.float32))
vis, visamp, cphase, logcamp = fwd(test_img)

np.savez(
    os.path.join(OUTPUT_DIR, "orig_forward.npz"),
    vis=vis.detach().numpy(),
    visamp=visamp.detach().numpy(),
    cphase=cphase.detach().numpy(),
    logcamp=logcamp.detach().numpy(),
)
print("Saved orig_forward.npz")

print("Parity fixtures saved to:", OUTPUT_DIR)
