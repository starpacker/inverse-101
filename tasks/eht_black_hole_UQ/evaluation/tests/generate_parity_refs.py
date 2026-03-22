"""
Generate parity reference fixtures from the ORIGINAL DPI code.

Must be run in a separate process (not alongside cleaned code) due to
pynfft segfault when NFFTInfo is called twice.

Usage:
    cd tasks/eht_black_hole_UQ
    conda run -n dpi python evaluation/tests/generate_parity_refs.py
"""

import os
import sys
import numpy as np
import torch

ORIGINAL_DIR = "/home/groot/Documents/PKUlab/DPI/DPItorch"
DATA_DIR = "/home/groot/Documents/PKUlab/DPI/dataset/interferometry1"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "fixtures", "parity")

sys.path.insert(0, ORIGINAL_DIR)

import ehtim as eh
from interferometry_helpers import Obs_params_torch, eht_observation_pytorch
from torchkbnufft import KbNufft

os.makedirs(OUTPUT_DIR, exist_ok=True)

npix = 32
fov_uas = 160.0
fov = fov_uas * eh.RADPERUAS
prior_fwhm = 50 * eh.RADPERUAS

obs = eh.obsdata.load_uvfits(os.path.join(DATA_DIR, "obs.uvfits"))
flux_const = np.median(obs.unpack_bl('APEX', 'ALMA', 'amp')['amp'])
prior = eh.image.make_square(obs, npix, fov)
prior = prior.add_gauss(flux_const, (prior_fwhm, prior_fwhm, 0, 0, 0))
prior = prior.add_gauss(flux_const * 1e-6,
                         (prior_fwhm, prior_fwhm, 0, prior_fwhm, prior_fwhm))
simim = prior.copy()
simim.ra = obs.ra
simim.dec = obs.dec
simim.rf = obs.rf

print("Running Obs_params_torch...")
dft_mat, ktraj, pulsefac, cp_ind, cp_sign, ca_ind = \
    Obs_params_torch(obs, simim, snrcut=0.0, ttype='nfft')

np.savez(os.path.join(OUTPUT_DIR, "orig_preproc.npz"),
         ktraj=ktraj.numpy(),
         pulsefac=pulsefac.numpy(),
         cp_ind0=cp_ind[0].numpy(), cp_ind1=cp_ind[1].numpy(), cp_ind2=cp_ind[2].numpy(),
         cp_sign0=cp_sign[0].numpy(), cp_sign1=cp_sign[1].numpy(), cp_sign2=cp_sign[2].numpy(),
         ca_ind0=ca_ind[0].numpy(), ca_ind1=ca_ind[1].numpy(),
         ca_ind2=ca_ind[2].numpy(), ca_ind3=ca_ind[3].numpy(),
         flux_const=np.array(flux_const),
         prior_image=prior.imvec.reshape((npix, npix)))

# Forward model
nufft_ob = KbNufft(im_size=(npix, npix), numpoints=3)
device = torch.device("cpu")
eht_obs = eht_observation_pytorch(
    npix, nufft_ob, dft_mat, ktraj, pulsefac,
    cp_ind, cp_sign, ca_ind, device, ttype='nfft')

np.random.seed(42)
test_img = torch.tensor(np.abs(np.random.randn(4, npix, npix)).astype(np.float32))
vis, visamp, cphase, logcamp = eht_obs(test_img)
np.savez(os.path.join(OUTPUT_DIR, "orig_forward.npz"),
         vis=vis.detach().numpy(),
         visamp=visamp.detach().numpy(),
         cphase=cphase.detach().numpy(),
         logcamp=logcamp.detach().numpy())

print("Parity fixtures saved to:", OUTPUT_DIR)
