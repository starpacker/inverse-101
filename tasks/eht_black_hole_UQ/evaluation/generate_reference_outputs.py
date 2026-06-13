"""
Generate reference outputs from the original DPI training checkpoint.

Run after training the original DPI code:
    cd tasks/eht_black_hole_UQ
    conda run -n dpi python evaluation/generate_reference_outputs.py

Expects trained checkpoint at:
    /home/groot/Documents/PKUlab/DPI/DPItorch/save_path_interferometry1/
"""

import os
import sys
import json
import numpy as np
import torch

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)

TRAIN_DIR = "/home/groot/Documents/PKUlab/DPI/DPItorch/save_path_interferometry1"
OUTPUT_DIR = os.path.join(TASK_DIR, "evaluation", "reference_outputs")

MODEL_NAME = "generativemodel_realnvp_res32flow16logdet1.0_closure_fluxcentermemtsv"
SCALE_NAME = "generativescale_realnvp_res32flow16logdet1.0_closure_fluxcentermemtsv"
LOSS_NAME = "loss_realnvp_res32flow16logdet1.0_closure_fluxcentermemtsv.npy"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Copy model weights ──────────────────────────────────────────────────

print("Loading trained model from original code...")
model_path = os.path.join(TRAIN_DIR, MODEL_NAME)
scale_path = os.path.join(TRAIN_DIR, SCALE_NAME)
loss_path = os.path.join(TRAIN_DIR, LOSS_NAME)

for p, name in [(model_path, "model"), (scale_path, "scale"), (loss_path, "loss")]:
    if not os.path.exists(p):
        print(f"ERROR: {name} not found at {p}")
        print("Has the training finished?")
        sys.exit(1)

device = torch.device("cpu")
model_state = torch.load(model_path, map_location=device)
scale_state = torch.load(scale_path, map_location=device)
loss_all = np.load(loss_path, allow_pickle=True).item()

# Save as reference
torch.save(model_state, os.path.join(OUTPUT_DIR, "model_state_dict.pt"))
torch.save(scale_state, os.path.join(OUTPUT_DIR, "logscale_state_dict.pt"))
np.save(os.path.join(OUTPUT_DIR, "loss_history.npy"), loss_all)
print(f"  Saved model weights + loss history")

# ── 2. Load into cleaned code and generate samples ─────────────────────────

from src.solvers import RealNVP, Img_logscale
from src.preprocessing import load_ground_truth

npix = 32
n_flow = 16

model = RealNVP(npix * npix, n_flow, affine=True).to(device)
model.load_state_dict(model_state)
model.eval()

logscale = Img_logscale(scale=1.0).to(device)
logscale.load_state_dict(scale_state)

scale_factor = torch.exp(logscale.forward())

print("Generating 1024 posterior samples...")
n_samples = 1024
batch_size = 128
all_samples = []

with torch.no_grad():
    for start in range(0, n_samples, batch_size):
        bs = min(batch_size, n_samples - start)
        z = torch.randn(bs, npix * npix)
        img_samp, _ = model.reverse(z)
        img_samp = img_samp.reshape(-1, npix, npix)
        img = torch.nn.Softplus()(img_samp) * scale_factor
        all_samples.append(img.cpu().numpy())

samples = np.concatenate(all_samples, axis=0)
posterior_mean = np.mean(samples, axis=0)
posterior_std = np.std(samples, axis=0)

np.save(os.path.join(OUTPUT_DIR, "posterior_samples_1024.npy"), samples)
np.save(os.path.join(OUTPUT_DIR, "posterior_mean.npy"), posterior_mean)
np.save(os.path.join(OUTPUT_DIR, "posterior_std.npy"), posterior_std)
print(f"  Saved 1024 samples, mean, std")

# ── 3. Ground truth ────────────────────────────────────────────────────────

gt = load_ground_truth(os.path.join(TASK_DIR, "data"), npix, 160.0)
np.save(os.path.join(OUTPUT_DIR, "ground_truth.npy"), gt)
print(f"  Saved ground truth")

# ── 4. Compute metrics ─────────────────────────────────────────────────────

from src.visualization import compute_metrics, compute_uq_metrics

metrics = compute_metrics(posterior_mean, gt)
uq_metrics = compute_uq_metrics(posterior_mean, posterior_std, gt)

all_metrics = {**metrics, **uq_metrics}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

print(f"  Metrics: NRMSE={metrics['nrmse']:.4f}, NCC={metrics['ncc']:.4f}")
print(f"  UQ: calibration={uq_metrics['calibration']:.4f}, "
      f"mean_uncertainty={uq_metrics['mean_uncertainty']:.6f}")

# ── 5. Summary ─────────────────────────────────────────────────────────────

print(f"\nAll reference outputs saved to: {OUTPUT_DIR}")
print("Contents:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:<30s} {size / 1024:.1f} KB")
