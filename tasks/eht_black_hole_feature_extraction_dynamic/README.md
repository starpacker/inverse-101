# Dynamic EHT Black Hole Feature Extraction (α-DPI)

## Overview

Per-frame geometric parameter inference from time-varying EHT closure quantities using α-DPI (α-divergence variational inference + importance sampling). This task applies the α-DPI method independently to each time snapshot of a dynamic black hole observation, producing posterior distributions of geometric crescent parameters at each epoch.

The output is a **ridge plot** showing how parameter posteriors evolve over time, similar to Figure 13 of the EHT Sgr A* Paper IV (2022, ApJL 930, L15).

## Physical Background

Sagittarius A* (Sgr A*) exhibits rapid structural variability — the bright emission region orbits the black hole on timescales of minutes to hours. Unlike M87*, whose morphology is essentially static during a single observation, Sgr A* requires time-resolved analysis.

This task simulates this scenario with a **rotating simple crescent**: the position angle of the bright side rotates linearly over the observation window while diameter, width, and asymmetry remain constant. The EHT 2017 array (8 stations: ALMA, APEX, JCMT, SMA, SMT, LMT, PV, SPT) observes each time snapshot independently.

## Method

1. **Data generation**: A time-varying simple crescent is observed by the EHT 2017 array. The 8-hour observation is split into 10 frames (~48 min each), each saved as a separate UVFITS file.

2. **Per-frame α-DPI inference**: For each frame independently:
   - Extract closure phases and log closure amplitudes (gain-invariant observables)
   - Train a Real-NVP normalizing flow to minimize α-divergence between the approximate posterior and the true posterior over 4 geometric parameters
   - Importance sampling reweighting for accurate posterior calibration

3. **Visualization**: Ridge plots showing posterior parameter distributions over time, with ground truth overlaid.

## Parameters

The SimpleCrescent geometric model has 4 parameters:

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Diameter | d | [20, 80] μas | Outer ring diameter |
| Width | w | [1, 40] μas | Ring thickness (Gaussian σ) |
| Asymmetry | A | [0, 1] | Brightness contrast |
| Position angle | φ | [-181, 181] deg | Bright side orientation (E of N) |

## Data

- `data/obs_frame_XX.uvfits` — per-frame EHT observations (10 files)
- `data/gt.fits` — ground truth image (first frame)
- `data/meta_data` — configuration and per-frame ground truth parameters

## Usage

```bash
cd tasks/eht_black_hole_feature_extraction_dynamic
pip install -r requirements.txt

# Generate synthetic data (optional, already included)
python -c "from src.generate_data import generate_dynamic_dataset; generate_dynamic_dataset()"

# Run the full pipeline
python main.py

# Run tests
python -m pytest evaluation/tests/ -v
```

## Output

- `output/ridge_plot.png` — Ridge plot of posterior distributions over time
- `output/param_evolution.png` — Parameter mean ± 1σ evolution
- `output/frame_images.png` — Posterior mean images vs ground truth
- `output/all_params.npy` — Posterior samples per frame (10, N, 4)
- `output/all_weights.npy` — Importance weights per frame (10, N)
- `output/metrics.json` — Per-frame recovery metrics

## References

- Sun et al. (2022), ApJ 932:99 — α-DPI method
- EHT Collaboration (2022), ApJL 930:L15 — Sgr A* Paper IV (Variability, Figure 13)
- EHT Collaboration (2022), ApJL 930:L12 — Sgr A* Paper I
