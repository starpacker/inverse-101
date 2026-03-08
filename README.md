# imaging-101: Scientific Imaging Benchmark

A benchmark suite for **computational and scientific imaging** inverse problems.
Each task provides physics-based forward models, curated data, and reference solvers
ranging from classical algorithms to deep learning methods.

---

## Tasks

| Task | Domain | Measurement | Methods |
|------|--------|-------------|---------|
| [EHT Black Hole Imaging](tasks/eht_black_hole/) | Radio astronomy | Sparse Fourier (VLBI) | CLEAN, RML-TV, RML-MEM |
| *More coming soon* | | | |

---

## Design Philosophy

Each task is structured around a common template:

```
tasks/<task_name>/
├── README.md           # Physics background, problem formulation, references
├── requirements.txt    # Dependencies
├── src/
│   ├── forward_model.py    # Physics-based measurement operator
│   ├── solvers.py          # Inverse problem solvers
│   └── visualization.py    # Plotting utilities
├── generate_data.py    # Synthetic data generation
└── notebooks/
    └── <task_name>.ipynb   # End-to-end tutorial
```

**Principles:**
- Forward models grounded in physics, not black boxes
- Solvers span the spectrum: analytical → iterative → regularized → learned
- All tasks include synthetic data generation (no external downloads required)
- Code written for clarity and education, not just performance

---

## Getting Started

```bash
git clone https://github.com/HeSunPU/imaging-101.git
cd imaging-101

# Install dependencies for a specific task
pip install -r tasks/eht_black_hole/requirements.txt

# Run the tutorial notebook
jupyter notebook tasks/eht_black_hole/notebooks/eht_black_hole.ipynb
```

---

## Contributing

To add a new imaging task, follow the task template and submit a pull request.
Tasks of interest: X-ray CT, MRI, phase retrieval, lensless imaging,
seismic imaging, optical coherence tomography, and more.
