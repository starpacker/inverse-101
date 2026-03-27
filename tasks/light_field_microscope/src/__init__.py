"""
Light Field Microscope — Artifact-free 3D Deconvolution
========================================================

Modules
-------
preprocessing : Camera setup, geometry, lenslet center detection, image alignment
physics_model : Wave-optics LFPSF, MLA transmittance, RS propagation, projection operators
solvers       : Depth-adaptive anti-aliasing filters, EMS deconvolution
visualization : Volume rendering, metrics, comparison plots
generate_data : Synthetic LFM data generation
"""

from . import preprocessing
from . import physics_model
from . import solvers
from . import visualization
from . import generate_data
