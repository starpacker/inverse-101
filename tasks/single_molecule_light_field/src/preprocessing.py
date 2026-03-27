"""Preprocessing for Single-Molecule Light Field Microscopy (SMLFM).

Converts raw PeakFit CSV output to centred 2D localisation arrays
ready for microlens assignment and 3D fitting.

Extracted from: hexSMLFM / PySMLFM (TheLeeLab / Photometrics, Cambridge).
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt


def load_localizations(csv_file: Path, meta: dict) -> npt.NDArray[float]:
    """Load 2D localizations from a PeakFit/Thunderstorm/Picasso CSV file.

    Reads the CSV, selects the appropriate columns for the given format,
    and converts coordinates to physical microns.

    Supported formats and their column layouts:

    PEAKFIT (ImageJ GDSC SMLM2):
        col 0: frame, col 7: background (ph), col 8: intensity (ph),
        col 9: X (pixels), col 10: Y (pixels), col 12: sigma (pixels),
        col 13: precision (nm).  X/Y/sigma are in pixels — multiply by
        pixel_size_sample to get microns.

    THUNDERSTORM:
        col 0: frame, col 1: X (nm), col 2: Y (nm), col 3: sigma (nm),
        col 4: intensity (ph), col 5: background (ph), col 6: precision (nm).

    PICASSO:
        col 1: frame, col 2: X (nm), col 3: Y (nm), col 4: sigma (nm),
        col 5: intensity (ph), col 6: background (ph), col 8: precision (nm).

    Args:
        csv_file: Path to the 2D localisation CSV.
        meta: Dictionary loaded from meta_data.json.  Uses keys:
            csv_format (str), num_aperture, mla_lens_pitch, focal_length_mla,
            focal_length_obj_lens, focal_length_tube_lens,
            focal_length_fourier_lens, pixel_size_camera.

    Returns:
        locs_2d_csv (N, 8) float array with columns:
            [0] frame
            [1] X (microns)
            [2] Y (microns)
            [3] sigma_X (microns)
            [4] sigma_Y (microns)
            [5] intensity (photons)
            [6] background (photons)
            [7] precision (microns)
    """
    fmt = meta["csv_format"].upper()

    if fmt == "PEAKFIT":
        id_frame, id_x, id_y = 0, 9, 10
        id_sigma_x = id_sigma_y = 12
        id_intensity, id_background, id_precision = 8, 7, 13
        min_columns = 14
        scale_xy = 1.0      # pixels; will be multiplied by pixel_size_sample below
        scale_pr = 1000.0   # nm → µm
    elif fmt == "THUNDERSTORM":
        id_frame, id_x, id_y = 0, 1, 2
        id_sigma_x = id_sigma_y = 3
        id_intensity, id_background, id_precision = 4, 5, 6
        min_columns = 7
        scale_xy = 1000.0   # nm → µm
        scale_pr = 1000.0
    elif fmt == "PICASSO":
        id_frame, id_x, id_y = 1, 2, 3
        id_sigma_x = id_sigma_y = 4
        id_intensity, id_background, id_precision = 5, 6, 8
        min_columns = 9
        scale_xy = 1000.0   # nm → µm
        scale_pr = 1000.0
    else:
        raise ValueError(f"Unsupported csv_format: {fmt!r}. "
                         "Choose PEAKFIT, THUNDERSTORM, or PICASSO.")

    # Auto-detect header rows (lines starting with non-digit) and delimiter
    csv_header_rows = 0
    csv_delimiter = None
    with open(csv_file, "r", encoding="utf-8") as f:
        for line in f:
            if line[0].isdigit():
                if len(line.split(",")) >= min_columns:
                    csv_delimiter = ","
                break
            csv_header_rows += 1

    raw = np.genfromtxt(
        csv_file, delimiter=csv_delimiter, dtype=float,
        skip_header=csv_header_rows,
    )
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]

    data = np.empty((raw.shape[0], 8))
    data[:, 0] = raw[:, id_frame]
    data[:, 1] = raw[:, id_x] / scale_xy
    data[:, 2] = raw[:, id_y] / scale_xy
    data[:, 3] = raw[:, id_sigma_x] / scale_xy
    data[:, 4] = raw[:, id_sigma_y] / scale_xy
    data[:, 5] = raw[:, id_intensity]
    data[:, 6] = raw[:, id_background]
    data[:, 7] = raw[:, id_precision] / scale_pr

    # For PEAKFIT, X/Y/sigma are in pixels; convert to sample-plane microns.
    if fmt == "PEAKFIT":
        pixel_size_sample = _pixel_size_sample(meta)
        data[:, 1:5] *= pixel_size_sample

    return data


def center_localizations(locs_2d_csv: npt.NDArray[float]) -> npt.NDArray[float]:
    """Subtract the mean X and Y so the field of view is centred at the origin.

    This places the optical axis at (0, 0), which is required for microlens
    assignment (the MLA lattice is also defined relative to its own centre).

    Args:
        locs_2d_csv: (N, 8) array from load_localizations.

    Returns:
        Copy of input with columns 1 and 2 (X, Y) mean-subtracted.
    """
    out = locs_2d_csv.copy()
    out[:, 1] -= out[:, 1].mean()
    out[:, 2] -= out[:, 2].mean()
    return out


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _pixel_size_sample(meta: dict) -> float:
    """Return pixel size in sample space (microns).

    pixel_size_sample = pixel_size_camera / magnification
    magnification     = (f_tube / f_obj) * (f_mla / f_fourier)
    """
    mag = (meta["focal_length_tube_lens"] / meta["focal_length_obj_lens"]
           * meta["focal_length_mla"] / meta["focal_length_fourier_lens"])
    return meta["pixel_size_camera"] / mag
