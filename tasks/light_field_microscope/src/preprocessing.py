"""
Preprocessing — Camera Configuration, Geometry, and Image Alignment
====================================================================

Handles all geometric computations for the light field microscope:
  - Parsing microscope configuration into a Camera parameter dictionary
  - Processing white calibration images to detect micro-lens centers
  - Computing resolution parameters, depth grids, and patch masks
  - Aligning raw light field images to the ideal micro-lens grid

Adapted from pyolaf/geometry.py and pyolaf/transform.py
(Stefanoiu et al., Optics Express 27(22):31644, 2019)
"""

import json
import numpy as np
from scipy import signal
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.signal import convolve2d
from scipy.ndimage import affine_transform, shift
from skimage.feature import peak_local_max
from skimage.morphology import disk


# ═══════════════════════════════════════════════════════════════════════════
# Camera Parameter Setup
# ═══════════════════════════════════════════════════════════════════════════

def load_metadata(path: str = "data/meta_data") -> dict:
    """
    Load task metadata from JSON file.

    Parameters
    ----------
    path : str
        Path to meta_data JSON file.

    Returns
    -------
    dict
        Parsed metadata dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def set_camera_params(metadata: dict, new_spacing_px: int) -> dict:
    """
    Build Camera parameter dictionary from metadata.

    Computes derived optical quantities (focal lengths, wave number, radii,
    conjugate distances) needed by the forward model.

    Parameters
    ----------
    metadata : dict
        Parsed meta_data dictionary with 'microscope' and 'mla' sections.
    new_spacing_px : int
        Desired number of pixels per micro-lens (downsample for speed).

    Returns
    -------
    dict
        Camera dictionary with all optical and geometric parameters.
    """
    micro = metadata["microscope"]
    mla = metadata["mla"]

    Camera = {}

    # Copy base parameters
    Camera["M"] = micro["M"]
    Camera["NA"] = micro["NA"]
    Camera["ftl"] = micro["ftl"]
    Camera["n"] = micro["n"]
    Camera["WaveLength"] = micro["WaveLength"]
    Camera["lensPitch"] = mla["lensPitch"]
    Camera["fm"] = mla["fm"]
    Camera["pixelPitch"] = mla["pixelPitch"]
    Camera["gridType"] = mla["gridType"]
    Camera["focus"] = mla["focus"]
    Camera["uLensMask"] = mla["uLensMask"]
    Camera["plenoptic"] = mla["plenoptic"]

    # For regular grids, exploit quarter-symmetry of the PSF
    if Camera["gridType"] == "reg":
        Camera["range"] = "quarter"
    else:
        Camera["range"] = "full"

    # Derived quantities
    Camera["fobj"] = Camera["ftl"] / Camera["M"]
    Camera["Delta_ot"] = Camera["ftl"] + Camera["fobj"]
    Camera["k"] = 2 * np.pi * Camera["n"] / Camera["WaveLength"]

    # Lenslet spacing in pixels
    spacing_px = Camera["lensPitch"] / Camera["pixelPitch"]
    Camera["spacingPx"] = spacing_px
    Camera["newSpacingPx"] = new_spacing_px
    Camera["newPixelPitch"] = Camera["lensPitch"] / new_spacing_px

    obj_rad = Camera["fobj"] * Camera["NA"]

    if Camera["plenoptic"] == 1:
        # Original LFM: MLA at native image plane
        Camera["tube2mla"] = Camera["ftl"]
        Camera["mla2sensor"] = Camera["fm"]
        tube_rad = obj_rad
        dof = Camera["fobj"]
        m_mla = Camera["M"]
    elif Camera["plenoptic"] == 2:
        # Defocused LFM — compute tube2mla from F-number matching
        Camera["tube2mla"] = mla.get("tube2mla", Camera["ftl"])
        Camera["mla2sensor"] = mla.get("mla2sensor", Camera["fm"])
        obj2tube = Camera["fobj"] + Camera["ftl"]
        dot = Camera["ftl"] * Camera["tube2mla"] / (Camera["tube2mla"] - Camera["ftl"])
        dio = obj2tube - dot
        dof = Camera["fobj"] * dio / (dio - Camera["fobj"])
        if np.isnan(dof):
            dof = Camera["fobj"]
        tube_rad = (dio - obj2tube) * obj_rad / dio
        m_mla = Camera["M"]

    u_rad = tube_rad * Camera["mla2sensor"] / Camera["tube2mla"]
    offset_fobj = dof - Camera["fobj"]

    Camera["objRad"] = obj_rad
    Camera["uRad"] = u_rad
    Camera["tubeRad"] = tube_rad
    Camera["dof"] = dof
    Camera["offsetFobj"] = offset_fobj
    Camera["M_mla"] = m_mla

    return Camera


# ═══════════════════════════════════════════════════════════════════════════
# Micro-lens Grid Detection (from white calibration image)
# ═══════════════════════════════════════════════════════════════════════════

def build_grid(grid_model: dict, grid_type: str) -> np.ndarray:
    """
    Build a 2D array of micro-lens center coordinates from a grid model.

    Parameters
    ----------
    grid_model : dict
        Grid model with HSpacing, VSpacing, HOffset, VOffset, Rot, UMax, VMax.
    grid_type : str
        'reg' for regular or 'hex' for hexagonal grid.

    Returns
    -------
    np.ndarray
        Shape (VMax, UMax, 2) array of (x, y) center coordinates.
    """
    rot_cent = np.eye(3)
    rot_cent[0:2, 2] = [grid_model['HOffset'], grid_model['VOffset']]

    to_offset = np.eye(3)
    to_offset[0:2, 2] = [grid_model['HOffset'], grid_model['VOffset']]

    r = Rotation.from_euler('Z', grid_model['Rot'])
    R = to_offset @ rot_cent @ r.as_matrix() @ np.linalg.inv(rot_cent)

    vv, uu = np.meshgrid(
        -1 + np.arange(0, grid_model['VMax']) * grid_model['VSpacing'],
        -1 + np.arange(0, grid_model['UMax']) * grid_model['HSpacing']
    )

    if grid_type == 'hex':
        uu[grid_model['FirstPosShiftRow']::2, :] += 0.5 * grid_model['HSpacing']

    coords = np.column_stack((uu.ravel(order='C'), vv.ravel(order='C'), np.ones(vv.size)))
    coords = np.dot(R, coords.T).T[:, 0:2]
    coords = coords.reshape([int(grid_model['VMax']), int(grid_model['UMax']), 2], order="F")
    return coords


def set_grid_model(spacing_px, first_pos_shift_row, u_max, v_max,
                   h_offset, v_offset, rot, orientation, grid_type):
    """
    Create a grid model dictionary from explicit parameters.

    Parameters
    ----------
    spacing_px : float
        Spacing in pixels between lenslets.
    first_pos_shift_row : int
        Which row type gets the half-spacing offset (hex grids).
    u_max, v_max : int
        Number of lenslets in horizontal/vertical directions.
    h_offset, v_offset : float
        Offset of the first lenslet.
    rot : float
        Rotation angle in radians.
    orientation : str
        'horz' or 'vert'.
    grid_type : str
        'reg' or 'hex'.

    Returns
    -------
    dict
        Grid model dictionary.
    """
    if grid_type == 'hex':
        spacing = [spacing_px * np.cos(np.deg2rad(30)), spacing_px]
        spacing = np.ceil(spacing).astype(int)
        spacing = (np.ceil(np.array(spacing) / 2) * 2).tolist()
    else:
        spacing = [spacing_px, spacing_px]

    return {
        'HSpacing': spacing[1],
        'VSpacing': spacing[0],
        'HOffset': h_offset,
        'VOffset': v_offset,
        'Rot': rot,
        'UMax': u_max,
        'VMax': v_max,
        'Orientation': orientation,
        'FirstPosShiftRow': first_pos_shift_row,
    }


def process_white_image(white_image: np.ndarray, spacing_px: float,
                        grid_type: str) -> tuple:
    """
    Detect micro-lens centers from a white calibration image.

    Uses disk-convolution filtering, peak detection, KD-tree traversal,
    and line fitting to estimate the regular lenslet grid.

    Parameters
    ----------
    white_image : np.ndarray
        2D white/calibration image from the light field camera.
    spacing_px : float
        Approximate lenslet spacing in pixels.
    grid_type : str
        'reg' or 'hex'.

    Returns
    -------
    tuple
        (LensletGridModel dict, GridCoords ndarray)
    """
    crop_amt = 15
    skip_step = 10
    filter_disk_radius_mult = 1 / 3

    # Convolve with disk to enhance lenslet centers
    hr = disk(int(spacing_px * filter_disk_radius_mult))
    hr = hr / hr.max()
    conv = signal.fftconvolve(white_image, hr, mode='same')
    conv = (conv - conv.min()) / np.abs(conv.max() - conv.min())

    # Peak detection
    peaks = peak_local_max(conv, exclude_border=crop_amt)
    peak_y, peak_x = peaks.T
    peak_ref = np.column_stack([peak_x, peak_y])
    tree = cKDTree(peak_ref)

    # Traverse horizontally to estimate spacing and angle
    x_start = crop_amt * 2 + 1
    x_stop = white_image.shape[1] - crop_amt * 2 - 1
    rec_pts_x = []
    line_fit_x = []
    for y_start in range(crop_amt * 2, white_image.shape[0] - crop_amt * 2, skip_step):
        cur_pos = np.array([x_start, y_start])
        pts = []
        while True:
            _, closest_label = tree.query(cur_pos)
            closest_pt = peak_ref[closest_label]
            pts.append(closest_pt)
            cur_pos = np.copy(closest_pt)
            cur_pos[0] = round(cur_pos[0] + spacing_px)
            if cur_pos[0] > x_stop:
                break
        pts = np.array(pts)
        if len(pts) > 10:
            line_fit_x.append(np.polyfit(pts[3:-3, 0], pts[3:-3, 1], 1))
            rec_pts_x.append(pts[3:-3])

    # Traverse vertically
    y_start_v = crop_amt * 2 + 1
    y_stop = white_image.shape[0] - crop_amt * 2 - 1
    rec_pts_y = []
    line_fit_y = []
    for x_s in range(crop_amt * 2, white_image.shape[1] - crop_amt * 2, skip_step):
        cur_pos = [x_s, y_start_v]
        pts = []
        while True:
            _, closest_label = tree.query(cur_pos)
            closest_pt = peak_ref[closest_label]
            pts.append(closest_pt)
            cur_pos = np.copy(closest_pt)
            cur_pos[1] = round(cur_pos[1] + spacing_px * np.sqrt(3))
            if cur_pos[1] > y_stop:
                break
        pts = np.array(pts)
        if len(pts) > 10:
            line_fit_y.append(np.polyfit(pts[3:-3, 1], pts[3:-3, 0], 1))
            rec_pts_y.append(pts[3:-3])

    rec_pts_x = rec_pts_x[3:-3]
    rec_pts_y = rec_pts_y[3:-3]

    # Estimate rotation angle
    possible = []
    if len(line_fit_x) > 0:
        slope_x = np.mean(line_fit_x, axis=0)[0]
        possible.append(np.arctan2(-slope_x, 1))
    if len(line_fit_y) > 0:
        slope_y = np.mean(line_fit_y, axis=0)[0]
        possible.append(np.arctan2(slope_y, 1))
    est_angle = np.mean(possible) if possible else 0.0

    # Estimate spacing
    y_spacing = np.mean([np.mean(np.diff(row[:, 1])) for row in rec_pts_y]) / 2 / (np.sqrt(3) / 2)
    x_spacing = np.mean([np.mean(np.diff(row[:, 0])) for row in rec_pts_x])
    x_spacing /= np.cos(est_angle)
    y_spacing /= np.cos(est_angle)

    # Build initial grid model
    if grid_type == 'reg':
        grid_model = {
            'HSpacing': x_spacing, 'VSpacing': x_spacing,
            'HOffset': crop_amt, 'VOffset': crop_amt,
            'Rot': -est_angle, 'Orientation': 'horz', 'FirstPosShiftRow': 2,
            'UMax': int(np.ceil((white_image.shape[1] - crop_amt * 2) / x_spacing)),
            'VMax': int(np.ceil((white_image.shape[0] - crop_amt * 2) / x_spacing)),
        }
    else:
        grid_model = {
            'HSpacing': x_spacing, 'VSpacing': y_spacing * np.sqrt(3) / 2,
            'HOffset': crop_amt, 'VOffset': crop_amt,
            'Rot': -est_angle, 'Orientation': 'horz', 'FirstPosShiftRow': 2,
            'UMax': int(np.ceil((white_image.shape[1] - crop_amt * 2) / x_spacing)),
            'VMax': int(np.ceil((white_image.shape[0] - crop_amt * 2) / (y_spacing * np.sqrt(3) / 2))),
        }

    # Refine offset by matching grid to detected peaks
    grid_coords = build_grid(grid_model, grid_type)
    build_coords = np.column_stack((grid_coords[..., 0].ravel(), grid_coords[..., 1].ravel()))
    _, ix = tree.query(build_coords)
    ideal_pts = peak_ref[ix]
    est_offset = np.median(ideal_pts - build_coords, axis=0)
    grid_model['HOffset'] += est_offset[0]
    grid_model['VOffset'] += est_offset[1]

    # Remove crop offset
    new_v_offset = grid_model["VOffset"] % grid_model["VSpacing"]
    v_steps = round((grid_model["VOffset"] - new_v_offset) / grid_model["VSpacing"])
    if grid_type == 'reg':
        new_h_offset = grid_model["HOffset"] % grid_model["HSpacing"]
    else:
        v_step_parity = v_steps % 2
        if v_step_parity == 1:
            grid_model["HOffset"] += grid_model["HSpacing"] / 2
        new_h_offset = grid_model["HOffset"] % (grid_model["HSpacing"] / 2)
        h_steps = round((grid_model["HOffset"] - new_h_offset) / (grid_model["HSpacing"] / 2))
        grid_model["FirstPosShiftRow"] = 2 - (h_steps % 2)

    grid_model["HOffset"] = new_h_offset
    grid_model["VOffset"] = new_v_offset
    grid_model["UMax"] = np.floor((white_image.shape[1] - grid_model["HOffset"]) / grid_model["HSpacing"]) + 1
    grid_model["VMax"] = np.floor((white_image.shape[0] - grid_model["VOffset"]) / grid_model["VSpacing"]) + 1

    grid_coords = build_grid(grid_model, grid_type)
    return grid_model, grid_coords


# ═══════════════════════════════════════════════════════════════════════════
# Patch Mask and Resolution Computation
# ═══════════════════════════════════════════════════════════════════════════

def fix_mask(mask: np.ndarray, new_lenslet_spacing: np.ndarray,
             grid_type: str) -> np.ndarray:
    """
    Correct a micro-lens patch mask for holes and overlaps.

    When tiling the mask at lenslet centers, every sensor pixel should be
    covered exactly once. This function iteratively fixes violations.

    Parameters
    ----------
    mask : np.ndarray
        Initial binary patch mask.
    new_lenslet_spacing : np.ndarray
        (row_spacing, col_spacing) in pixels.
    grid_type : str
        'reg' or 'hex'.

    Returns
    -------
    np.ndarray
        Corrected binary mask.
    """
    trial_space = np.zeros((3 * mask.shape[0], 3 * mask.shape[1]))
    r_center = int(np.ceil(trial_space.shape[0] / 2))
    c_center = int(np.ceil(trial_space.shape[1] / 2))
    rs = int(new_lenslet_spacing[0])
    cs = int(new_lenslet_spacing[1])

    if grid_type == 'hex':
        for a in [r_center - rs, r_center + rs]:
            for b in [c_center - round(cs / 2), c_center + round(cs / 2)]:
                trial_space[a - 1, b - 1] = 1
        for b in [c_center - cs, c_center, c_center + cs]:
            trial_space[r_center - 1, b - 1] = 1
    else:
        for a in [r_center - rs, r_center, r_center + rs]:
            for b in [c_center - cs, c_center, c_center + cs]:
                trial_space[a - 1, b - 1] = 1

    r, c = mask.shape[:2]
    space = convolve2d(trial_space, mask, mode='same')
    space_center = space[r:r * 2, c:c * 2]
    new_mask = mask.copy()

    for i in range(r):
        for j in range(c):
            if space_center[i, j] == 0:
                new_mask[i, j] = 1
                space = convolve2d(trial_space, new_mask, mode='same')
                space_center = space[r:r * 2, c:c * 2]
            elif space_center[i, j] == 2:
                new_mask[i, j] = 0
                space = convolve2d(trial_space, new_mask, mode='same')
                space_center = space[r:r * 2, c:c * 2]
    return new_mask


def compute_patch_mask(n_spacing: np.ndarray, grid_type: str,
                       pixel_size: np.ndarray, patch_rad: float,
                       nnum: np.ndarray) -> np.ndarray:
    """
    Compute the binary mask for a single micro-lens patch.

    Parameters
    ----------
    n_spacing : np.ndarray
        Lenslet spacing in pixels [V, H].
    grid_type : str
        'reg' or 'hex'.
    pixel_size : np.ndarray
        Pixel size [y, x] in physical units.
    patch_rad : float
        Radius of the micro-lens patch in physical units.
    nnum : np.ndarray
        Number of pixels per patch [V, H].

    Returns
    -------
    np.ndarray
        Binary mask array.
    """
    y_space = np.arange(-np.floor(nnum[0] / 2), np.floor(nnum[0] / 2) + 1)
    x_space = np.arange(-np.floor(nnum[1] / 2), np.floor(nnum[1] / 2) + 1)
    x, y = np.meshgrid(pixel_size[0] * y_space, pixel_size[1] * x_space)
    mask = (np.sqrt(x * x + y * y) < patch_rad).astype(int)
    mask = fix_mask(mask, n_spacing, grid_type)
    return mask


def compute_resolution(lenslet_grid_model: dict, texture_grid_model: dict,
                       Camera: dict, depth_range: list, depth_step: float) -> dict:
    """
    Compute resolution-related parameters for the reconstruction.

    Parameters
    ----------
    lenslet_grid_model : dict
        New (target) lenslet grid model.
    texture_grid_model : dict
        Texture-space grid model.
    Camera : dict
        Camera parameter dictionary.
    depth_range : list
        [min_depth, max_depth] in um.
    depth_step : float
        Step between depth planes in um.

    Returns
    -------
    dict
        Resolution dictionary with Nnum, texRes, depths, masks, etc.
    """
    n_spacing_lenslet = np.array([lenslet_grid_model['VSpacing'],
                                  lenslet_grid_model['HSpacing']])
    n_spacing_texture = np.array([texture_grid_model['VSpacing'],
                                  texture_grid_model['HSpacing']])

    if Camera['gridType'] == 'reg':
        sensor_res = np.array([Camera['lensPitch'] / n_spacing_lenslet[0],
                               Camera['lensPitch'] / n_spacing_lenslet[1]])
        nnum = n_spacing_lenslet.copy()
        tex_nnum = n_spacing_texture.copy()
    else:
        sensor_res = np.array([Camera['lensPitch'] * np.cos(np.deg2rad(30)) / n_spacing_lenslet[0],
                               Camera['lensPitch'] / n_spacing_lenslet[1]])
        nnum = np.array([np.max(n_spacing_lenslet) + 1, np.max(n_spacing_lenslet) + 1])
        tex_nnum = np.array([np.max(n_spacing_texture) + 1, np.max(n_spacing_texture) + 1])

    nnum = nnum + (1 - np.mod(nnum, 2))
    tex_nnum = tex_nnum + (1 - np.mod(tex_nnum, 2))
    tex_scale_factor = tex_nnum / nnum
    tex_res = np.append(sensor_res / (tex_scale_factor * Camera['M']), depth_step)

    sens_mask = compute_patch_mask(n_spacing_lenslet, Camera['gridType'],
                                   sensor_res, Camera['uRad'], nnum)
    tex_mask = compute_patch_mask(n_spacing_texture, Camera['gridType'], tex_res,
                                  tex_nnum[0] * tex_res[0] / 2, tex_nnum)

    Resolution = {
        'Nnum': nnum,
        'Nnum_half': np.ceil(nnum / 2).astype(int),
        'TexNnum': tex_nnum,
        'TexNnum_half': np.ceil(tex_nnum / 2).astype(int),
        'sensorRes': sensor_res,
        'texRes': tex_res,
        'sensMask': sens_mask,
        'texMask': tex_mask,
        'depthStep': depth_step,
        'depthRange': depth_range,
        'depths': np.arange(depth_range[0], depth_range[1] + depth_step, depth_step),
        'texScaleFactor': tex_scale_factor,
        'maskFlag': Camera['uLensMask'],
        'NspacingLenslet': n_spacing_lenslet,
        'NspacingTexture': n_spacing_texture,
    }
    return Resolution


def compute_lens_centers(new_grid_model: dict, texture_grid_model: dict,
                         sensor_res: np.ndarray, grid_type: str) -> dict:
    """
    Compute micro-lens centers in pixel, metric, and voxel coordinates.

    Parameters
    ----------
    new_grid_model : dict
        Target lenslet grid model.
    texture_grid_model : dict
        Texture-space grid model.
    sensor_res : np.ndarray
        Sensor resolution [y, x] in um/pixel.
    grid_type : str
        'reg' or 'hex'.

    Returns
    -------
    dict
        LensletCenters with 'px', 'metric', 'vox', and 'offset' keys.
    """
    centers_pixels = build_grid(new_grid_model, grid_type)
    center_of_sensor = np.round(np.array(centers_pixels.shape[:2]) / 2.0 + 0.01).astype(int) - 1

    lc = {}
    lc['px'] = np.copy(centers_pixels)
    center_offset = [centers_pixels[center_of_sensor[0], center_of_sensor[1], 1],
                     centers_pixels[center_of_sensor[0], center_of_sensor[1], 0]]
    lc['offset'] = np.array(center_offset) + 1

    lc['px'][:, :, 0] = centers_pixels[:, :, 1] - center_offset[0]
    lc['px'][:, :, 1] = centers_pixels[:, :, 0] - center_offset[1]

    lc['metric'] = np.copy(centers_pixels)
    lc['metric'][:, :, 0] = lc['px'][:, :, 0] * sensor_res[0]
    lc['metric'][:, :, 1] = lc['px'][:, :, 1] * sensor_res[1]

    centers_voxels = build_grid(texture_grid_model, grid_type)
    center_of_texture = np.round(np.array(centers_voxels.shape[:2]) / 2 + 0.01).astype(int) - 1
    center_offset_v = [centers_voxels[center_of_texture[0], center_of_texture[1], 1],
                       centers_voxels[center_of_texture[0], center_of_texture[1], 0]]

    lc['vox'] = np.zeros_like(centers_voxels)
    lc['vox'][:, :, 0] = centers_voxels[:, :, 1] - center_offset_v[0]
    lc['vox'][:, :, 1] = centers_voxels[:, :, 0] - center_offset_v[1]

    return lc


# ═══════════════════════════════════════════════════════════════════════════
# Top-level Geometry Computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_geometry(Camera: dict, white_image: np.ndarray,
                     depth_range: list, depth_step: float,
                     super_res_factor: int,
                     img_size: np.ndarray = None) -> tuple:
    """
    Compute all geometry parameters from camera config and calibration image.

    If white_image is empty (for synthetic data), builds grid from specs.

    Parameters
    ----------
    Camera : dict
        Camera parameter dictionary.
    white_image : np.ndarray
        White calibration image (empty array for synthetic mode).
    depth_range : list
        [min_depth, max_depth] in um.
    depth_step : float
        Step between depth planes in um.
    super_res_factor : int
        Super-resolution factor (voxels per lenslet).
    img_size : np.ndarray, optional
        Image size for synthetic mode (when white_image is empty).

    Returns
    -------
    tuple
        (LensletCenters, Resolution, LensletGridModel, NewLensletGridModel)
    """
    if white_image.size == 0:
        # Synthetic mode: build grid from specs
        mla_size = np.ceil(np.array(img_size) / Camera['newSpacingPx'])
        lenslet_grid_model = {
            'UMax': int(mla_size[1]),
            'VMax': int(mla_size[0]),
            'VSpacing': np.round(Camera['spacingPx']),
            'HSpacing': np.round(Camera['spacingPx']),
            'FirstPosShiftRow': 1,
            'Orientation': 'horz',
            'HOffset': 0,
            'VOffset': 0,
            'Rot': 0,
        }
    else:
        lenslet_grid_model, _ = process_white_image(
            white_image, Camera['spacingPx'], Camera['gridType'])

    # Create target grid model with desired spacing
    new_grid_model = set_grid_model(
        Camera['newSpacingPx'], lenslet_grid_model['FirstPosShiftRow'],
        lenslet_grid_model['UMax'], lenslet_grid_model['VMax'],
        0, 0, 0, lenslet_grid_model['Orientation'], Camera['gridType'])

    input_spacing = np.array([lenslet_grid_model['HSpacing'], lenslet_grid_model['VSpacing']])
    new_spacing = np.array([new_grid_model['HSpacing'], new_grid_model['VSpacing']])
    xform_scale = new_spacing / input_spacing
    new_offset = np.round(np.array([lenslet_grid_model['HOffset'],
                                     lenslet_grid_model['VOffset']]) * xform_scale)
    new_grid_model['HOffset'] = new_offset[0]
    new_grid_model['VOffset'] = new_offset[1]

    # Texture grid model
    texture_grid_model = set_grid_model(
        super_res_factor, lenslet_grid_model['FirstPosShiftRow'],
        lenslet_grid_model['UMax'], lenslet_grid_model['VMax'],
        0, 0, 0, lenslet_grid_model['Orientation'], Camera['gridType'])

    # Resolution
    Resolution = compute_resolution(new_grid_model, texture_grid_model,
                                    Camera, depth_range, depth_step)
    Resolution['superResFactor'] = super_res_factor

    print(f"  Super-resolution factor: {Resolution['TexNnum']}")
    print(f"  Pixel size: [{Resolution['sensorRes'][0]:.2f}, {Resolution['sensorRes'][1]:.2f}] um")
    print(f"  Voxel size: [{Resolution['texRes'][0]:.2f}, {Resolution['texRes'][1]:.2f}, {Resolution['texRes'][2]:.1f}] um")

    # Lens centers
    new_grid_model['FirstPosShiftRow'] = lenslet_grid_model['FirstPosShiftRow']
    texture_grid_model['FirstPosShiftRow'] = new_grid_model['FirstPosShiftRow']
    lenslet_centers = compute_lens_centers(
        new_grid_model, texture_grid_model,
        Resolution['sensorRes'], Camera['gridType'])

    return lenslet_centers, Resolution, lenslet_grid_model, new_grid_model


# ═══════════════════════════════════════════════════════════════════════════
# Image Alignment (Affine Transform)
# ═══════════════════════════════════════════════════════════════════════════

def retrieve_transformation(lenslet_grid_model: dict,
                            new_grid_model: dict) -> np.ndarray:
    """
    Compute the affine transformation between original and target grids.

    Parameters
    ----------
    lenslet_grid_model : dict
        Original (detected) grid model.
    new_grid_model : dict
        Target (ideal) grid model.

    Returns
    -------
    np.ndarray
        3x3 affine transformation matrix.
    """
    input_spacing = np.array([lenslet_grid_model['HSpacing'],
                              lenslet_grid_model['VSpacing']])
    new_spacing = np.array([new_grid_model['HSpacing'],
                            new_grid_model['VSpacing']])
    xform_scale = new_spacing / input_spacing

    r_scale = np.eye(3)
    r_scale[0, 0] = xform_scale[0]
    r_scale[1, 1] = xform_scale[1]

    new_offset = np.array([lenslet_grid_model['HOffset'],
                           lenslet_grid_model['VOffset']]) * xform_scale
    rounded_offset = np.round(new_offset).astype(int)
    xform_trans = rounded_offset - new_offset

    r_trans = np.eye(3)
    r_trans[-1, :2] = xform_trans

    r_rot = Rotation.from_euler('ZYX', [lenslet_grid_model['Rot'], 0, 0]).as_matrix()

    return r_rot @ r_scale @ r_trans


def format_transform(fix_all: np.ndarray) -> np.ndarray:
    """Convert transformation matrix to the format expected by affine_transform."""
    transform = np.linalg.inv(fix_all).T
    ttnew = np.zeros_like(transform)
    ttnew[:2, :2] = transform[:2, :2].T
    ttnew[:, 2] = transform[[1, 0, 2], 2]
    return ttnew


def get_transformed_shape(img_shape: tuple, ttnew: np.ndarray) -> np.ndarray:
    """Compute the output shape after applying the affine transform."""
    scale = np.diag(ttnew)[:2]
    new_shape = np.floor(np.round(np.array(img_shape) / scale) / 2) * 2 + 1
    return new_shape.astype('int32')


def transform_image(img: np.ndarray, ttnew: np.ndarray,
                    lens_offset: np.ndarray) -> np.ndarray:
    """
    Apply affine transformation to align raw LF image to the target grid.

    Parameters
    ----------
    img : np.ndarray
        Raw light field image.
    ttnew : np.ndarray
        3x3 formatted transformation matrix.
    lens_offset : np.ndarray
        Offset of the central lenslet.

    Returns
    -------
    np.ndarray
        Aligned light field image.
    """
    scale = np.diag(ttnew)[:2]
    new_shape = np.floor(np.round(np.array(img.shape) / scale) / 2) * 2 + 1
    new_shape = new_shape.astype('int32')
    offset = np.ceil(new_shape / 2) - lens_offset

    new = affine_transform(img, ttnew[:2, :2], offset=ttnew[:2, 2],
                           output_shape=tuple(new_shape), order=1, prefilter=False)
    new = shift(new, [offset[0], offset[1]], order=1, prefilter=False)
    new[new < np.mean(new)] = np.mean(new)
    return new


def prepare_data(metadata_path: str = "data/meta_data") -> tuple:
    """
    Load metadata and prepare Camera parameters.

    Parameters
    ----------
    metadata_path : str
        Path to the meta_data JSON file.

    Returns
    -------
    tuple
        (Camera dict, metadata dict)
    """
    metadata = load_metadata(metadata_path)
    recon = metadata["reconstruction"]
    Camera = set_camera_params(metadata, recon["newSpacingPx"])
    return Camera, metadata
