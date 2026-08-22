import os
import shutil
import tempfile

import numpy as np
import nibabel as nib

from sparc.utils.nii import (
    get_nii_range,
    N4_corr,
    remove_zero_filling,
)
from sparc.tools.mrtrix import (
    mrconvert,
    mrcalc,
    dwidenoise,
    mrdegibbs,
)


@staticmethod
def dcm2nii(
    dcm_dir,
    nii_path,
    meta_path=None,
    to_rad=False,
):
    """Convert a DICOM folder to a NIfTI file."""

    nii_dir = os.path.dirname(nii_path)
    os.makedirs(nii_dir, exist_ok=True)

    if meta_path:
        meta_dir = os.path.dirname(meta_path)
        os.makedirs(meta_dir, exist_ok=True)

    mrconvert(
        dcm_dir=dcm_dir,
        nii_path=nii_path,
        json_path=meta_path,
    )

    # Convert to radians (phase data)
    if to_rad:
        # Get range
        nii_range = get_nii_range(nii_path)
        scale = f"{2*np.pi/nii_range:.4f}"

        mrcalc(
            operand1=nii_path,
            operand2=scale,
            operator="multiply",
            output_path=nii_path,
        )


@staticmethod
def remove_zero_filling_funct(
    mag_nii_path,
    pha_nii_path=None,
    acq_mat=None,
    debug=False,
):
    """Remove k-space zero-filling interpolation from a reconstructed NIfTI volume."""
    remove_zero_filling(
        mag_nii_path=mag_nii_path,
        pha_nii_path=pha_nii_path,
        acq_mat=acq_mat,
        debug=debug,
    )


@staticmethod
def denoise_funct(
    mag_nii_path,
    pha_nii_path=None,
    debug=False,
):
    """Denoise a NIfTI volume (in place) using MRtrix's 'dwidenoise'."""

    mag_nii = nib.load(mag_nii_path)
    dim = mag_nii.header["dim"][1:5].tolist()

    k = int(np.ceil(np.sqrt(dim[3])))
    extent = (k, k, 1)
    if debug:
        debug_mag_nii_path = mag_nii_path.replace(".nii.gz", "_pre_denoise.nii.gz")
        shutil.copy(mag_nii_path, debug_mag_nii_path)
        if pha_nii_path is not None:
            debug_pha_nii_path = pha_nii_path.replace(".nii.gz", "_pre_denoise.nii.gz")
            shutil.copy(pha_nii_path, debug_pha_nii_path)

    if pha_nii_path is None:
        dwidenoise(input_nii_path=mag_nii_path, output_nii_path=mag_nii_path, extent=extent)
    else:
        temp_dir = tempfile.mkdtemp()
        cmpl_path = os.path.join(temp_dir, "cmpl.nii.gz")

        # To complex
        mrcalc(
            operand1=mag_nii_path,
            operand2=pha_nii_path,
            operator="polar",
            output_path=cmpl_path,
        )

        # Denoise
        dwidenoise(input_nii_path=cmpl_path, output_nii_path=cmpl_path, extent=extent)

        # To magnitude and phase
        mrcalc(
            operand1=cmpl_path,
            operand2=None,
            operator="abs",
            output_path=mag_nii_path,
        )
        mrcalc(
            operand1=cmpl_path,
            operand2=None,
            operator="phase",
            output_path=pha_nii_path,
        )

        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


@staticmethod
def degibbs_funct(
    mag_nii_path,
    pha_nii_path=None,
    axes=(0, 1),
    debug=False,
):
    """Apply Gibbs-ringing correction using MRtrix's 'mrdegibbs'."""

    if debug:
        debug_mag_nii_path = mag_nii_path.replace(".nii.gz", "_pre_degibbs.nii.gz")
        shutil.copy(mag_nii_path, debug_mag_nii_path)
        if pha_nii_path is not None:
            debug_pha_nii_path = pha_nii_path.replace(".nii.gz", "_pre_degibbs.nii.gz")
            shutil.copy(pha_nii_path, debug_pha_nii_path)

    if pha_nii_path is None:
        mrdegibbs(input_nii_path=mag_nii_path, output_nii_path=mag_nii_path, axes=axes)
    else:
        temp_dir = tempfile.mkdtemp()
        cmpl_path = os.path.join(temp_dir, "cmpl.nii.gz")

        # To complex
        mrcalc(
            operand1=mag_nii_path,
            operand2=pha_nii_path,
            operator="polar",
            output_path=cmpl_path,
        )

        # Degibbs
        mrdegibbs(input_nii_path=cmpl_path, output_nii_path=cmpl_path, axes=axes)

        # To magnitude and phase
        mrcalc(
            operand1=cmpl_path,
            operand2=None,
            operator="abs",
            output_path=mag_nii_path,
        )
        mrcalc(
            operand1=cmpl_path,
            operand2=None,
            operator="phase",
            output_path=pha_nii_path,
        )

        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


@staticmethod
def N4_corr_funct(
    nii_path,
    level=8,
    debug=False,
):
    """Apply N4 bias-field correction to a NIfTI volume."""

    nii = nib.load(nii_path)
    pixdim = nii.header["pixdim"][1:5].tolist()

    d1 = int(np.ceil(level / pixdim[0]))
    d2 = int(np.ceil(level / pixdim[1]))
    downsample = (d1, d2, 1)
    N4_corr(nii_path=nii_path, downsample=downsample, debug=debug)


@staticmethod
def get_nii_header(nii_path):
    """Read the spatial/temporal dimensions and voxel sizes of a NIfTI file."""

    nii = nib.load(nii_path)
    dim = nii.header["dim"][1:5].tolist()
    pixdim = nii.header["pixdim"][1:5].tolist()

    return dim, pixdim


@staticmethod
def set_nii_time_res_header(
    nii_path,
    time_res,
    toffset=0.0,
    xyzt_units=18,  # mm / ms
):
    """Set the temporal resolution in a NIfTI file's header, in place."""

    nii = nib.load(nii_path)
    nii.header["pixdim"][4] = time_res
    nii.header["toffset"] = toffset
    nii.header["xyzt_units"] = xyzt_units
    nib.save(nii, nii_path)


@staticmethod
def get_nii_orientation(nii_path):
    """Infer the acquisition orientation of a NIfTI volume.

    Identifies the slice-select axis as the smallest of the three
    spatial dimensions, then classifies its alignment with the
    RAS+ coordinate system.
    """

    # Load nifti
    nii = nib.load(nii_path)
    affine = nii.affine
    header = nii.header

    # arccos(0.8) = 37 degress
    oblique_threshold = 0.8

    # Slice orientation
    slice_idx = np.argmin(header["dim"][1:4])  # Not great
    slice_vec = affine[:3, slice_idx]
    slice_vec = slice_vec / np.linalg.norm(slice_vec)

    # Alignment with RAS+ coordinate system
    alignment = np.abs(slice_vec)
    dominant_axis = np.argmax(alignment)
    strength = alignment[dominant_axis]

    if strength < oblique_threshold:
        return "oblique"
    if dominant_axis == 2:
        return "tra"  # superior-inferior
    elif dominant_axis == 1:
        return "cor"  # anterior-posterior
    elif dominant_axis == 0:
        return "sag"  # right-left


@staticmethod
def get_z_smooth(nii_path):
    """Compute a z-smoothness metric for automatic template stack selection."""

    # Load nifti
    nii = nib.load(nii_path)
    img = nii.get_fdata()
    dim = nii.header["dim"][1:5].tolist()

    if dim[3] == 1:
        img = img[..., np.newaxis]
    img = np.mean(img, axis=-1)
    mu = np.mean(img)

    if dim[2] < 2:
        raise ValueError(
            f"Cannot compute z-smoothness: stack has only {dim[2]} " f"slice(s) ({nii_path})."
        )
    if mu == 0:
        raise ValueError(
            f"Cannot compute z-smoothness: time-averaged volume has "
            f"zero mean intensity ({nii_path})."
        )

    # Compute z-smoothness
    z_smooth = np.std(
        [np.mean(np.abs((img[:, :, z] - img[:, :, z + 1]) / mu)) for z in range(0, dim[2] - 1)]
    )

    return z_smooth
