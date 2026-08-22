import os

import numpy as np
import nibabel as nib

from sparc.tools.svrlite import svr_reconstruct
from sparc.utils.nii import N4_corr, smooth_roi_mask


@staticmethod
def cine_post_processing(nii_path):
    """Reorient a reconstructed cine volume to RAS+ canonical
    orientation and zero out the reconstruction's background
    sentinel value (-1).
    """

    # Load image
    nii = nib.load(nii_path)

    # To RAS+ orientation
    nii = nib.as_closest_canonical(nii)
    header = nii.header
    affine = nii.affine

    # Set background to 0
    img = nii.get_fdata()
    img[img == -1] = 0

    # Save image
    nii = nib.Nifti1Image(img, affine, header)
    nib.save(nii, nii_path)


@staticmethod
def N4_corr_funct(
    cine_nii_path,
    level=8,
    debug=False,
):
    """Apply N4 bias-field correction to a NIfTI volume."""

    nii = nib.load(cine_nii_path)
    pixdim = nii.header["pixdim"][1:5].tolist()

    d1 = int(np.ceil(level / pixdim[0]))
    d2 = int(np.ceil(level / pixdim[1]))
    d3 = int(np.ceil(level / pixdim[2]))
    downsample = (d1, d2, d3)
    N4_corr(nii_path=cine_nii_path, downsample=downsample, debug=debug)


@staticmethod
def smooth_roi_mask_funct(
    cine_nii_path,
    smooth_sigma_mm=1.0,
    debug=False,
):
    """Smooth ROI mask in cine volume."""
    smooth_roi_mask(
        cine_nii_path=cine_nii_path,
        smooth_sigma_mm=smooth_sigma_mm,
        debug=debug,
    )


def run(
    self,
    stack_nii_paths,
    stack_infos,
    output_dir,
    file_prefix=None,
    verbose=False,
    profile=False,
    debug=False,
    log=False,
):
    """Reconstruct a 3D+time cine volume from a set of gated stacks."""

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    suffix = "cine.nii.gz"
    cine_filename = f"{file_prefix}_{suffix}" if file_prefix is not None else suffix
    cine_nii_path = os.path.join(output_dir, cine_filename)

    # Initialize cfg
    cfg = self.init_svr_cfg(stack_infos=stack_infos)

    # Run recon
    qc_report_path = svr_reconstruct(
        cine_nii_path=cine_nii_path,
        stack_nii_paths=stack_nii_paths,
        profile=profile,
        verbose=verbose,
        debug=debug,
        log=log,
        **cfg,
    )

    # Post-processing
    self.cine_post_processing(nii_path=cine_nii_path)

    # N4 correction
    if self.N4_corr:
        self.N4_corr_funct(cine_nii_path=cine_nii_path, debug=debug)

    # ROI mask smoothing
    if self.smooth_roi_mask:
        self.smooth_roi_mask_funct(cine_nii_path=cine_nii_path, debug=debug)

    return cine_nii_path, qc_report_path
