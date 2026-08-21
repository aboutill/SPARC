import shutil
import logging

import numpy as np
import nibabel as nib
import SimpleITK as sitk

from scipy.ndimage import binary_dilation
from skimage.morphology import ball, cube
from scipy.ndimage import distance_transform_edt, gaussian_filter


def get_nii_range(nii_path):
    """Extract intensity range from a NIfTI volume."""
    
    # Load mask
    nii = nib.load(nii_path)
    img = nii.get_fdata()
    return np.max(img) - np.min(img)


def N4_corr(
        nii_path,
        debug=False,
        downsample=(2, 2, 2),
    ):
    """Apply N4 bias-field correction to a 4D NIfTI volume.

    The bias field is estimated once, from a spatially-downsampled
    version of the first temporal frame, and applied identically to
    every frame.
    """

    # Save uncorrected image
    if debug:
        nii_debug_path = nii_path.replace(".nii.gz", "_N4_uncorr.nii.gz")
        shutil.copy(nii_path, nii_debug_path)

    # Load nifti
    nii = nib.load(nii_path)
    dim = nii.header["dim"][1:5].tolist()
    affine = nii.affine
    header = nii.header
    pixdim = header["pixdim"][1:4].tolist()

    img = nii.get_fdata()
    if dim[3] == 1:
        img = img[..., np.newaxis]

    # Init N4 corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    sitk_img = sitk.GetImageFromArray(img[..., 0])
    sitk_img.SetSpacing(pixdim)

    if len(downsample) != sitk_img.GetDimension():
        raise ValueError(
            f"downsample must have one factor per spatial dimension "
            f"({sitk_img.GetDimension()}), got {downsample}."
        )
    sitk_img_down = sitk.Shrink(sitk_img, downsample)

    _ = corrector.Execute(sitk_img_down)
    log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_img)

    # Apply the single estimated bias field over all timepoints
    for t in range(dim[3]):
        sitk_img = sitk.GetImageFromArray(img[..., t])
        sitk_img.SetSpacing(pixdim)
        N4_corrected_sitk_img = sitk_img / sitk.Cast(sitk.Exp(log_bias_field), sitk.sitkFloat64)
        img[..., t] = sitk.GetArrayFromImage(N4_corrected_sitk_img)

    if dim[3] == 1:
        img = np.squeeze(img, axis=-1)
    img = np.clip(np.round(img), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    # Update nifti
    nii = nib.Nifti1Image(img, affine, header)
    nib.save(nii, nii_path)    
    
    
def dilate_mask(
        mask_path,
        dil_rad=0.0,
        dil_strc="ellipsoid",
    ):    
    """Binary-dilate a NIfTI mask in place by a physical radius."""
    
    # Load mask
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata()
    
    # Resolution
    vox = mask_nii.header["pixdim"][1:4]
    
    if dil_rad < 0:
        return
        
    # Define structure element
    if dil_strc == "ball":
        dilation_radius = np.max(np.ceil(dil_rad / vox)) # from mm to voxel
        structure = ball(dilation_radius)
    elif dil_strc == "cube":
        dilation_radius = np.max(np.ceil(dil_rad / vox)) # from mm to voxel
        structure = cube(dilation_radius)
    elif dil_strc == "ellipsoid":
        # Init ellipsoidal structural element
        center = np.ceil(dil_rad / vox)
        sz = np.array(2*center + 1, dtype=int)
        distance = np.zeros(sz)
        for i in range(sz[0]):
            for j in range(sz[1]):
                for k in range(sz[2]):
                    distance[i,j,k] = np.linalg.norm(([i, j, k] - center) * vox)
        structure = np.ones(sz) * (distance <= max(center*vox))
    else:
        raise ValueError(
            f"Unrecognised dil_strc '{dil_strc}'; expected "
            f"'ball', 'cube', or 'ellipsoid'."
        )
        
    # Perform binary dilation
    mask = binary_dilation(mask, structure=structure)
    
    # Save mask
    output_mask_nii = nib.Nifti1Image(mask.astype(np.uint8), mask_nii.affine, mask_nii.header)
    nib.save(output_mask_nii, mask_path)
    
    
def remove_zero_filling(
        mag_nii_path,
        pha_nii_path=None,
        acq_mat=None,
        debug=False,
    ):
    """Remove k-space zero-filling interpolation from a reconstructed
    NIfTI volume, in place.
    """
    def _acquisition_matrix_to_shape(acq_mat):
        """Derive the true 2D (X, Y) acquired matrix size from the raw
        4-element DICOM Acquisition Matrix tag
        [freq_rows, freq_cols, phase_rows, phase_cols], of which exactly
        two entries are expected to be non-zero.
        """
        acq_mat = list(acq_mat)
        if len(acq_mat) != 4:
            logging.warning(f"Unexpected Acquisition Matrix format: {acq_mat}; skipping.")
            return None

        freq_rows, freq_cols, phase_rows, phase_cols = acq_mat
        if freq_rows and phase_cols:
            return int(freq_rows), int(phase_cols)
        elif freq_cols and phase_rows:
            return int(phase_rows), int(freq_cols) 
        else:
            logging.warning(f"Unable to interpret Acquisition Matrix {acq_mat}; skipping.")
            return None
        
    def _center_crop_xy(arr, target_shape):
        """Symmetrically crop the first two (in-plane X, Y) axes of arr
        to target_shape, keeping the centered region. Assumes a
        symmetric (non-partial-Fourier) acquisition."""
        cur_x, cur_y = arr.shape[0], arr.shape[1]
        tgt_x, tgt_y = target_shape
        start_x = max((cur_x - tgt_x) // 2, 0)
        start_y = max((cur_y - tgt_y) // 2, 0)
        return arr[start_x:start_x + tgt_x, start_y:start_y + tgt_y, ...]

    def _recentre_affine_after_inplane_resize(old_affine, old_shape, new_shape):
        """Return a new affine for an array whose first two (X, Y) axes
        changed size via a centred crop/pad, preserving orientation and
        keeping the array's centre at the same world-space location.
        Z (and any further axes) are assumed unchanged."""
        old_affine = np.asarray(old_affine, dtype=np.float64)
        new_affine = old_affine.copy()
    
        scale_x = old_shape[0] / new_shape[0]
        scale_y = old_shape[1] / new_shape[1]
        new_affine[:3, 0] *= scale_x
        new_affine[:3, 1] *= scale_y
    
        ndim = old_affine.shape[0] - 1  # 3
        old_center = np.array([(s - 1) / 2.0 for s in old_shape[:ndim]] + [1.0])
        new_center = np.array(
            [(new_shape[0] - 1) / 2.0, (new_shape[1] - 1) / 2.0]
            + [(s - 1) / 2.0 for s in old_shape[2:ndim]] + [1.0]
        )
    
        world_center = old_affine @ old_center
        new_affine[:3, 3] = world_center[:3] - new_affine[:3, :3] @ new_center[:3]
        return new_affine
    
    if acq_mat is None:
        return

    acq_shape = _acquisition_matrix_to_shape(acq_mat)
    if acq_shape is None:
        return

    mag_nii = nib.load(mag_nii_path)
    mag = mag_nii.get_fdata()
    recon_shape = mag.shape[:2]

    if acq_shape[0] >= recon_shape[0] and acq_shape[1] >= recon_shape[1]:
        # Acquired matrix already covers (or exceeds) the reconstructed
        # matrix -- no zero-filling interpolation to remove.
        return

    if debug:
        debug_mag_nii_path = mag_nii_path.replace(".nii.gz", "_pre_zero_filling.nii.gz")
        shutil.copy(mag_nii_path, debug_mag_nii_path)
        if pha_nii_path is not None:
            debug_pha_nii_path = pha_nii_path.replace(".nii.gz", "_pre_zero_filling.nii.gz")
            shutil.copy(pha_nii_path, debug_pha_nii_path)

    if pha_nii_path is not None:
        pha_nii = nib.load(pha_nii_path)
        pha = pha_nii.get_fdata()
        cplx = mag * np.exp(1j * pha)
    else:
        cplx = mag.astype(np.complex128)

    # Centered FFT convention: DC at the array centre in both image
    # and k-space, avoiding a spurious linear phase ramp.
    kspace = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(cplx, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    kspace_cropped = _center_crop_xy(kspace, acq_shape)

    cplx_corrected = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace_cropped, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    mag_corrected = np.abs(cplx_corrected).astype(mag.dtype)
    
    new_affine = _recentre_affine_after_inplane_resize(
        mag_nii.affine, recon_shape + mag.shape[2:], acq_shape + mag.shape[2:],
    )
    mag_nii_new = nib.Nifti1Image(mag_corrected, new_affine)
    new_zooms = np.abs(np.linalg.norm(new_affine[:3,:3], axis=0))
    mag_nii_new.header.set_zooms(np.concatenate([new_zooms, [1]]))
    nib.save(mag_nii_new, mag_nii_path)

    if pha_nii_path is not None:
        pha_corrected = np.angle(cplx_corrected).astype(pha.dtype)
        
        pha_nii_new = nib.Nifti1Image(pha_corrected, new_affine)
        pha_nii_new.header.set_zooms(np.concatenate([new_zooms, [1]]))
        nib.save(pha_nii_new, pha_nii_path)
        
        
def roll_nii(nii_path, t_start):
    
    nii = nib.load(nii_path)
    img = nii.get_fdata()
    header = nii.header
    affine = nii.affine

    img = np.roll(img, -t_start, axis=-1)
    nii = nib.Nifti1Image(img, affine, header)
    nib.save(nii, nii_path)
    

def smooth_roi_mask(
        cine_nii_path, 
        smooth_sigma_mm=1.0,
        debug=False,
    ):
    """Smooth ROI mask in cine volume."""
    
    if debug:
        debug_cine_nii_path = cine_nii_path.replace(".nii.gz", "_pre_roi_mask_smoothing.nii.gz")
        shutil.copy(cine_nii_path, debug_cine_nii_path)

    nii = nib.load(cine_nii_path)
    cine = nii.get_fdata()
    pixdim = nii.header["pixdim"][1:4]

    # Time-average
    cine_3D = cine.mean(axis=-1) if cine.ndim == 4 else cine
    mask = cine_3D > 1e-6

    dist_outside = distance_transform_edt(~mask, sampling=pixdim)
    dist_inside = distance_transform_edt(mask, sampling=pixdim)
    sdf = dist_outside - dist_inside
    sdf = gaussian_filter(sdf, sigma=smooth_sigma_mm / pixdim)

    mask_smooth = (sdf <= 0).astype(np.float32)
    cine_masked = cine * mask_smooth[..., None]
    
    nii = nib.Nifti1Image(cine_masked, nii.affine, nii.header)
    nib.save(nii, cine_nii_path)