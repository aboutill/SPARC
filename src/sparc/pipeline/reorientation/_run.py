import os
import torch
import shutil
import logging
import datetime

import numpy as np
import nibabel as nib

from monai.networks.layers import AffineTransform

from sparc.tools.mrtrix import mrmath


@staticmethod
def center_on_heart_mask(
        input_cine_nii_path,
        output_cine_nii_path,
        output_ctr_aff_path=None,
        input_heart_mask_path=None,
        output_heart_mask_path=None,
        max_pad_frac=1/3,
    ):
    """Recentre a cine volume on the segmented heart mask's centre by
    padding the output bounding box so the full original field of view
    is preserved.

    A no-op copy if no heart mask is provided.
    """
    if input_heart_mask_path is None:
        shutil.copy2(input_cine_nii_path, output_cine_nii_path)
        return

    cine_nii = nib.load(input_cine_nii_path)
    mask_nii = nib.load(input_heart_mask_path)
    affine = cine_nii.affine.copy()
    header = cine_nii.header
    cine = cine_nii.get_fdata()
    mask = mask_nii.get_fdata().astype(bool)

    dims = header["dim"][1:4].astype(np.int64)

    vol_thresh = 5e3
    img_center = (dims - 1) // 2
    vol = np.sum(mask) * np.prod(header["pixdim"][1:4])
    if vol < vol_thresh:
        mask_center = img_center
    else:
        mask_center = np.floor(np.median(np.nonzero(mask), axis=1)).astype(np.int64)

    dist_lo = mask_center
    dist_hi = (dims - 1) - mask_center

    max_pad = np.floor(dims * max_pad_frac).astype(np.int64)
    needed_pad_lo = np.maximum(dist_hi - dist_lo, 0)
    needed_pad_hi = np.maximum(dist_lo - dist_hi, 0)
    pad_lo = np.clip(needed_pad_lo, 0, max_pad)
    pad_hi = np.clip(needed_pad_hi, 0, max_pad)

    if np.any(needed_pad_lo > max_pad) or np.any(needed_pad_hi > max_pad):
        logging.warning(
            "center_on_heart_mask: heart mask centroid is far enough "
            f"off-centre that padding was capped at {max_pad_frac:.0%} "
            "of each dimension; some cropping may still occur."
        )

    new_dims = dims + pad_lo + pad_hi
    ctr_cine = np.zeros((*new_dims, cine.shape[3]), dtype=cine.dtype)
    dst_slices = tuple(slice(pl, pl + d) for pl, d in zip(pad_lo, dims)) + (slice(None),)
    ctr_cine[dst_slices] = cine

    # Only low-side padding shifts existing voxel indices; high-side
    # padding is appended after the data and needs no correction.
    shift_mm = -(affine[:3, :3] @ pad_lo)
    affine[:3, 3] += shift_mm

    ctr_cine_nii = nib.Nifti1Image(ctr_cine, affine)
    ctr_cine_nii = nib.as_closest_canonical(ctr_cine_nii)
    nib.save(ctr_cine_nii, output_cine_nii_path)

    if output_ctr_aff_path is not None:
        np.savetxt(output_ctr_aff_path, shift_mm)
        
    if output_heart_mask_path is not None:
        affine = mask_nii.affine.copy()
        
        ctr_mask = np.zeros(new_dims, dtype=mask.dtype)
        dst_slices = tuple(slice(pl, pl + d) for pl, d in zip(pad_lo, dims))
        ctr_mask[dst_slices] = mask
        
        shift_mm = -(affine[:3, :3] @ pad_lo)
        affine[:3, 3] += shift_mm

        ctr_mask_nii = nib.Nifti1Image(ctr_mask.astype(np.uint8), affine)
        ctr_mask_nii = nib.as_closest_canonical(ctr_mask_nii)
        nib.save(ctr_mask_nii, output_heart_mask_path)
    
    
@staticmethod
def apply_affine_to_cine(
        input_cine_nii_path,
        input_aff_path,
        output_cine_nii_path,
        max_pad_frac=1.0,
        content_margin_vox=2,
    ):
    """Resample a cine volume by a predicted (4,4) affine, applied
    identically to every cardiac phase, padding the output bounding
    box so the transformed volume isn't clipped.
    """

    cine_nii = nib.load(input_cine_nii_path)
    cine_nii = nib.as_closest_canonical(cine_nii)

    cine = cine_nii.get_fdata()
    header = cine_nii.header
    dim = header["dim"][1:5]
    pixdim = header["pixdim"][1:4]

    input_aff = np.loadtxt(input_aff_path)

    orig_dims = np.array(dim[:3], dtype=np.float64)
    center = orig_dims / 2.0

    # Content bounding box
    content_mask = cine.max(axis=-1) > 1e-6
    if content_mask.any():
        nz = np.array(np.nonzero(content_mask))  # (3, N)
        content_min_idx = np.maximum(nz.min(axis=1) - content_margin_vox, 0)
        content_max_idx = np.minimum(nz.max(axis=1) + 1 + content_margin_vox, orig_dims)
    else:
        # No content at all (e.g. a failed/empty reconstruction)
        logging.warning(
            "apply_affine_to_cine: no content foundm falling back "
            "to the full array extent for padding."
        )
        content_min_idx = np.zeros(3)
        content_max_idx = orig_dims

    content_min_centered = content_min_idx - center
    content_max_centered = content_max_idx - center

    # Determine padding needed for this specific transform
    R = input_aff[:3, :3]
    t = input_aff[:3, 3]

    corners = np.array([
        [content_min_centered[0] if sx < 0 else content_max_centered[0],
         content_min_centered[1] if sy < 0 else content_max_centered[1],
         content_min_centered[2] if sz < 0 else content_max_centered[2]]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])  # (8, 3)
    transformed_corners = corners @ R.T + t

    required_half_extent = np.max(np.abs(transformed_corners), axis=0)
    pad = np.maximum(0, np.ceil(required_half_extent - center)).astype(int)

    max_pad = np.floor(orig_dims * max_pad_frac).astype(int)
    if np.any(pad > max_pad):
        logging.warning(
            "apply_affine_to_cine: predicted affine would require "
            f"padding {pad.tolist()} voxels to avoid clipping content, "
            f"capped at {max_pad.tolist()}, some cropping may still occur."
        )
        pad = np.minimum(pad, max_pad)

    padded_dims = (orig_dims + 2 * pad).astype(int)  # symmetric: both sides

    # Apply the transform onto the padded cine
    monai_affine = AffineTransform(
        mode="bilinear",
        padding_mode="zeros",
        normalized=False,
        align_corners=True,
        reverse_indexing=True,
        zero_centered=True,
    )

    T = dim[3]
    cine = torch.from_numpy(cine).permute(3, 0, 1, 2).unsqueeze(1)  # (T, 1, X, Y, Z)
    input_aff_t = torch.from_numpy(input_aff)
    theta = input_aff_t.unsqueeze(0).expand(T, -1, -1)  # (T, 4, 4)

    spatial_size = tuple(int(d) for d in padded_dims)
    reo_cine = monai_affine(cine, theta, spatial_size=spatial_size)  # (T, 1, X', Y', Z')

    reo_cine = reo_cine.squeeze(1).permute(1, 2, 3, 0).contiguous().cpu().numpy()
    
    # Crop the transformed output to its own content
    out_dims = np.array(reo_cine.shape[:3])
    content_mask_out = reo_cine.max(axis=-1) > 1e-6

    if content_mask_out.any():
        nz = np.array(np.nonzero(content_mask_out))
        out_min_idx = nz.min(axis=1)
        out_max_idx = nz.max(axis=1) + 1
        max_symmetric_crop = np.minimum(out_min_idx, out_dims - out_max_idx)
        crop = np.maximum(max_symmetric_crop - content_margin_vox, 0).astype(int)
    else:
        logging.warning(
            "apply_affine_to_cine: no content found in the transformed "
            "output, skipping the post-transform crop."
        )
        crop = np.zeros(3, dtype=int)

    if np.any(crop > 0):
        reo_cine = reo_cine[
            crop[0]:out_dims[0]-crop[0],
            crop[1]:out_dims[1]-crop[1],
            crop[2]:out_dims[2]-crop[2],
            :,
        ]

    reo_aff_fRAS = np.eye(4)
    reo_aff_fRAS[0, 0] *= pixdim[0]
    reo_aff_fRAS[1, 1] *= pixdim[1]
    reo_aff_fRAS[2, 2] *= pixdim[2]

    nii_reo_cine = nib.Nifti1Image(reo_cine, reo_aff_fRAS)
    nib.save(nii_reo_cine, output_cine_nii_path)
    
    return pad, crop


@staticmethod
def apply_affine_to_mask(
        input_mask_nii_path,
        input_aff_path,
        output_mask_nii_path,
        pad,
        crop,
    ):
    """Resample a binary mask by a predicted (4,4) affine, padding the
    output bounding box to match the padding already applied by
    apply_affine_to_cine for the same subject/transform.
    """

    mask_nii = nib.load(input_mask_nii_path)
    mask_nii = nib.as_closest_canonical(mask_nii)

    mask = mask_nii.get_fdata()
    header = mask_nii.header
    dim = header["dim"][1:4]
    pixdim = header["pixdim"][1:4]

    input_aff = np.loadtxt(input_aff_path)

    # Determine padding needed for this specific transform
    orig_dims = np.array(dim[:3], dtype=np.float64)
    padded_dims = (orig_dims + 2 * pad).astype(int)  # symmetric: both sides

    # Apply the transform onto the padded mask
    monai_affine = AffineTransform(
        mode="nearest",
        padding_mode="zeros",
        normalized=False,
        align_corners=True,
        reverse_indexing=True,
        zero_centered=True,
    )

    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(1)  # (1, 1, X, Y, Z)
    input_aff_t = torch.from_numpy(input_aff)
    theta = input_aff_t.unsqueeze(0) # (1, 4, 4)

    spatial_size = tuple(int(d) for d in padded_dims)
    reo_mask = monai_affine(mask, theta, spatial_size=spatial_size)  # (1, 1, X', Y', Z')

    reo_mask = reo_mask.squeeze(1).squeeze(0).contiguous().cpu().numpy() > 0.5
    
    out_dims = np.array(reo_mask.shape)
    if np.any(crop > 0):
        reo_mask = reo_mask[
            crop[0]:out_dims[0]-crop[0],
            crop[1]:out_dims[1]-crop[1],
            crop[2]:out_dims[2]-crop[2],
        ]

    reo_aff_fRAS = np.eye(4)
    reo_aff_fRAS[0, 0] *= pixdim[0]
    reo_aff_fRAS[1, 1] *= pixdim[1]
    reo_aff_fRAS[2, 2] *= pixdim[2]

    nii_reo_mask = nib.Nifti1Image(reo_mask.astype(np.uint8), reo_aff_fRAS)
    nib.save(nii_reo_mask, output_mask_nii_path)
    
    
@staticmethod
def apply_affine_to_stacks_header(
        input_cine_nii_path,
        input_stack_nii_paths,
        input_aff_path,
        output_stack_nii_paths,
        pad=None,
        crop=None,
    ):
    """Update raw stacks' NIfTI headers to reflect the predicted
    reorientation, without resampling their pixel data."""
    
    # Load nifti image
    cine_nii = nib.load(input_cine_nii_path)
    cine_nii = nib.as_closest_canonical(cine_nii)
    
    # Load affine and header
    cine = cine_nii.get_fdata()
    header = cine_nii.header
    nii_aff = cine_nii.affine
    pixdim = header["pixdim"][1:4]
    
    # Load affine pred
    input_aff = np.loadtxt(input_aff_path)
    
    # Compute translation affine
    voxel_center = (np.array(cine.shape[0:3]) - 1) / 2.0
    tr_aff = np.eye(4)
    tr_aff[:3, 3] = voxel_center
    reo_aff = nii_aff @ tr_aff @ input_aff @ np.linalg.inv(tr_aff)
    
    # Fix reoriented affine
    reo_aff_fRAS = np.eye(4)
    reo_aff_fRAS[0,0] *= pixdim[0]
    reo_aff_fRAS[1,1] *= pixdim[1]
    reo_aff_fRAS[2,2] *= pixdim[2]
    if pad is not None and crop is not None:
         reo_aff_fRAS[:3, 3] = pixdim * (pad - crop)

    for input_path, output_path in zip(input_stack_nii_paths, output_stack_nii_paths):
        
        # Load nifti
        stack_nii = nib.load(input_path)
        stack = stack_nii.get_fdata()
        stack_affine = stack_nii.affine
        
        # Set affine fRAS
        stack_affine_fRAS = reo_aff_fRAS @ np.linalg.inv(reo_aff) @ stack_affine
        
        # Save image
        nii_reo_stack = nib.Nifti1Image(stack, stack_affine_fRAS)
        nib.save(nii_reo_stack, output_path)


@staticmethod
def load_itksnap_affine(affine_path, flip=False):
    """Parse an ITK transform text file (e.g. from ITK-SNAP's manual
    registration tool) into a (4,4) LPS-to-RAS-corrected affine
    matrix, including both rotation and translation."""
    
    # Read file
    with open(affine_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Extract rotation matrix and translation vector
    param_line = [l for l in lines if l.startswith('Parameters:')][0]
    params = np.array([float(x) for x in param_line.split(':')[1].split()])
    R = params[:9].reshape(3, 3)
    # t = params[9:12]
    
    # # Extract centre
    # param_line = [l for l in lines if l.startswith('FixedParameters:')][0]
    # params = np.array([float(x) for x in param_line.split(':')[1].split()])
    # c = params
    
    # Affine
    itksnap_affine = np.eye(4)
    itksnap_affine[:3, :3] = R
    # itksnap_affine[:3, 3] = t + c - R @ c # Disable as the translation defined by ITKSNAP is ambiguous
    
    # LPS to RAS
    if flip:
        flip = np.diag([-1., -1., 1., 1.])
        itksnap_affine = flip @ itksnap_affine @ flip
    
    return itksnap_affine


def run(
        self,
        cine_nii_path,
        stack_nii_paths,
        output_dir,
        mode,
        heart_mask_path=None,
        profile=False,
        debug=False,
    ):
    """Reorient the reconstructed 3D+time cine volume: recentre on
    the heart mask, predict a canonical-orientation affine via the
    ensemble, and apply it (optionally also updating raw stacks'
    headers)."""
    
    logging.info("Reorientation...")
    
    if profile or debug:
        start_time = datetime.datetime.now()
        
    cine_nii_path = os.path.abspath(cine_nii_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Center on heart
    ctr_cine_nii_path = cine_nii_path.replace(".nii.gz", "_ctr.nii.gz")
    ctr_heart_mask_path = heart_mask_path.replace(".nii.gz", "_ctr.nii.gz") if heart_mask_path is not None else None
    ctr_aff_path = os.path.join(output_dir, "heart_ctr_aff.txt") if heart_mask_path is not None else None
    
    self.center_on_heart_mask(
        input_cine_nii_path=cine_nii_path,
        input_heart_mask_path=heart_mask_path,
        output_cine_nii_path=ctr_cine_nii_path,
        output_heart_mask_path=ctr_heart_mask_path,
        output_ctr_aff_path=ctr_aff_path,
    )
    if mode == "manual":
        if not debug and ctr_aff_path is not None and os.path.exists(ctr_aff_path):
            os.remove(ctr_aff_path)
        
        if profile or debug:
            elapsed_time = datetime.datetime.now() - start_time
            logging.info(f"Time for reorientation: {elapsed_time}")
            
        return None, ctr_cine_nii_path, ctr_heart_mask_path, None, None
    
    # Time-averaged 3D volume
    filename = os.path.basename(ctr_cine_nii_path).split(".nii.gz")[0]
    ctr_3D_nii_path = os.path.join(output_dir, f"{filename}_3D.nii.gz")
    mrmath(
        input_nii_path=ctr_cine_nii_path,
        operation="mean",
        output_nii_path=ctr_3D_nii_path,
        axis=3,
    )
    
    filename = os.path.basename(cine_nii_path).split(".nii.gz")[0]
    suffix = "reo_affine_auto"
    aff_filename = f"{filename}_{suffix}.txt"
    aff_path = os.path.join(output_dir, aff_filename)
    qc_report_path = os.path.join(output_dir, f"{filename}_{suffix}_qc.json")
    indiv_pred_dir = os.path.join(output_dir, f"{filename}_{suffix}_indiv_pred") if debug else None
    
    # Run reorientor
    self.ensemble_tester.run_from_file(
        input_path=ctr_3D_nii_path,
        output_path=aff_path,
        models_dir=self.models_dir,
        qc_report_path=qc_report_path,
        indiv_pred_dir=indiv_pred_dir,
        rotation_avg=self.rotation_avg,
    )
    
    # Apply affine transform
    reo_cine_nii_path = os.path.join(output_dir, f"{filename}_reo_auto.nii.gz")
    cine_pad, cine_crop = self.apply_affine_to_cine(
        input_cine_nii_path=ctr_cine_nii_path,
        input_aff_path=aff_path,
        output_cine_nii_path=reo_cine_nii_path,
    )
    if heart_mask_path is not None:
        reo_heart_mask_path = heart_mask_path.replace(".nii.gz", "_reo.nii.gz")
        self.apply_affine_to_mask(
            input_mask_nii_path=ctr_heart_mask_path,
            input_aff_path=aff_path,
            output_mask_nii_path=reo_heart_mask_path,
            pad=cine_pad,
            crop=cine_crop,
        )
        
    if self.reo_stacks:
        reo_stack_nii_paths = [p.replace(".nii.gz", "_reo.nii.gz")
                               for p in stack_nii_paths]
        
        self.apply_affine_to_stacks_header(
            input_cine_nii_path=ctr_cine_nii_path,
            input_stack_nii_paths=stack_nii_paths,
            input_aff_path=aff_path,
            output_stack_nii_paths=reo_stack_nii_paths,
            pad=cine_pad,
            crop=cine_crop,
        )
    
    # Print Quality Control
    self.print_qc(qc_report_path=qc_report_path)
    
    if not debug:
        os.remove(ctr_3D_nii_path)
        if ctr_aff_path is not None and os.path.exists(ctr_aff_path):
            os.remove(ctr_aff_path)
    
    if profile or debug:
        elapsed_time = datetime.datetime.now() - start_time
        logging.info(f"Time for reorientation: {elapsed_time}")
    
    return (reo_cine_nii_path, ctr_cine_nii_path, ctr_heart_mask_path,
            aff_path, qc_report_path)


def manual_run(
        self,
        input_cine_nii_path, 
        output_cine_nii_path,
        stack_nii_paths,
        itksnap_aff_path,
        input_heart_mask_path=None,
        output_heart_mask_path=None,
        monai_aff_path=None,
    ):
    """Apply a manually-defined (ITK-SNAP) reorientation, optionally
    composed with a prior automatic prediction.
    """
    
    # Load affines
    itksnap_aff = self.load_itksnap_affine(itksnap_aff_path, flip=True)
    if monai_aff_path is  None:
        cine_nii = nib.load(input_cine_nii_path)
        cine_nii = nib.as_closest_canonical(cine_nii)
        nii_aff = cine_nii.affine
        nii_aff[:3, 3] = [0, 0, 0]
        aff = np.linalg.inv(nii_aff) @ itksnap_aff @ nii_aff
    else:
        pred_aff = np.loadtxt(monai_aff_path)
        aff = pred_aff @ itksnap_aff
    np.savetxt(itksnap_aff_path, aff)
    
    # Apply affine transform
    cine_pad, cine_crop = self.apply_affine_to_cine(
        input_cine_nii_path=input_cine_nii_path,
        input_aff_path=itksnap_aff_path,
        output_cine_nii_path=output_cine_nii_path,
    )
    if input_heart_mask_path is not None and output_heart_mask_path is not None:
        self.apply_affine_to_mask(
            input_mask_nii_path=input_heart_mask_path,
            input_aff_path=itksnap_aff_path,
            output_mask_nii_path=output_heart_mask_path,
            pad=cine_pad,
            crop=cine_crop,
        )
    
    if self.reo_stacks:
        reo_stack_nii_paths = [p.replace(".nii.gz", "_reo.nii.gz")
                               for p in stack_nii_paths]
        
        self.apply_affine_to_stacks_header(
            input_cine_nii_path=input_cine_nii_path,
            input_stack_nii_paths=stack_nii_paths,
            input_aff_path=itksnap_aff_path,
            output_stack_nii_paths=reo_stack_nii_paths,
            pad=cine_pad,
            crop=cine_crop,
        )