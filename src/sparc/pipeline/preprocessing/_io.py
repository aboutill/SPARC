import os
import shutil
import json
import logging


def print_model_info(self):
    """Log the configured preprocessing parameters."""

    logging.info("Preprocessing parameters:")
    if self.img_type_dcm_tag is not None and self.mag_dcm_flag is not None:
        logging.info(f"  DICOM Magnitude '{self.img_type_dcm_tag}:{self.mag_dcm_flag}'")
    if self.img_type_dcm_tag is not None and self.pha_dcm_flag is not None:
        logging.info(f"  DICOM Phase '{self.img_type_dcm_tag}:{self.pha_dcm_flag}'")
    if self.stack_id_dcm_tag is not None:
        logging.info(f"  DICOM StackID '{self.stack_id_dcm_tag}'")
    if self.rr_interval_dcm_tag is not None:
        logging.info(f"  DICOM RR Interval '{self.rr_interval_dcm_tag}'")
    if self.acq_mat_dcm_tag is not None:
        logging.info(f"  DICOM '{self.acq_mat_dcm_tag}'")
    if self.slice_thickness_dcm_tag is not None:
        logging.info(f"  DICOM '{self.slice_thickness_dcm_tag}'")

    if self.remove_zero_filling:
        logging.info("  Zero-filling removal activated")
    if self.denoise:
        logging.info("  Denoising activated")
    if self.degibbs:
        logging.info("  Degibbs activated")
    if self.N4_corr:
        logging.info("  Bias-field correction activated")


@staticmethod
def print_stack_infos(stack_infos):
    """Log a one-line summary per preprocessed stack."""

    logging.info("Input stack information:")
    for stack_info in stack_infos:
        stack_id = stack_info["ID"]
        ornt = stack_info["ornt"]
        rr_interval = stack_info["rr_interval"]
        z_smooth = stack_info["z_smooth"]
        slice_thickness = stack_info["slice_thickness"]
        pixdim = stack_info["pixdim"]

        logging.info(f"  Stack {stack_id}:")
        logging.info(f"    Orientation: {ornt}")
        logging.info(f"    Resolution: {pixdim[0]:.2f}x{pixdim[1]:.2f} [mm2]")
        logging.info(f"    Slice thickness: {slice_thickness:.2f} [mm]")
        logging.info(f"    RR interval: {rr_interval:.2f} [ms]")
        logging.info(f"    z-smooth: {z_smooth:.2f}")


@staticmethod
def move_excluded_stacks(
    excluded_stack_mag_nii_paths,
    excluded_stack_pha_nii_paths,
    excluded_stack_infos,
):
    """Move excluded stacks to 'excluded' subdirectory and
    update excluded_stack_infos dictionary."""
    if not excluded_stack_mag_nii_paths:
        return excluded_stack_infos

    n = len(excluded_stack_mag_nii_paths)
    assert len(excluded_stack_pha_nii_paths) == n, (
        "excluded_stack_pha_nii_paths must be the same length as " "excluded_stack_mag_nii_paths."
    )
    assert len(excluded_stack_infos) == n, (
        "excluded_stack_infos must be the same length as " "excluded_stack_mag_nii_paths."
    )

    n = len(excluded_stack_mag_nii_paths)
    dirname = os.path.dirname(excluded_stack_mag_nii_paths[0])
    output_dir = os.path.join(dirname, "excluded")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(n):
        mag_path = excluded_stack_mag_nii_paths[i]
        new_mag_path = os.path.join(output_dir, os.path.basename(mag_path))
        shutil.move(mag_path, new_mag_path)
        excluded_stack_infos[i]["MAG_NIFTI"] = new_mag_path
        if excluded_stack_pha_nii_paths[i] is not None:
            pha_path = excluded_stack_pha_nii_paths[i]
            new_pha_path = os.path.join(output_dir, os.path.basename(pha_path))
            shutil.move(pha_path, new_pha_path)
            excluded_stack_infos[i]["PHA_NIFTI"] = new_pha_path

    return excluded_stack_infos


@staticmethod
def save_stack_infos(
    stack_infos,
    output_path,
    excluded_stack_infos=None,
):
    """Save per-stack metadata to a JSON file."""

    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    report = {"included": stack_infos, "excluded": excluded_stack_infos}

    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
