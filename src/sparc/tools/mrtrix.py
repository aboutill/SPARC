import os
import shutil
import tempfile
import subprocess

MRTRIX_FLAGS = ["-force", "-quiet"]


def mrconvert(dcm_dir, nii_path, json_path=None):
    """Convert a DICOM folder to a NIfTI file via MRtrix's ``mrconvert``."""

    # Initialize output directory
    output_dir = os.path.dirname(nii_path)
    os.makedirs(output_dir, exist_ok=True)

    # Build MRtrix command
    cmd = [
        "mrconvert",
        dcm_dir,
        nii_path,
    ]
    if json_path:
        cmd += [
            "-json_export",
            json_path,
        ]
    cmd += MRTRIX_FLAGS

    # Run subprocess
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"MRtrix mrcalc failed: executable not found ({e})") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix mrcalc failed: {e}") from e


def mrcalc(
    operand1,
    operand2,
    operator,
    output_path,
):
    """Perform voxel-wise arithmetic via MRtrix's ``mrcalc``."""

    # Initialize output directory
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Build MRtrix command
    cmd = ["mrcalc", operand1]
    if operand2 is not None:
        cmd += [operand2]
    cmd += [f"-{operator}"]
    cmd += [output_path]
    cmd += MRTRIX_FLAGS

    # Run subprocess
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"MRtrix mrcalc failed: executable not found ({e})") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix mrcalc failed: {e}") from e


def mrmath(
    input_nii_path,
    operation,
    output_nii_path,
    axis=3,
):
    """Reduce a NIfTI volume along an axis via MRtrix's 'mrmath'."""

    in_place = False
    if input_nii_path == output_nii_path:
        in_place = True
        tempdir = tempfile.mkdtemp()
        input_nii_path = os.path.join(tempdir, os.path.basename(input_nii_path))
        shutil.copy(output_nii_path, input_nii_path)

    # Initialize output directory
    output_dir = os.path.dirname(output_nii_path)
    os.makedirs(output_dir, exist_ok=True)

    # Build MRtrix command
    cmd = [
        "mrmath",
        input_nii_path,
        operation,
        output_nii_path,
        "-axis",
        str(axis),
    ]
    cmd += MRTRIX_FLAGS

    # Run subprocess
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"MRtrix mrmath failed: executable not found ({e})") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix mrmath failed: {e}") from e

    if in_place:
        shutil.rmtree(tempdir, ignore_errors=True)


def dwidenoise(input_nii_path, output_nii_path, estimator="Exp2", extent=(5, 5, 1)):
    """Denoise a NIfTI volume via MRtrix's 'dwidenoise'."""

    # Initialize output directory
    output_dir = os.path.dirname(output_nii_path)
    os.makedirs(output_dir, exist_ok=True)

    in_place = False
    if input_nii_path == output_nii_path:
        in_place = True
        tempdir = tempfile.mkdtemp()
        input_nii_path = os.path.join(tempdir, os.path.basename(input_nii_path))
        shutil.copy(output_nii_path, input_nii_path)

    # Build MRtrix command
    cmd = [
        "dwidenoise",
        input_nii_path,
        output_nii_path,
        "-estimator",
        estimator,
        "-extent",
        ",".join(str(e) for e in extent),
    ]
    cmd += MRTRIX_FLAGS

    # Run subprocess
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"MRtrix mrmath failed: executable not found ({e})") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix mrmath failed: {e}") from e

    if in_place:
        shutil.rmtree(tempdir, ignore_errors=True)


def mrdegibbs(input_nii_path, output_nii_path, axes=(0, 1)):
    """Apply Gibbs-ringing correction via MRtrix's 'mrdegibbs'."""

    # Initialize output directory
    output_dir = os.path.dirname(output_nii_path)
    os.makedirs(output_dir, exist_ok=True)

    in_place = False
    if input_nii_path == output_nii_path:
        in_place = True
        tempdir = tempfile.mkdtemp()
        input_nii_path = os.path.join(tempdir, os.path.basename(input_nii_path))
        shutil.copy(output_nii_path, input_nii_path)

    axes = [str(ax) for ax in axes]
    # Build MRtrix command
    cmd = [
        "mrdegibbs",
        input_nii_path,
        output_nii_path,
        "-axes",
        ",".join(axes),
    ]
    cmd += MRTRIX_FLAGS

    # Run subprocess
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"MRtrix mrmath failed: executable not found ({e})") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MRtrix mrmath failed: {e}") from e

    if in_place:
        shutil.rmtree(tempdir, ignore_errors=True)
