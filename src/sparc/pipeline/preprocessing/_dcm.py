import os
import glob
import tempfile
import shutil
import pydicom
import logging


def sort_dcm_series(self, input_dcm_dir):
    """Discover and sort all DICOM stacks under a subject's raw
    DICOM directory into per-stack, per-type (magnitude/phase)
    subfolders of a temporary directory.
    """

    reg_ex = os.path.join(input_dcm_dir, "**", "*")
    dcm_paths = glob.glob(reg_ex, recursive=True)
    dcm_paths = sorted([
        path for path in dcm_paths
        if (os.path.isfile(path)
            and os.path.splitext(path)[1] in (".dcm", ".DCM", "")
            and os.path.basename(path) != ".DS_Store")
    ])

    records = []
    for dcm_path in dcm_paths:
        dcm = pydicom.dcmread(dcm_path)

        dcm_type = None
        if (self.img_type_dcm_tag is not None
                and self.mag_dcm_flag is not None
                and self.pha_dcm_flag is not None
                and hasattr(dcm, self.img_type_dcm_tag)):
            image_type = dcm[self.img_type_dcm_tag].value
            if self.mag_dcm_flag in image_type:
                dcm_type = "mag"
            elif self.pha_dcm_flag in image_type:
                dcm_type = "pha"

        dcm_id = None
        if (self.stack_id_dcm_tag is not None
                and hasattr(dcm, self.stack_id_dcm_tag)):
            raw_id = dcm[self.stack_id_dcm_tag].value
            if raw_id and dcm_type == "mag":
                dcm_id = f"s{raw_id}"
            if raw_id and dcm_type == "pha":
                dcm_id = f"s{raw_id-1}" # Only works if mag/pha are split into two consecutive series

        records.append((dcm_path, dcm_type, dcm_id))

    tempdir = tempfile.mkdtemp()
    stack_ids_seen = set()

    for dcm_path, dcm_type, dcm_id in records:
        if dcm_type is None or dcm_id is None:
            continue

        output_dir = os.path.join(tempdir, f"{dcm_id}_{dcm_type}")
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(dcm_path, output_dir)
        stack_ids_seen.add(dcm_id)

    stack_mag_dcm_dirs = []
    stack_pha_dcm_dirs = []
    stack_ids = []

    for stack_id in sorted(stack_ids_seen):
        stack_mag_dcm_dir = os.path.join(tempdir, f"{stack_id}_mag")
        stack_pha_dcm_dir = os.path.join(tempdir, f"{stack_id}_pha")

        if not os.path.isdir(stack_mag_dcm_dir) or not os.listdir(stack_mag_dcm_dir):
            logging.warning(
                f"Stack {stack_id}: no magnitude DICOM files found; skipping."
            )
            continue

        stack_mag_dcm_dirs.append(stack_mag_dcm_dir)
        stack_ids.append(stack_id)

        if os.path.isdir(stack_pha_dcm_dir) and os.listdir(stack_pha_dcm_dir):
            stack_pha_dcm_dirs.append(stack_pha_dcm_dir)
        else:
            stack_pha_dcm_dirs.append(None)

    return stack_mag_dcm_dirs, stack_pha_dcm_dirs, stack_ids

            
def read_rr_intervals_from_dcm_series(
        self,
        dcm_paths,
    ):
    """Read the RR interval (ms) from each of a list of DICOM files."""

    # Default values (in ms)
    default_rr_interval = 400 
    
    # Extract RR intervals
    rr_intervals = []
    for dcm_path in dcm_paths:
        dcm = pydicom.dcmread(dcm_path)
        if (self.rr_interval_dcm_tag is not None
            and hasattr(dcm, self.rr_interval_dcm_tag)):
            rr_interval = dcm[self.rr_interval_dcm_tag].value
            if not rr_interval:
                rr_intervals.append(default_rr_interval)
            else:
                rr_intervals.append(float(rr_interval))
        else:
            rr_intervals.append(default_rr_interval)
                
    return rr_intervals
    

def read_id_from_dcm(
        self,
        dcm_path,
    ):
    """Read the stack/subject ID from a single DICOM file."""

    dcm = pydicom.dcmread(dcm_path)
    
    # Extract ID from DICOM
    if (self.stack_id_dcm_tag is not None 
        and hasattr(dcm, self.stack_id_dcm_tag)):
        stack_id = dcm[self.stack_id_dcm_tag].value
        if not stack_id:
            stack_id = None
        else:
            stack_id = str(stack_id) 
    else:
        stack_id = None
    
    return stack_id


def read_acquisition_matrix_from_dcm(
        self,
        dcm_path,
    ):
    """Read the acquisition matrix from a single DICOM file."""

    dcm = pydicom.dcmread(dcm_path)
    
    # Extract acquisition matrix
    if (self.acq_mat_dcm_tag is not None 
        and hasattr(dcm, self.acq_mat_dcm_tag)):
        acq_mat = dcm[self.acq_mat_dcm_tag].value
        if not acq_mat or all(dim == 0 for dim in acq_mat):
            acq_mat = None
    else:
        acq_mat = None
        
    return acq_mat


def read_slice_thickness_from_dcm(
        self,
        dcm_path,
    ):
    """Read the slice thickness (mm) from a single DICOM file."""

    dcm = pydicom.dcmread(dcm_path)
    
    # Extract slice thickness from DICOM
    if (self.slice_thickness_dcm_tag is not None 
        and hasattr(dcm, self.slice_thickness_dcm_tag)):
        slice_thickness = dcm[self.slice_thickness_dcm_tag].value
        if not slice_thickness:
            slice_thickness = None
        else:
            slice_thickness = float(slice_thickness) 
    else:
        slice_thickness = None
    
    return slice_thickness