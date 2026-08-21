import os
import logging
import datetime

from sparc.tools.mrtrix import mrmath
from sparc.utils.nii import dilate_mask


def run(
        self,
        stack_nii_path,
        output_dir,
        profile=False,
        debug=False,
    ):
    """Segment the thoracic region on the time-averaged volume of
    one 2D+time stack.
    """
    
    logging.info("Thoracic segmentation...")
    
    if profile or debug:
        start_time = datetime.datetime.now()
        
    stack_nii_path = os.path.abspath(stack_nii_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Time-averaged 3D volume
    filename = os.path.basename(stack_nii_path).split(".nii.gz")[0]
    stack_3D_nii_path = os.path.join(output_dir, f"{filename}_3D.nii.gz")
    mrmath(
        input_nii_path=stack_nii_path,
        operation="mean",
        output_nii_path=stack_3D_nii_path,
        axis=3,
    )
    
    suffix = "chest_mask_auto"
    mask_path = os.path.join(output_dir,  f"{filename}_{suffix}.nii.gz")
    qc_report_path = os.path.join(output_dir, f"{filename}_{suffix}_qc.json")
    indiv_pred_dir = os.path.join(output_dir, f"{filename}_{suffix}_indiv_pred") if debug else None
    
    # Run segmentator
    self.ensemble_tester.run_from_file(
        input_path=stack_3D_nii_path,
        output_path=mask_path,
        models_dir=self.models_dir,
        qc_report_path=qc_report_path,
        indiv_pred_dir=indiv_pred_dir,
    )
    
    # Print Quality Control
    self.print_qc(qc_report_path=qc_report_path)
    
    # Post-processing dilation, applied after QC reporting
    dilate_mask(
        mask_path=mask_path, 
        dil_rad=self.dil_rad,
    )
    
    if not debug:
        os.remove(stack_3D_nii_path)
    
    if profile or debug:
        elapsed_time = datetime.datetime.now() - start_time
        logging.info(f"Time for stack chest segmentation: {elapsed_time}")
    
    return mask_path, qc_report_path