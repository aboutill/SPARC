import os
import logging
import datetime

from sparc.tools.mrtrix import mrmath


def run(
        self,
        cine_nii_path,
        output_dir,
        profile=False,
        debug=False,
    ):
    """Segment the heart region on the time-averaged 3D+time reconstructed 
    volume.
    """
    
    logging.info("Heart segmentation...")
    
    if profile or debug:
        start_time = datetime.datetime.now()
        
    cine_nii_path = os.path.abspath(cine_nii_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Time-averaged 3D volume
    filename = os.path.basename(cine_nii_path).split(".nii.gz")[0]
    nii_3D_path = os.path.join(output_dir, f"{filename}_3D.nii.gz")
    mrmath(
        input_nii_path=cine_nii_path,
        operation="mean",
        output_nii_path=nii_3D_path,
        axis=3,
    )
    
    suffix = "heart_mask_auto"
    mask_filename = f"{filename}_{suffix}.nii.gz"
    mask_path = os.path.join(output_dir, mask_filename)
    qc_report_path = os.path.join(output_dir, f"{filename}_{suffix}_qc.json")
    indiv_pred_dir = os.path.join(output_dir, f"{filename}_{suffix}_indiv_pred") if debug else None
    
    # Run segmentator
    self.ensemble_tester.run_from_file(
        input_path=nii_3D_path,
        output_path=mask_path,
        models_dir=self.models_dir,
        qc_report_path=qc_report_path,
        indiv_pred_dir=indiv_pred_dir,
    )
    
    # Print Quality Control
    self.print_qc(qc_report_path=qc_report_path)
    
    if not debug:
        os.remove(nii_3D_path)
    
    if profile or debug:
        elapsed_time = datetime.datetime.now() - start_time
        logging.info(f"Time for heart segmentation: {elapsed_time}")
    
    return mask_path, qc_report_path