import os
import glob
import logging
import datetime

import numpy as np


@staticmethod
def compute_rr_interval(
        dim,
        rr_intervals,
    ):
    """Compute per-slice and per-stack representative RR intervals."""
              
    # RR interval per slice
    rr_interval_slices = np.median(
        np.reshape(rr_intervals, (dim[2], dim[3])),
        axis=1,
    ).tolist()
    # RR interval (stack)
    rr_interval = np.median(rr_intervals)
    return rr_interval, rr_interval_slices


def run(
        self,
        input_dir,
        output_dir,
        file_prefix=None,
        profile=False,
        debug=False,
    ):
    """Preprocess every DICOM stack found under a subject's input directory."""
    
    logging.info("DICOM to NIfTI conversion...")
    
    # Initialize output dir
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    stack_mag_dcm_dirs, stack_pha_dcm_dirs, stack_ids = self.sort_dcm_series(
        input_dcm_dir=input_dir,
    )
    
    n = len(stack_mag_dcm_dirs)
    
    # Initialize datalists
    stack_mag_nii_paths = []
    stack_pha_nii_paths = []
    stack_infos = []
    
    # Iterate over stacks
    for i in range(n):
        
        # Initialize timer
        if profile or debug:
            start_time = datetime.datetime.now()
        
        stack_mag_dcm_dir = stack_mag_dcm_dirs[i]
        stack_pha_dcm_dir = stack_pha_dcm_dirs[i]
        stack_id = stack_ids[i]
        pha_exist = True if stack_pha_dcm_dir is not None else False
       
        # Display step
        logging.info(f"Converting stack {stack_id}...")
        
        # Get series as list
        stack_mag_dcm_paths = glob.glob(os.path.join(stack_mag_dcm_dir, "*"))
 
        # Read RR intervals 
        rr_intervals = self.read_rr_intervals_from_dcm_series(
            dcm_paths=stack_mag_dcm_paths,
        )
        
        # Get 1st dicom serie
        stack_mag_dcm_path = stack_mag_dcm_paths[0]
        
        # Read acquisition matrix
        acq_mat = self.read_acquisition_matrix_from_dcm(
            dcm_path=stack_mag_dcm_path,
        )
        
        # Read slice thickness
        slice_thickness = self.read_slice_thickness_from_dcm(
            dcm_path=stack_mag_dcm_path,
        )
            
        # Initialize output magnitude and phase paths
        stack_file_prefix = f"{file_prefix}_{stack_id}" if file_prefix is not None else stack_id 
        stack_mag_nii_path = os.path.join(output_dir, f"{stack_file_prefix}_mag.nii.gz")
        stack_pha_nii_path = os.path.join(output_dir, f"{stack_file_prefix}_pha.nii.gz") if pha_exist else None
        stack_meta_nii_path = os.path.join(output_dir, f"{stack_file_prefix}.json") if debug else None
        
        # Convert magnitude and phase DICOM to NIFTI
        self.dcm2nii(
            dcm_dir=stack_mag_dcm_dir, 
            nii_path=stack_mag_nii_path,
            meta_path=stack_meta_nii_path,
        )
        if pha_exist:
            self.dcm2nii(
                dcm_dir=stack_pha_dcm_dir, 
                nii_path=stack_pha_nii_path,
                to_rad=True,
            )
            
        # Image preprocessing
        if self.remove_zero_filling and acq_mat is not None:
            self.remove_zero_filling_funct(
                mag_nii_path=stack_mag_nii_path, 
                pha_nii_path=stack_pha_nii_path,
                acq_mat=acq_mat,
                debug=debug,
            )
        if self.denoise:
            self.denoise_funct(
                mag_nii_path=stack_mag_nii_path, 
                pha_nii_path=stack_pha_nii_path,
                debug=debug,
            )
        if self.degibbs:
            self.degibbs_funct(
                mag_nii_path=stack_mag_nii_path, 
                pha_nii_path=stack_pha_nii_path,
                debug=debug,
            )
        if self.N4_corr:
            self.N4_corr_funct(
                nii_path=stack_mag_nii_path,
                debug=debug,
            )
            
        # Extract info
        ornt = self.get_nii_orientation(nii_path=stack_mag_nii_path)
        z_smooth = self.get_z_smooth(nii_path=stack_mag_nii_path)
        dim, pixdim = self.get_nii_header(nii_path=stack_mag_nii_path)
        
        # Compute median RR interval and time resolution
        rr_interval, rr_interval_slices = self.compute_rr_interval(
            dim=dim,
            rr_intervals=rr_intervals,
        )
        time_res = rr_interval / dim[3]     
        
        # Set temporal resolution
        self.set_nii_time_res_header(
            nii_path=stack_mag_nii_path,
            time_res=time_res,
        )
        if stack_pha_nii_path is not None:
            self.set_nii_time_res_header(
                nii_path=stack_pha_nii_path,
                time_res=time_res,
            )
        
        # Get updated nifti header
        dim, pixdim = self.get_nii_header(nii_path=stack_mag_nii_path)
        
        # Stack info
        stack_info = {
            "DICOM": stack_mag_dcm_path,
            "MAG_NIFTI": stack_mag_nii_path,
            "PHA_NIFTI": stack_pha_nii_path,
            "ID": stack_id,
            "ornt": ornt,
            "z_smooth": z_smooth,
            "acq_mat": acq_mat,
            "dim": dim,
            "pixdim": pixdim,
            "slice_thickness": slice_thickness,
            "rr_intervals": rr_intervals,
            "rr_interval_slices": rr_interval_slices,
            "rr_interval": rr_interval,
        }
            
        # Update datalists
        stack_mag_nii_paths.append(stack_mag_nii_path)
        stack_pha_nii_paths.append(stack_pha_nii_path)
        stack_infos.append(stack_info)
            
        # Print timer
        if profile or debug:
            elapsed_time = datetime.datetime.now() - start_time
            logging.info(f"Time for stack conversion: {elapsed_time}")
         
        # Step display
        logging.info(f"Successfully converted stack {stack_id}!")
        
    return stack_mag_nii_paths, stack_pha_nii_paths, stack_infos