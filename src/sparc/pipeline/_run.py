import os
import shutil
import datetime
import logging


def run(
        self,
        input_dir,
        output_dir,
        mode,
        gui_mode,
        manual_stack_review=False,
        file_prefix=None,
        verbose=False,
        profile=False,
        debug=False,
        log=False,
    ):
    """Run the full SPARC pipeline on one subject's raw DICOM input."""
    
    def _clean_up():
        
        for d in out_dirs:
            if not os.listdir(d):
                shutil.rmtree(d, ignore_errors=True)
                
        if not os.listdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
            
    
    start_time = datetime.datetime.now()
    self.gui_elapsed = datetime.timedelta(0)
    
    if mode == "manual":
        self.heart_segmentator.activate = True
        self.heart_segmentator.activate_gui = True
        
    # Set output directories
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    stacks_nii_dir = os.path.join(output_dir, "stacks_nii")
    chest_mask_dir = os.path.join(output_dir, "chest_mask")
    cine_nii_dir = os.path.join(output_dir, "cine_nii")
    heart_mask_dir = os.path.join(output_dir, "heart_mask")
    cine_dcm_dir = os.path.join(output_dir, "cine_dcm")
    out_dirs = [stacks_nii_dir, chest_mask_dir, cine_nii_dir,
                heart_mask_dir, cine_dcm_dir]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    # Initialize output dir
    os.makedirs(output_dir, exist_ok=True)
        
    # Setup logging
    filename = f"{file_prefix}_SPARC" if file_prefix is not None else "SPARC" 
    log_path = os.path.join(output_dir, f"{filename}.log") if log else None
    self.setup_logging(log_path=log_path, verbose=verbose)
    
    # Print pipeline info
    if verbose:
        self.print_pipeline_info()
    
    # Save pipeline_info
    if debug:
        report_path = os.path.join(output_dir, f"{filename}.yaml")
        self.save_pipeline_info(
            input_dir=input_dir,
            output_path=report_path,
        )
    
    # Run preprocessing with optional gui
    (stack_mag_nii_paths,
     stack_pha_nii_paths,
     stack_infos,
     gui_elapsed) = self.pre_processor.run_with_gui(
        input_dir=input_dir,
        output_dir=stacks_nii_dir,
        mode=mode,
        gui_mode=gui_mode,
        manual_stack_review=manual_stack_review,
        file_prefix=file_prefix,
        profile=profile,
        debug=debug,
    )
    
    # Validation
    if stack_mag_nii_paths is None:
        _clean_up()
        return
    
    # Update GUI timer
    if gui_elapsed is not None:
        self.gui_elapsed += gui_elapsed
    
    # Run chest segmentation with gui
    (stack_mag_nii_paths,
     stack_pha_nii_paths,
     stack_infos,
     chest_mask_path,
     chest_mask_qc_path,
     gui_elapsed) = self.chest_segmentator.run_with_gui(
        stack_mag_nii_paths=stack_mag_nii_paths,
        stack_pha_nii_paths=stack_pha_nii_paths,
        stack_infos=stack_infos,
        output_dir=chest_mask_dir,
        mode=mode,
        gui_mode=gui_mode,
        profile=profile,
        debug=debug,
    )
    
    # Validation
    if chest_mask_path is None or not os.path.exists(chest_mask_path): 
        _clean_up()
        return
    
    # Update GUI timer
    if gui_elapsed is not None:
        self.gui_elapsed += gui_elapsed
    
    # Run SVR reconstruction
    cine_nii_path, cine_qc_report_path = self.svr_reconstructor.run(
        stack_nii_paths=stack_mag_nii_paths,
        stack_infos=stack_infos,
        output_dir=cine_nii_dir,
        file_prefix=file_prefix,
        profile=profile,
        debug=debug,
        verbose=verbose,
        log=log,
    )      
    
    # Validation
    if cine_nii_path is None or not os.path.exists(cine_nii_path):
        _clean_up()
        return
    
    # Run heart segmentation with gui
    heart_mask_path = None
    heart_mask_qc_path = None
    if self.heart_segmentator.activate:
        (heart_mask_path,
         heart_mask_qc_path,
         gui_elapsed) = self.heart_segmentator.run_with_gui(
             cine_nii_path=cine_nii_path,
             output_dir=heart_mask_dir,
             mode=mode,
             gui_mode=gui_mode,
             profile=profile,
             debug=debug,
         )
        
        # Validation
        if heart_mask_path is None or not os.path.exists(heart_mask_path):
            _clean_up()
            return
        
        # Update GUI timer
        if gui_elapsed is not None:
            self.gui_elapsed += gui_elapsed
        
    # Run reorientation network with gui
    (reo_cine_nii_path,
    reo_qc_report_path,
    gui_elapsed) = self.reorientor.run_with_gui(
        cine_nii_path=cine_nii_path,
        heart_mask_path=heart_mask_path,
        stack_nii_paths=stack_mag_nii_paths+[chest_mask_path],
        output_dir=cine_nii_dir,
        mode=mode,
        gui_mode=gui_mode,
        profile=profile,
        debug=debug,
    )
    
    # Validation
    if reo_cine_nii_path is None or not os.path.exists(reo_cine_nii_path):
        _clean_up()
        return
    
    # Update GUI timer
    if gui_elapsed is not None:
        self.gui_elapsed += gui_elapsed
    
    # Run post-processing
    self.post_processor.run(
        cine_nii_path=reo_cine_nii_path,
        stack_infos=stack_infos,
        output_dir=cine_dcm_dir,
        profile=profile,
        debug=debug,
    )
    
    elapsed_time = datetime.datetime.now() - start_time
    processing_time = elapsed_time - self.gui_elapsed
    elapsed_seconds = elapsed_time.total_seconds()
    processing_seconds = processing_time.total_seconds()
    
    filename = f"{file_prefix}_SPARC" if file_prefix is not None else "SPARC" 
    pipeline_qc_report_path = os.path.join(output_dir, f"{filename}_qc.json")
    self.save_qc_report(
        chest_mask_qc_path=chest_mask_qc_path,
        cine_qc_report_path=cine_qc_report_path,
        heart_mask_qc_path=heart_mask_qc_path,
        reo_qc_report_path=reo_qc_report_path,
        elapsed_seconds=elapsed_seconds,
        processing_seconds=processing_seconds,
        output_path=pipeline_qc_report_path,
    )
    
    logging.info(
        f"Time for SPARC pipeline: {elapsed_time} total "
        f"({processing_time} processing, {self.gui_elapsed} review wait)"
    )  
    
    _clean_up()