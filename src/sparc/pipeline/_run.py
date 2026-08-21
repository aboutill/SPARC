import os
import shutil
import datetime
import logging
import contextlib


@contextlib.contextmanager
def excluded_from_timer(self):
    """Pause pipeline runtime."""
    pause_start = datetime.datetime.now()
    try:
        yield
    finally:
        self.gui_elapsed += datetime.datetime.now() - pause_start


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
    
    def _clean_up(delete_paths=None):
        
        if delete_paths is not None:
            for del_path in delete_paths:
                if del_path is not None and os.path.exists(del_path):
                    os.remove(del_path)
        
        for d in out_dirs:
            if not os.listdir(d):
                shutil.rmtree(d, ignore_errors=True)
                
        if not os.listdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
            
    start_time = datetime.datetime.now()
    self.gui_elapsed = datetime.timedelta(0)
    
    # Set modes
    self.mode = mode
    self.gui_mode = gui_mode
    self.manual_stack_review = manual_stack_review
    
    if self.mode == "manual":
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
    
    # Run preprocessing
    stack_mag_nii_paths, stack_pha_nii_paths, stack_infos = self.pre_processor.run(
        input_dir=input_dir,
        output_dir=stacks_nii_dir,
        file_prefix=file_prefix,
        profile=profile,
        debug=debug,
    )
    
    # Optional manual stack review
    excluded_stack_mag_nii_paths = None
    excluded_stack_pha_nii_paths = None
    excluded_stack_infos = None
    if self.manual_stack_review and self.mode != "full_auto":
        with self.excluded_from_timer():
            outs = self.stack_review_gui(
                stack_mag_nii_paths=stack_mag_nii_paths,
                stack_pha_nii_paths=stack_pha_nii_paths,
                stack_infos=stack_infos,
            )
            stack_mag_nii_paths = outs[0]
            stack_pha_nii_paths = outs[1]
            stack_infos = outs[2]
            excluded_stack_mag_nii_paths = outs[3]
            excluded_stack_pha_nii_paths = outs[4] 
            excluded_stack_infos = outs[5]
            
            excluded_stack_infos = self.pre_processor.move_excluded_stacks(
                excluded_stack_mag_nii_paths=excluded_stack_mag_nii_paths,
                excluded_stack_pha_nii_paths=excluded_stack_pha_nii_paths,
                excluded_stack_infos=excluded_stack_infos,
            )
            
    # Validation
    if stack_mag_nii_paths is None:
        _clean_up()
        return
        
    # Print and save stack information
    filename = f"{file_prefix}_stacks_qc" if file_prefix is not None else "stacks_qc"
    report_path = os.path.join(stacks_nii_dir, f"{filename}.json")
    self.pre_processor.print_stack_infos(
        stack_infos=stack_infos,
    )
    self.pre_processor.save_stack_infos(
        stack_infos=stack_infos, 
        excluded_stack_infos=excluded_stack_infos,
        output_path=report_path,
    )
    
    # Sort stacks
    stack_mag_nii_paths, stack_pha_nii_paths, stack_infos = self.chest_segmentator.sort_stacks(
        stack_mag_nii_paths=stack_mag_nii_paths,
        stack_pha_nii_paths=stack_pha_nii_paths,
        stack_infos=stack_infos,
    )
    
    # Select 1st stack
    target_stack_nii_path = stack_mag_nii_paths[0]
    chest_mask_idx = 0
    
    # Run chest segmentation
    chest_mask_path = None
    chest_mask_qc_path = None
    chest_mask_valid = False
    if self.mode != "manual":
        chest_mask_path, chest_mask_qc_path = self.chest_segmentator.run(
            stack_nii_path=target_stack_nii_path,
            output_dir=chest_mask_dir,
            profile=profile,
            debug=debug,
        )
        if self.mode == "monitored_auto":
            chest_mask_valid = self.chest_segmentator.validate_mask(
                qc_report_path=chest_mask_qc_path,
            )
    if (self.mode in ("manual", "semi_auto")
        or (self.mode == "monitored_auto" and not chest_mask_valid)):
        with self.excluded_from_timer():
            chest_mask_path, chest_mask_idx = self.chest_segmentation_gui(
                stack_nii_paths=stack_mag_nii_paths,
                chest_mask_path=chest_mask_path,
                stack_infos=stack_infos,
                output_dir=chest_mask_dir,
                profile=profile,
                debug=debug,
            )
       
    # Validation
    if chest_mask_path is not None and os.path.exists(chest_mask_path): 
        stack_infos[chest_mask_idx]["chest_mask"] = chest_mask_path
    else:
        _clean_up()
        return
    
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
    
    # Run heart segmentation
    heart_mask_path = None
    heart_mask_qc_path = None
    heart_mask_valid = False
    if self.heart_segmentator.activate:
        if self.mode != "manual":
            heart_mask_path, heart_mask_qc_path = self.heart_segmentator.run(
                cine_nii_path=cine_nii_path,
                output_dir=heart_mask_dir,
                profile=profile,
                debug=debug,
            )
            if self.mode == "monitored_auto":
                heart_mask_valid = self.heart_segmentator.validate_mask(
                    qc_report_path=heart_mask_qc_path,
                )
        if (self.heart_segmentator.activate_gui 
            and (self.mode in ("manual", "semi_auto")
            or (self.mode == "monitored_auto" and not heart_mask_valid))):
            with self.excluded_from_timer():
                heart_mask_path = self.heart_segmentation_gui(
                    cine_nii_path=cine_nii_path,
                    heart_mask_path=heart_mask_path,
                    output_dir=heart_mask_dir,
                )
        
        # Validation
        if heart_mask_path is None or not os.path.exists(heart_mask_path):
            _clean_up()
            return
        
    # Run reorientation network
    outs = self.reorientor.run(
        cine_nii_path=cine_nii_path,
        heart_mask_path=heart_mask_path,
        stack_nii_paths=stack_mag_nii_paths+[chest_mask_path],
        output_dir=cine_nii_dir,
        mode=mode,
        profile=profile,
        debug=debug,
    )
    reo_cine_nii_path = outs[0]
    ctr_cine_nii_path = outs[1]
    ctr_heart_mask_path = outs[2]
    aff_path = outs[3]
    reo_qc_report_path = outs[4]
    
    reo_valid = False
    if self.mode == "monitored_auto":
        reo_valid = self.reorientor.validate_reo(
            qc_report_path=reo_qc_report_path,
        )
        
    if (self.mode in ("manual", "semi_auto")
        or (self.mode == "monitored_auto" and not reo_valid)):
        with self.excluded_from_timer():
            reo_cine_nii_path = self.reorientation_gui(
                reo_cine_nii_path=reo_cine_nii_path,
                ctr_cine_nii_path=ctr_cine_nii_path,
                ctr_heart_mask_path=ctr_heart_mask_path,
                aff_path=aff_path,
                stack_nii_paths=stack_mag_nii_paths+[chest_mask_path],
                output_dir=cine_nii_dir,
            )
        
    # Validation
    if reo_cine_nii_path is None or not os.path.exists(reo_cine_nii_path):
        delete_paths = [ctr_cine_nii_path, ctr_heart_mask_path] if not debug else None
        _clean_up(delete_paths=delete_paths)
        return
    
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
    
    delete_paths = [ctr_cine_nii_path, ctr_heart_mask_path] if not debug else None
    _clean_up(delete_paths=delete_paths)