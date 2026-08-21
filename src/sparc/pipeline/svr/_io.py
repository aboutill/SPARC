import logging


def print_model_info(self):
    """Log the configured SVR reconstruction and post-processing
    parameters."""
    
    logging.info("SVR parameters:")
    logging.info(f"  Resolution: {self.resolution} [mm]")
    logging.info(f"  Registration iterations: {self.iterations}")
    logging.info(f"  Reconstruction iteration: {self.sr_iterations}")
    logging.info(f"  Smoothing parameter: {self.lambda_smooth}")
    logging.info(f"  Edge parameter: {self.delta_edge}")
    
    if self.no_stack_zcrop:
        logging.info("  Disable cropping of stacks in the throung plane direction")
    if self.mask_slices_not_svr:
        logging.info("  Mask slices everywhere except Slice-to-Volume registration")
    
    if not self.no_robust_statistics:
        logging.info("  Robust statistics activated")
    if not self.no_intensity_matching:
        logging.info("  Intensity matching activated")
        
    logging.info("SVR post-processing:")
    if self.N4_corr:
        logging.info("  Bias-field correction activated")
    if self.smooth_roi_mask:
        logging.info("  Thoracic ROI mask smoothing activated")