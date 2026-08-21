import logging


def print_model_info(self):
    """Log the configured postprocessing parameters."""
    
    logging.info("Postprocessing parameters:")
    if self.save_dcm_private_tags:
        logging.info("  Saving DICOM private flags")
    if self.zip_dcm_files:
        logging.info("  Saving compressed DICOM output")
        
    logging.info("DICOM tags:")
    if self.ImplementationVersionName is not None:
        logging.info(f"  ImplementationVersionName: {self.ImplementationVersionName}")
    if self.ProtocolName is not None:
        logging.info(f"  ProtocolName: {self.ProtocolName}")
    if self.StudyDescription is not None:
        logging.info(f"  StudyDescription: {self.StudyDescription}")
    if self.SeriesDescription is not None:
        logging.info(f"  SeriesDescription: {self.SeriesDescription}")