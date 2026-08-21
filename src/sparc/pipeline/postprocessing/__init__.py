
class PostProcessor:
    """Converts a reconstructed cine NIfTI volume into a DICOM series,
    copying acquisition metadata from a source stack's DICOM header."""

    from ._io import (
        print_model_info, 
    )
    from ._run import (
        set_private_dcm_tags,
        nii2dcm,
        run,
    )
    
    def __init__(
            self,
            save_dcm_private_tags=False,
            zip_dcm_files=False,
            ImplementationVersionName=None,
            ProtocolName=None,
            StudyDescription=None,
            SeriesDescription=None,
        ):
        """Configure postprocessing behaviour."""
        
        # nii2dcm parameters
        self.save_dcm_private_tags = save_dcm_private_tags
        self.zip_dcm_files = zip_dcm_files
        
        # Dicom tags
        self.ImplementationVersionName = ImplementationVersionName
        self.ProtocolName = ProtocolName
        self.StudyDescription = StudyDescription
        self.SeriesDescription = SeriesDescription