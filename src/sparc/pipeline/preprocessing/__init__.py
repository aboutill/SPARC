        
class PreProcessor:
    
    """Preprocess a subject's raw DICOM cine stacks into NIfTI volumes."""
    
    from ._run import (
        run,
        run_with_gui,
        compute_rr_interval,
    )
    from ._dcm import (
        sort_dcm_series,
        read_rr_intervals_from_dcm_series,
        read_id_from_dcm,
        read_acquisition_matrix_from_dcm,
        read_slice_thickness_from_dcm,
    )
    from ._nii import (
        dcm2nii,
        remove_zero_filling_funct,
        denoise_funct,
        degibbs_funct,
        N4_corr_funct,
        get_nii_header,
        set_nii_time_res_header,
        get_nii_orientation,
        get_z_smooth,
    )
    from ._io import (
        print_model_info,
        print_stack_infos,
        move_excluded_stacks,
        save_stack_infos,
    )
    from ._gui import stack_review_gui

    
    def __init__(
            self,
            img_type_dcm_tag=None,
            mag_dcm_flag=None,
            pha_dcm_flag=None,
            rr_interval_dcm_tag=None,
            stack_id_dcm_tag=None,
            acq_mat_dcm_tag=None,
            slice_thickness_dcm_tag=None,
            remove_zero_filling=False,
            denoise=False,
            degibbs=False,
            N4_corr=False,
        ):
        """Configure preprocessing behaviour for a subject's DICOM input."""
        
        # DICOM tags
        self.img_type_dcm_tag = img_type_dcm_tag
        self.mag_dcm_flag = mag_dcm_flag
        self.pha_dcm_flag = pha_dcm_flag
        self.rr_interval_dcm_tag = rr_interval_dcm_tag
        self.stack_id_dcm_tag = stack_id_dcm_tag
        self.acq_mat_dcm_tag = acq_mat_dcm_tag
        self.slice_thickness_dcm_tag = slice_thickness_dcm_tag
        
        # Image processing
        self.remove_zero_filling = remove_zero_filling
        self.denoise = denoise
        self.degibbs = degibbs
        self.N4_corr = N4_corr