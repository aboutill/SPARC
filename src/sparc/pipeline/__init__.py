from sparc.pipeline.preprocessing import PreProcessor
from sparc.pipeline.chest_segmentation import ChestSegmentator
from sparc.pipeline.svr import SliceVolumeReconstructor
from sparc.pipeline.heart_segmentation import HeartSegmentator
from sparc.pipeline.reorientation import Reorientor
from sparc.pipeline.postprocessing import PostProcessor


class Pipeline:
    """Build together every SPARC pipeline stage."""
    
    from ._io import (
        setup_logging, 
        print_pipeline_info,
        save_pipeline_info,
        save_qc_report,
    )
    from ._run import (
        excluded_from_timer,
        run,
    )
    from ._gui import (
        _check_file,
        stack_review_gui,
        chest_segmentation_gui,
        heart_segmentation_gui,
        reorientation_gui,
    )
    
    def __init__(
            self,
            preprocessing_cfg,
            chest_segmentation_cfg,
            svr_cfg,
            heart_segmentation_cfg,
            reorientation_cfg,
            postprocessing_cfg,
            models_chest_seg_dir=None,
            models_chest_seg_cfg_path=None,
            models_heart_seg_dir=None,
            models_heart_seg_cfg_path=None,
            models_reo_dir=None,
            models_reo_cfg_path=None,
        ):
        """Build every stage object from its config section."""

        self.preprocessing_cfg = preprocessing_cfg
        self.chest_segmentation_cfg = chest_segmentation_cfg
        self.svr_cfg = svr_cfg
        self.heart_segmentation_cfg = heart_segmentation_cfg
        self.reorientation_cfg = reorientation_cfg
        self.postprocessing_cfg = postprocessing_cfg
        
        self.pre_processor = PreProcessor(**self.preprocessing_cfg)
        self.chest_segmentator = ChestSegmentator(
            models_dir=models_chest_seg_dir,
            models_cfg_path=models_chest_seg_cfg_path,
            **self.chest_segmentation_cfg,
        )
        self.svr_reconstructor = SliceVolumeReconstructor(**self.svr_cfg)
        self.heart_segmentator = HeartSegmentator(
            models_dir=models_heart_seg_dir,
            models_cfg_path=models_heart_seg_cfg_path,
            **self.heart_segmentation_cfg,
        )
        self.reorientor = Reorientor(
            models_dir=models_reo_dir,
            models_cfg_path=models_reo_cfg_path,
            **self.reorientation_cfg,
        )
        self.post_processor = PostProcessor(**self.postprocessing_cfg)