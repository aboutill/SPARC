import os
import yaml
import json


from sparc.segmentation.tester import EnsembleTester
from sparc.pipeline.heart_segmentation.models import (
    DEFAULT_MODELS_DIR, 
    DEFAULT_MODEL_CFG_PATH, 
)

class HeartSegmentator:
    """Automated heart segmentation stage can be optionally disabled."""

    from ._io import (
        print_model_info, 
        print_qc,
    )
    from ._run import (
        run,
        run_with_gui,
    )
    from ._gui import gui
    
    
    def __init__(
            self,
            activate=False,
            activate_gui=False,
            models=None,
            dice_qc_thresh=0.75,
            hd_qc_thresh=7.5,
            models_dir=None,
            models_cfg_path=None,
        ):
        """Store configuration and initialise ensemble tester."""
        
        self.activate = activate
        self.activate_gui = activate_gui
        self.models = models
        self.dice_qc_thresh = dice_qc_thresh
        self.hd_qc_thresh = hd_qc_thresh
        
        self.init_ensemble_tester(
            models_dir=models_dir,
            models_cfg_path=models_cfg_path,
        )

        
    def init_ensemble_tester(
            self,
            models_dir=None,
            models_cfg_path=None,
        ):
        """Load model configuration and build the ensemble tester.

        Either provide both models_dir and models_cfg_path (a
        custom/fine-tuned checkpoint set), or leave both as None to
        use the default pretrained models named by self.models.
        """

        if models_dir is not None and models_cfg_path is not None:
            self.models = "user-provided"
        elif models_dir is None and models_cfg_path is None:
            if self.models is None:
                self.models = "siemens_transfer"
            models_dir = os.path.join(DEFAULT_MODELS_DIR, self.models)
            models_cfg_path = DEFAULT_MODEL_CFG_PATH
            if not os.path.isdir(models_dir):
                raise ValueError(
                    f"Default heart segmentation models directory not "
                    f"found for models='{self.models}': {models_dir}"
                )
        else:
            raise ValueError(
                "models_dir and models_cfg_path must both be provided "
                "together, or both left as None to use default "
                "pretrained models."
            )

        if not os.path.isfile(models_cfg_path):
            raise ValueError(f"Configuration file not found: {models_cfg_path}")
        
        cfg = yaml.safe_load(open(models_cfg_path))
        self.unet_cfg = cfg["unet"]
        self.inferer_cfg = cfg["inferer"]
        self.transforms_cfg = cfg["transforms"] 
        self.post_processing_cfg = cfg["post_processing"] 
        self.models_dir = models_dir
        
        self.ensemble_tester = EnsembleTester(
            transforms_cfg=self.transforms_cfg,
            unet_cfg=self.unet_cfg,
            inferer_cfg=self.inferer_cfg,
            post_processing_cfg=self.post_processing_cfg,
        )
        
    def validate_mask(
            self, 
            qc_report_path,
        ):
        """Return True if a segmentation QC report meets both the
        Dice and Hausdorff-distance acceptance thresholds."""
        
        with open(qc_report_path) as f:
            report = json.load(f)
            
        dice_qc = float(report['DICE_QC']['mean'])
        hd_qc = float(report['HD_QC']['mean'])
        
        valid = dice_qc > self.dice_qc_thresh and hd_qc < self.hd_qc_thresh
        return valid