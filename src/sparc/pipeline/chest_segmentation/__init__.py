import os
import yaml
import json

import numpy as np

from sparc.segmentation.tester import EnsembleTester
from sparc.pipeline.chest_segmentation.models import (
    DEFAULT_MODELS_DIR, 
    DEFAULT_MODEL_CFG_PATH, 
)

class ChestSegmentator:
    """Automated chest segmentation stage: template stack
    selection and ensemble U-Net inference over the selected template."""

    from ._io import (
        print_model_info, 
        print_qc,
    )
    from ._run import run 
    
    def __init__(
            self,
            target_stack_orientation=None,
            models=None,
            dil_rad=2,
            dice_qc_thresh=0.85,
            hd_qc_thresh=5.0,
            models_dir=None,
            models_cfg_path=None,
        ):
        """Store configuration and initialise ensemble tester."""
        
        self.models = models
        self.target_stack_orientation = target_stack_orientation
        self.dil_rad = dil_rad
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
                    f"Default chest segmentation models directory not "
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
            post_processing_cfg=self.post_processing_cfg
        )
        
        
    def sort_stacks(
            self,
            stack_mag_nii_paths,
            stack_pha_nii_paths,
            stack_infos,
        ):
        """Rank all candidate stacks in one combined priority order:
        correctly-oriented, adequately-sized stacks first (best-to-worst
        by z-smoothness), then any remaining wrong-orientation,
        adequately-sized stacks as a fallback tier (also best-to-worst),
        then too-small stacks last (order among these doesn't matter --
        they are never selected downstream; see unet_too_small).
        """
        strides = self.unet_cfg["strides"]
        thresh = np.prod(strides)
        unet_pixdim = self.transforms_cfg["pixdim"]
        
        z_smooth_raws = []
        groups = []
        for stack_info in stack_infos:
        
            dim = stack_info["dim"]
            img_pixdim = stack_info["pixdim"]
            unet_dim = [
                int(img_pixdim[axis] * dim[axis] / unet_pixdim[axis])
                for axis in range(len(unet_pixdim))
            ]
        
            too_small = any(2*d < thresh for d in unet_dim)
            wrong_orientation = (
                self.target_stack_orientation is not None
                and self.target_stack_orientation != stack_info["ornt"]
            )
        
            true_z_smooth = stack_info.get("z_smooth_raw", stack_info["z_smooth"])
            stack_info["z_smooth_raw"] = true_z_smooth
        
            z_smooth = float("inf") if (too_small or wrong_orientation) else true_z_smooth
            stack_info["z_smooth"] = z_smooth
            stack_info["unet_too_small"] = too_small
        
            z_smooth_raws.append(true_z_smooth)
        
            if too_small:
                group = 2
            elif wrong_orientation:
                group = 1
            else:
                group = 0
            groups.append(group)
        
        # group is primary, z_smooth_raw is secondary (breaks ties WITHIN
        # each group).
        idx = np.lexsort((z_smooth_raws, groups))
        
        stack_mag_nii_paths = [stack_mag_nii_paths[i] for i in idx]
        stack_pha_nii_paths = [stack_pha_nii_paths[i] for i in idx]
        stack_infos = [stack_infos[i] for i in idx]
        
        return stack_mag_nii_paths, stack_pha_nii_paths, stack_infos
    
    
    def validate_mask(
            self, 
            qc_report_path,
        ):
        """Return True if a segmentation QC report meets both the
        Dice and Hausdorff-distance acceptance thresholds."""
        
        with open(qc_report_path) as f:
            report = json.load(f)
            
        dice_qc = float(report["DICE_QC"]["mean"])
        hd_qc = float(report["HD_QC"]["mean"])
        
        valid = dice_qc > self.dice_qc_thresh and hd_qc < self.hd_qc_thresh
        return valid