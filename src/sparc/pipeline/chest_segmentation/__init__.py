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
        """Rank candidate stacks by z-smoothness (ascending, best first),
        excluding stacks that would be too small at the U-Net's target
        resolution or (if configured) don't match the required
        orientation. Excluded stacks are sorted last.
        """
        strides = self.unet_cfg["strides"]
        thresh = np.prod(strides)
        unet_pixdim = self.transforms_cfg["pixdim"]
    
        z_smooths = []
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
            z_smooths.append(z_smooth)
    
        idx = np.argsort(z_smooths)
    
        if not np.isfinite(z_smooths[idx[0]]):
            raise ValueError(
                "No candidate stack meets the size/orientation criteria "
                f"(thresh={thresh}, target_orientation={self.target_stack_orientation})."
            )
    
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