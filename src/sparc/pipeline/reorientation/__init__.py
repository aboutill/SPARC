import os
import yaml
import json

import numpy as np

from sparc.reorientation.tester import EnsembleTester
from sparc.pipeline.reorientation.models import (
    DEFAULT_MODELS_DIR, 
    DEFAULT_MODEL_CFG_PATH, 
)


class Reorientor:
    """Automated reorientation stage."""

    from ._io import (
        print_model_info, 
        print_qc,
    )
    from ._run import (
        center_on_heart_mask,
        apply_affine_to_cine,
        apply_affine_to_mask,
        apply_affine_to_stacks_header,
        load_itksnap_affine,
        run,
        manual_run,
    )
    
    def __init__(
            self,
            models=None,
            rotation_avg="chordal",
            reo_stacks=False,
            gd_qc_thresh=25.0,
            cd_qc_thresh=5.0,
            models_dir=None,
            models_cfg_path=None,
        ):
        """Store configuration and initialise ensemble tester."""
        
        self.models = models
        self.rotation_avg = rotation_avg
        self.reo_stacks = reo_stacks
        self.gd_qc_thresh = gd_qc_thresh * np.pi / 180  #  to radians 
        self.cd_qc_thresh = cd_qc_thresh
        
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
                    f"Default reorientation models directory not "
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
        self.vit_cfg = cfg["vit"]
        self.transforms_cfg = cfg["transforms"] 
        self.models_dir = models_dir
        
        self.ensemble_tester = EnsembleTester(
            transforms_cfg=self.transforms_cfg,
            vit_cfg=self.vit_cfg,
        )
        
        
    def validate_reo(
            self, 
            qc_report_path,
        ):
        """Return True if a reorientation QC report meets both the
        geodesic-distance and centre-distance acceptance thresholds."""
        
        with open(qc_report_path) as f:
            report = json.load(f)
            
        gd_qc = float(report['GD_QC']['mean'])
        cd_qc = float(report['CD_QC']['mean'])
        
        valid = gd_qc < self.gd_qc_thresh and cd_qc < self.cd_qc_thresh
        return valid