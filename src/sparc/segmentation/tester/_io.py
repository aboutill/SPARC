import os
import glob
import json
import torch
import logging

import numpy as np
import nibabel as nib

from monai.data import FolderLayout

from sparc.utils.logging import setup_logging_config
from sparc.segmentation.model import Model


@staticmethod
def setup_logging(log_path=None, verbose=False):
    """Setup logging."""
    
    setup_logging_config(
        log_path=log_path, 
        verbose=verbose,
    )
    
    
def print_model_info(self):
    """Print model settings."""
    logging.info("Segmentation model testing configurations.")
    cfgs = {
        "Input data": self.data_cfg,
        "Image transformation": self.transforms_cfg,
        "UNet architecture": self.unet_cfg,
        "Inferer parameters": self.inferer_cfg,
        "Post-processing": self.post_processing_cfg,
    }
    for title, cfg in cfgs.items():
        if cfg is not None:
            logging.info(f"  {title}:")
            for k, v in cfg.items():
                logging.info(f"    {k}: {v}")
                
    
def save_model_info(self, datalist, output_path):
    """Save inputs and model parameters to a JSON file."""
    cfgs = {
        "data": self.data_cfg,
        "transforms": self.transforms_cfg,
        "unet": self.unet_cfg,
        "inferer": self.inferer_cfg,
        "post_processing": self.post_processing_cfg,
    }
    report = {}
    for k, v in cfgs.items():
        if v is not None:
            report[k] = v
    report["datalist"] = datalist
    
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    

def init_output_dir_layout(self, output_dir, data_root_dir=None):
    """Set up ensemble output folders: metrics/, quality_control/,
    prediction/.
    """
    
    # Set output folders
    self.metrics_dir = os.path.join(output_dir, "metrics")
    self.qc_dir = os.path.join(output_dir, "quality_control")
    self.prediction_dir = os.path.join(output_dir, "prediction")
    
    # Prediction folder layout
    self.prediction_layout = FolderLayout(
        output_dir=self.prediction_dir,
        postfix="pred",
        extension="nii.gz",
        makedirs=True,
        data_root_dir=data_root_dir,
    )
    
    
def load_ensemble(self, models_dir):
    """Load every *.pth checkpoint into self.models."""
    model_paths = sorted(glob.glob(os.path.join(models_dir, "*.pth")))
    if len(model_paths) < 2:
        raise ValueError("Ensemble inference requires at least two models.")

    self.models = []
    for model_path in model_paths:
        model = Model(**self.unet_cfg)
        model.init_inferer(**self.inferer_cfg)
        model.to_device(device=self.device)
        model.net.load_state_dict(torch.load(model_path), strict=False)
        self.models.append(model)
        
        
@staticmethod
def load_mask_as_tensor(mask_path):
    """Load a mask as a single-channel integer class label volume."""
    nii = nib.load(mask_path)
    data = nii.get_fdata()

    if data.ndim == 4:
        label_volume = np.argmax(data, axis=-1)
    else:
        label_volume = data

    label_volume = torch.from_numpy(label_volume.astype(np.int64))

    dim = nii.header["dim"][1:4]
    pixdim = nii.header["pixdim"][1:4]
    max_err = float(np.sqrt(sum((dim[k] * pixdim[k]) ** 2 for k in range(3))))

    return label_volume, max_err