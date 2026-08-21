import os
import json
import glob
import logging

from sparc.utils.logging import setup_logging_config


@staticmethod
def setup_logging(log_path=None, verbose=False):
    """Setup logging."""
    setup_logging_config(
        log_path=log_path, 
        verbose=verbose,
    )
    
    
def print_model_info(self):
    """Print model settings."""
    logging.info("Segmentation model training configurations.")
    cfgs = {
        "Input data": self.data_cfg,
        "Image transformation": self.transforms_cfg,
        "UNet architecture": self.unet_cfg,
        "Training hyperparameters": self.train_cfg,
        "Inferer parameters": self.inferer_cfg,
        "Post-processing": self.post_processing_cfg,
    }
    for title, cfg in cfgs.items():
        if cfg is not None:
            logging.info(f"  {title}:")
            for k, v in cfg.items():
                logging.info(f"    {k}: {v}")
    
    
def save_model_info(self, datalists, output_path):
    """Save inputs and model parameters to a JSON file."""
    cfgs = {
        "data": self.data_cfg,
        "transforms": self.transforms_cfg,
        "unet": self.unet_cfg,
        "train": self.train_cfg,
        "inferer": self.inferer_cfg,
        "post_processing": self.post_processing_cfg,
    }
    report = {}
    for k, v in cfgs.items():
        if v is not None:
            report[k] = v
    report["datalists"] = datalists
    
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    

@staticmethod    
def get_models_paths(models_dir, folds):
    """Load every *.pth checkpoint in models_dir."""
        
    models_paths = None
    if models_dir is not None:
        reg_ex = os.path.join(models_dir, "*.pth")
        models_paths = sorted(glob.glob(reg_ex))
        # In case number of pretrained models < number of fine tune models
        if len(models_paths) < folds:
            models_paths *= folds
            models_paths = models_paths[:folds]

    return models_paths