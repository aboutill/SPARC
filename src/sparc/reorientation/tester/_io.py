import os
import glob
import json
import logging

import torch
import numpy as np
import pandas as pd


from sparc.utils.logging import setup_logging_config
from sparc.reorientation.model import Model


@staticmethod
def setup_logging(log_path=None, verbose=False):
    """Configure logging for this run."""
    setup_logging_config(
        log_path=log_path,
        verbose=verbose,
    )
    
    
def print_model_info(self):
    """Print model settings."""
    logging.info("Reorientation model testing configurations.")
    cfgs = {
        "Input data": self.data_cfg,
        "Image transformation": self.transforms_cfg,
        "ViT architecture": self.vit_cfg,
        "Testing parameters": self.test_cfg,
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
        "vit": self.vit_cfg,
        "test": self.test_cfg,
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
        

def init_output_dir_layout(self, output_dir, data_root_dir):
    """Set up ensemble output folders: prediction/, metrics/, quality_control/."""
    self.output_dir = output_dir
    ensemble_dir = os.path.join(output_dir, "ensemble")
    self.prediction_dir = os.path.join(ensemble_dir, "prediction")
    self.metrics_dir = os.path.join(ensemble_dir, "metrics")
    self.qc_dir = os.path.join(ensemble_dir, "quality_control")
    self.data_root_dir = data_root_dir


def load_ensemble(self, models_dir):
    """Load every *.pth checkpoint in models_dir into self.models."""
    model_paths = sorted(glob.glob(os.path.join(models_dir, "*.pth")))
    if len(model_paths) < 2:
        raise ValueError("Ensemble inference requires at least two models.")

    self.models = []
    for model_path in model_paths:
        model = Model(
            roi_size=self.transforms_cfg["roi_size"],
            pixdim=self.transforms_cfg["pixdim"],
            **self.vit_cfg,
        )
        model.to_device(device=self.device)
        model.net.load_state_dict(torch.load(model_path), strict=False)
        self.models.append(model)
        
        
@staticmethod
def save_arrs(arrs, folder, test_step, model):
    """Save a dict of prediction arrays: 3D arrays as NIfTI, other
    arrays (affine matrices, anchor points) as plain text. Entries
    that are None are skipped."""
    for key, arr in arrs.items():
        if arr is None:
            continue
        if arr.ndim == 3:
            out_path = os.path.join(folder, f"{key}_{test_step}.nii.gz")
            model.save_img(arr, out_path)
        else:
            out_path = os.path.join(folder, f"{key}_{test_step}.txt")
            np.savetxt(out_path, arr)
            

@staticmethod
def save_metrics_csv(
        raw_df, 
        cols, 
        out_dir, 
        raw_name="metrics_raw.csv", 
        summary_name="metrics.csv",
    ):
    """Save a raw per-step metrics dict to CSV, plus a one-row
    summary CSV of column means."""
    os.makedirs(out_dir, exist_ok=True)
    raw_df = pd.DataFrame(raw_df)
    raw_df.to_csv(os.path.join(out_dir, raw_name), index=False)

    summary = {col: [np.mean(raw_df[col])] for col in cols}
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, summary_name), index=False)