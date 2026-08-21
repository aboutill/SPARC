import os
import torch

import numpy as np
import pandas as pd

from monai.data import DataLoader


def validate(
        self,
        val_ds, 
        num_val=5,
        rotate_range=None,
        translate_range=None,
        workers=8,
    ):
    """Load the best checkpoint and run repeated random-pose
    validation passes, saving raw and summary metrics."""
    
    # Load weights
    if os.path.exists(self.model_path):
        self.net.load_state_dict(torch.load(self.model_path))
    
    # Initialize data loader
    val_loader = DataLoader(
        dataset=val_ds, 
        batch_size=1,
        shuffle=False, 
        num_workers=workers,
    )
    
    # Define metrics
    metrics_str = [
        "geodesic-distance",
        "translation-distance",
        "normalised-mutual-information",
        "peak-signal-to-noise-ratio",
    ]
    
    # Raw metrics dataframe
    metrics_raw_df = {
        "filename": [],
        "angle": [],
        "translation": [],
        "step": [],
    }
    metrics_raw_df.update({metric_str: [] for metric_str in metrics_str})

    for val_step in range(num_val):
        self._validate_step(
            val_loader=val_loader,
            val_step=val_step,
            rotate_range=rotate_range,
            translate_range=translate_range,
            metrics_raw_df=metrics_raw_df,
            prediction_dir=self.prediction_dir,
        )
        
    # Save metrics
    os.makedirs(self.metrics_dir, exist_ok=True)
    metrics_raw_df = pd.DataFrame(metrics_raw_df)
    metrics_raw_df.to_csv(
        os.path.join(self.metrics_dir, "metrics_raw.csv"), index=False
    ) 
    
    # Metrics dataframe
    metrics_df = {metric_str: [] for metric_str in metrics_str}
    
    # Update dataframe
    for metric_str in metrics_str:
        col = metrics_raw_df[metric_str]
        metrics_df[metric_str].append(np.mean(col))
        
    # Save metrics
    metrics_df = pd.DataFrame(metrics_df)
    metrics_df.to_csv(
        os.path.join(self.metrics_dir, "metrics.csv"), index=False
    )  
    
    
def _validate_step(
        self, 
        val_loader, 
        val_step,
        metrics_raw_df,
        prediction_dir, 
        rotate_range=None,
        translate_range=None,
    ):
    """Run one validation pass over the full validation set and save
    predicted/deformed images."""
    
    # Set validation mode
    self.net.eval()
    
    # Iter over batches
    with torch.no_grad():
        for batch in val_loader:
            
            # Load image and mask
            img = batch["image"].to(self.device, non_blocking=True)
            if "label" in batch.keys():
                mask = batch["label"].to(self.device, non_blocking=True)
            else:
                mask = None
            file_path = batch["image_meta_dict"]["filename_or_obj"][0]
            
            # Validate batch
            imgs = self._validate_batch(
                img=img,
                mask=mask,
                rotate_range=rotate_range,
                translate_range=translate_range,
                file_path=file_path,
                val_step=val_step,
                metrics_raw_df=metrics_raw_df,
            )
            
            #
            dirname = os.path.dirname(file_path)
            file_ext = ".nii.gz"
            
            # Define folder for predictions
            folder = dirname.replace(self.data_root_dir, prediction_dir)
            os.makedirs(folder, exist_ok=True)
            
            # Save images
            for key, arr in imgs.items():
                out_name = f"{key}_{val_step}{file_ext}"
                out_path = os.path.join(folder, out_name)
                if arr is not None:
                    self.save_img(arr, out_path)
                
            
def _validate_batch(
        self,
        img,
        mask,
        file_path,
        val_step,
        metrics_raw_df,
        rotate_range=None,
        translate_range=None,
    ):
    """Simulate a random pose for one batch, predict its inverse,
    compute metrics, and return images for saving."""
    
    # Initialize inverse affine matrix and points
    affine_matrix, (angle, t) = self.init_affine_matrix(
        n=1, 
        r_range=rotate_range,
        t_range=translate_range,
    )
    inv_affine_matrix = self.invert_affine_matrix(affine_matrix)

    # Apply ground-truth transform
    img_def, _ = self.apply_transform(img, affine_matrix)
    mask_def, _ = self.apply_transform(mask, affine_matrix, mode="nearest")

    # Predict points and reconstruct affine
    points_pred, _ = self.net(img_def)
    affine_matrix_pred = self.points_to_matrix(points_pred)

    # Apply predicted transform
    img_pred, _ = self.apply_transform(img_def, affine_matrix_pred)
    mask_pred, _ = self.apply_transform(mask_def, affine_matrix_pred, mode="nearest")

    # Rotation matrices
    R_gt, R_pred = inv_affine_matrix[:, :3, :3], affine_matrix_pred[:, :3, :3]
    t_gt, t_pred = inv_affine_matrix[:, :3, 3], affine_matrix_pred[:, :3, 3]

    # Convert tensors to numpy
    img_gt = img.detach().cpu().numpy().squeeze()
    img_pred = img_pred.detach().cpu().numpy().squeeze()
    if mask is not None:
        mask_gt = mask.detach().cpu().numpy().squeeze().astype(bool)
    else:
        mask_gt = None
    
    # Metrics
    gd = self.geodesic_distance(R_gt, R_pred)
    td = self.translation_distance(t_gt, t_pred)
    nmi = self.normalised_mutual_information(img_gt, img_pred, mask_gt)
    psnr = self.peak_signal_to_noise_ratio(img_gt, img_pred, mask_gt)
    
    # Record metrics
    metrics_raw_df["filename"].append(file_path)
    metrics_raw_df["angle"].append(angle[0])
    metrics_raw_df["translation"].append(np.linalg.norm(t))
    metrics_raw_df["geodesic-distance"].append(gd.item())
    metrics_raw_df["translation-distance"].append(td.item())
    metrics_raw_df["normalised-mutual-information"].append(nmi)
    metrics_raw_df["peak-signal-to-noise-ratio"].append(psnr)
    metrics_raw_df["step"].append(val_step)
    
    # Convert tensors to numpy
    if mask is not None:
        mask_gt = mask_gt.astype(np.uint8)
        mask_def = mask_def.detach().cpu().numpy().squeeze()
        mask_pred = mask_pred.detach().cpu().numpy().squeeze()
    img_def = img_def.detach().cpu().numpy().squeeze()
    
    #
    imgs = {
        "img_gt": img_gt,
        "mask_gt": mask_gt,
        "img_def": img_def,
        "mask_def": mask_def,
        "img_pred": img_pred,
        "mask_pred": mask_pred,
    }
    
    return imgs