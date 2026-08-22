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
    validation passes, saving raw and summary metrics.
    """
    
    # Load weights
    if os.path.exists(self.model_path):
        self.net.load_state_dict(torch.load(self.model_path))
    
    # Initialize data loader
    val_loader = DataLoader(
        dataset=val_ds, 
        batch_size=len(val_ds),
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
    
    for metric_str in metrics_str:
        col = metrics_raw_df[metric_str]
        metrics_df[metric_str].append(np.mean(col))
        
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
    predicted/deformed images, one output set per sample per batch."""
    
    self.net.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            
            img = batch["image"].to(self.device, non_blocking=True)
            if "label" in batch.keys():
                mask = batch["label"].to(self.device, non_blocking=True)
            else:
                mask = None
            file_paths = batch["image_meta_dict"]["filename_or_obj"]
            
            per_sample_imgs = self._validate_batch(
                img=img,
                mask=mask,
                rotate_range=rotate_range,
                translate_range=translate_range,
                file_paths=file_paths,
                val_step=val_step,
                metrics_raw_df=metrics_raw_df,
            )
            
            file_ext = ".nii.gz"
            for file_path, imgs in zip(file_paths, per_sample_imgs):
                dirname = os.path.dirname(file_path)
                folder = dirname.replace(self.data_root_dir, prediction_dir)
                os.makedirs(folder, exist_ok=True)

                for key, arr in imgs.items():
                    out_name = f"{key}_{val_step}{file_ext}"
                    out_path = os.path.join(folder, out_name)
                    if arr is not None:
                        self.save_img(arr, out_path)
                
            
def _validate_batch(
        self,
        img,
        mask,
        file_paths,
        val_step,
        metrics_raw_df,
        rotate_range=None,
        translate_range=None,
    ):
    """Simulate an independent random pose per sample in the batch,
    predict its inverse, compute metrics, and return a list of
    per-sample image dicts (one per sample) for saving.
    """
    batch_size = img.shape[0]

    affine_matrix, (angle, t) = self.init_affine_matrix(
        n=batch_size, 
        r_range=rotate_range,
        t_range=translate_range,
    )
    inv_affine_matrix = self.invert_affine_matrix(affine_matrix)

    img_def, _ = self.apply_transform(img, affine_matrix)
    mask_def, _ = self.apply_transform(mask, affine_matrix, mode="nearest")

    points_pred, _ = self.net(img_def)
    affine_matrix_pred = self.points_to_matrix(points_pred)

    img_pred, _ = self.apply_transform(img_def, affine_matrix_pred)
    mask_pred, _ = self.apply_transform(mask_def, affine_matrix_pred, mode="nearest")

    R_gt, R_pred = inv_affine_matrix[:, :3, :3], affine_matrix_pred[:, :3, :3]
    t_gt, t_pred = inv_affine_matrix[:, :3, 3], affine_matrix_pred[:, :3, 3]

    # Batch-vectorized
    gd = self.geodesic_distance(R_gt, R_pred, reduction="none")
    td = self.translation_distance(t_gt, t_pred, reduction="none")

    img_gt_np = img.detach().cpu().numpy().squeeze(1)
    img_pred_np = img_pred.detach().cpu().numpy().squeeze(1)
    img_def_np = img_def.detach().cpu().numpy().squeeze(1)
    if mask is not None:
        mask_gt_np = mask.detach().cpu().numpy().squeeze(1).astype(bool)
        mask_def_np = mask_def.detach().cpu().numpy().squeeze(1)
        mask_pred_np = mask_pred.detach().cpu().numpy().squeeze(1)
    else:
        mask_gt_np = mask_def_np = mask_pred_np = None

    per_sample_imgs = []
    for i in range(batch_size):
        mask_gt_i = mask_gt_np[i] if mask_gt_np is not None else None

        nmi = self.normalised_mutual_information(img_gt_np[i], img_pred_np[i], mask_gt_i)
        psnr = self.peak_signal_to_noise_ratio(img_gt_np[i], img_pred_np[i], mask_gt_i)

        metrics_raw_df["filename"].append(file_paths[i])
        metrics_raw_df["angle"].append(angle[i])
        metrics_raw_df["translation"].append(np.linalg.norm(t[i]))
        metrics_raw_df["geodesic-distance"].append(gd[i].item())
        metrics_raw_df["translation-distance"].append(td[i].mean().item())
        metrics_raw_df["normalised-mutual-information"].append(nmi)
        metrics_raw_df["peak-signal-to-noise-ratio"].append(psnr)
        metrics_raw_df["step"].append(val_step)

        imgs = {
            "img_gt": img_gt_np[i],
            "mask_gt": mask_gt_i.astype(np.uint8) if mask_gt_i is not None else None,
            "img_def": img_def_np[i],
            "mask_def": mask_def_np[i] if mask_def_np is not None else None,
            "img_pred": img_pred_np[i],
            "mask_pred": mask_pred_np[i] if mask_pred_np is not None else None,
        }
        per_sample_imgs.append(imgs)

    return per_sample_imgs