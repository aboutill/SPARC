import os
import torch
import torch.optim as optim
import numpy as np

from monai.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm


def train(
        self, 
        train_ds, 
        val_ds, 
        learning_rate=1e-3, 
        batch_size=64, 
        epochs=200, 
        rotate_range=(0.1,0.1,0.1),
        translate_range=(10.0,10.0,10.0),
        prob_roi_mask=0.2,
        prob_background_mask=0.2,
        prob_dilate=0.8,
        prob_erode=0.8,
        dilate_range=(1,10),
        erode_range=(1,5),
        weight_geodesic_loss=1.0,
        weight_translation_loss=0.01,
        weight_img_loss=1.0,
        weight_point_loss=0.01,
        workers=8,
    ):
    """Train for epochs, logging losses to TensorBoard and saving the 
    best-validation-loss checkpoint."""
    
    torch.backends.cudnn.benchmark = True
    
    # Initialize data loaders
    train_loader = DataLoader(
        dataset=train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers,
    )
    val_loader = DataLoader(
        dataset=val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=workers,
    )
    
    # Initialize optimizer and loss function
    self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
    
    # Set parameters
    self.epochs = epochs
    self.batch_size = batch_size
    self.r_range = rotate_range
    self.t_range = translate_range
    self.prob_roi_mask = prob_roi_mask
    self.prob_bg_mask = prob_background_mask
    self.prob_dilate = prob_dilate
    self.prob_erode = prob_erode
    self.dilate_range = dilate_range
    self.erode_range = erode_range
    self.w_geodesic_loss = weight_geodesic_loss
    self.w_translation_loss = weight_translation_loss
    self.w_img_loss = weight_img_loss
    self.w_point_loss = weight_point_loss
    self.global_best = np.inf
    
    # Set summary writers
    self.writer_train = SummaryWriter(self.log_dir)
    self.writer_val = SummaryWriter(self.log_dir)
    
    # Iter over epochs
    for epoch in range(epochs):
        # Train and validation
        self.epoch = epoch
        self._train_epoch(train_loader)
        self._val_epoch(val_loader)
        

def _train_epoch(self, train_loader):
    """Run one training epoch and log its losses."""
    
    # Run epoch
    avg_losses = self._run_epoch(train_loader, train=True)

    # Log to TensorBoard
    self.writer_train.add_scalars(
        "train-losses",
        {
            "total-loss": avg_losses["total"],
            "image-loss": avg_losses["image"],
            "point-loss": avg_losses["point"],
            "geodesic-loss": avg_losses["geodesic"],
            "translation-loss": avg_losses["translation"],
        }, 
        self.epoch,
    )


def _val_epoch(self, val_loader):
    """Run one validation epoch and log its losses."""
    
    # Run epoch
    avg_losses = self._run_epoch(val_loader, train=False)

    # Log to TensorBoard
    self.writer_val.add_scalars(
        "val-losses", 
        {
            "total-loss": avg_losses["total"],
            "image-loss": avg_losses["image"],
            "point-loss": avg_losses["point"],
            "geodesic-loss": avg_losses["geodesic"],
            "translation-loss": avg_losses["translation"],
            },
        self.epoch,
    )

        
def _run_epoch(self, loader, train=True):
    """Run one epoch (train or eval): simulate a random pose, predict
    anchor points, compute losses, and (if training) backpropagate."""
    
    # Set training/evaluation mode
    self.net.train() if train else self.net.eval()
    
    # Loss accumulators
    running_losses = {
        "total": 0.0,
        "image": 0.0,
        "point": 0.0,
        "geodesic": 0.0,
        "translation": 0.0,
    }
    
    # Parameters
    step = 0
    context = torch.enable_grad() if train else torch.no_grad()
    desc = "Training" if train else "Validation"
    
    with context:
        # Progress bar
        epoch_desc = f"{desc} (epoch {self.epoch+1}/{self.epochs})"
        epoch_iterator = tqdm(loader, desc=epoch_desc, dynamic_ncols=True)
    
        # Iter over batchs
        for i, batch in enumerate(epoch_iterator):
            
            # Load image and mask
            img = batch["image"].to(self.device, non_blocking=True)
            if "label" in batch.keys():
                mask = batch["label"].to(self.device, non_blocking=True)
            else:
                mask = None
                
            # Mask transformation
            img_masked = self.apply_mask(
                img=img,
                roi_mask=mask, 
                prob_roi_mask=self.prob_roi_mask,
                prob_bg_mask=self.prob_bg_mask,
                prob_dilate=self.prob_dilate,
                prob_erode=self.prob_erode,
                dilate_range=self.dilate_range,
                erode_range=self.erode_range,
            )

            # Initialize affine matrix and points
            affine_matrix, _ = self.init_affine_matrix(
                n=self.batch_size,
                r_range=self.r_range,
                t_range=self.t_range,
            )
            inv_affine_matrix = self.invert_affine_matrix(affine_matrix)
            points = self.matrix_to_points(inv_affine_matrix)
            
            # Apply transform on image
            img_masked_def, img_masked = self.apply_transform(img_masked, affine_matrix)
            img_def, img = self.apply_transform(img, affine_matrix)
            
            # Predict points  
            points_pred, _ = self.net(img_masked_def)
                    
            # Points to affine
            affine_matrix_pred = self.points_to_matrix(points_pred)
                    
            # Transform image
            img_pred, _ = self.apply_transform(img_def, affine_matrix_pred)
    
            # Extract roation matrices and translation vectors
            R_gt = inv_affine_matrix[:, :3, :3]
            R_pred = affine_matrix_pred[:, :3, :3]
            t_gt = inv_affine_matrix[:, :3, 3]
            t_pred = affine_matrix_pred[:, :3, 3]
            
            # Compute losses
            img_loss = self.image_distance(img, img_pred)
            point_loss = self.euclidean_distance(points, points_pred)
            geodesic_loss = self.geodesic_distance(R_gt, R_pred)
            translation_loss = self.translation_distance(t_gt, t_pred)
            
            # Total loss
            total_loss = (
                self.w_img_loss * img_loss 
                + self.w_point_loss * point_loss 
                + self.w_geodesic_loss * geodesic_loss
                + self.w_translation_loss * translation_loss
            )
            
            # Back-propagation
            if train:
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            step += 1
            
            # Track running losses
            running_losses["total"] += total_loss.item()
            running_losses["image"] += img_loss.item()
            running_losses["point"] += point_loss.item()
            running_losses["geodesic"] += geodesic_loss.item()
            running_losses["translation"] += translation_loss.item()
 
            # Update progress bar
            epoch_iterator.set_postfix(loss=f"{total_loss.item():.4f}")
            
        # Compute epoch losses
        avg_losses = {k: v / step for k, v in running_losses.items()}
        
        # Save model best validation loss
        if not train and avg_losses["total"] < self.global_best:
            dirname = os.path.dirname(self.model_path)
            os.makedirs(dirname, exist_ok=True)
            torch.save(
                self.net.state_dict(), 
                self.model_path,
            )
            self.global_best = avg_losses["total"]
            
        if not train:
            self._save_batch_images(batch, img, img_masked_def, img_pred)

    return avg_losses


def _save_batch_images(self, batch, img, img_def, img_pred):
    """Periodically save ground-truth/deformed/predicted images from
    the last validation batch for visual inspection."""

    save_interval = max(self.epochs // 10, 1)
    if self.epoch == 0 or self.epoch % save_interval != 0:
        return

    # Convert to numpy
    imgs = {
        "gt": img.detach().cpu().numpy(),
        "def": img_def.detach().cpu().numpy(),
        "pred": img_pred.detach().cpu().numpy(),
    }

    batch_info = batch["image_meta_dict"]["filename_or_obj"]
    file_ext = ".nii.gz"

    for j in range(len(batch_info)):
        file_path = batch_info[j]
        dirname, filename = os.path.dirname(file_path), os.path.basename(file_path)

        # Replace data_root_dir with log_dir/imgs
        folder = dirname.replace(self.data_root_dir, os.path.join(self.log_dir, "imgs"))
        os.makedirs(folder, exist_ok=True)

        for key, arr in imgs.items():
            out_filename = filename.replace(file_ext, f"_{self.epoch}_{key}{file_ext}")
            out_path = os.path.join(folder, out_filename)

            # Save single image
            self.save_img(arr[j].squeeze(), out_path)