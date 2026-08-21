import torch

from monai.networks.nets import ViT
from monai.networks.layers import AffineTransform

torch.multiprocessing.set_sharing_strategy('file_system')


class Model():
    """A single pose-estimation network: ViT architecture, device,
    and output-path bookkeeping."""
    
    # Import class methods
    from ._metrics import (
        image_distance,
        euclidean_distance,
        geodesic_distance,
        translation_distance,
        normalised_mutual_information,
        peak_signal_to_noise_ratio,
    )
    from ._transformations import (
        apply_mask,
        init_affine_matrix,
        invert_affine_matrix,
        apply_transform,
        points_to_matrix,
        matrix_to_points,
    )
    from ._train import (
        train, 
        _train_epoch,
        _val_epoch, 
        _run_epoch, 
        _save_batch_images,
    )
    from ._validate import (
        validate, 
        _validate_step, 
        _validate_batch,
    )
    from ._io import (
        init_output_dir,
        save_img,
    )

    def __init__(
            self, 
            in_channels,
            roi_size,
            patch_size,
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            proj_type="conv",
            pos_embed_type="sincos",
            out_channels=9,
            spatial_dims=3,
            pixdim=[1,1,1],
        ):
        """Build the ViT network from architecture hyperparameters."""
        
        # Initialize network architecture  
        self.net = ViT(
            in_channels=in_channels,
            img_size=roi_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            classification=True,
            num_classes=out_channels,
            spatial_dims=spatial_dims,
            post_activation=0,
        )
        
        # Image size and resolution
        self.img_size = roi_size
        self.pixdim = pixdim
        
        #
        self.affine_warp = AffineTransform(
            mode="bilinear",
            padding_mode="zeros",
            normalized=False,       
            align_corners=True,
            reverse_indexing=True,
            zero_centered=True,
        )
        
        # Setup CUDA device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
    def to_device(self, device):
        """Move the network to the given device (CPU/GPU)."""
        
        self.device = device
        self.net = self.net.to(self.device)