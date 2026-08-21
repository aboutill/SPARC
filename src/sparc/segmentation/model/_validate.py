import os
import torch

from monai.transforms import (
    Compose, 
    AsDiscreted, 
    Activationsd, 
    SaveImaged, 
    Invertd, 
    KeepLargestConnectedComponentd,
)
from monai.handlers import (
    MetricsSaver, 
    MeanDice, 
    HausdorffDistance, 
    SurfaceDistance,
)
from monai.handlers.utils import from_engine
from monai.engines import SupervisedEvaluator
from monai.data import DataLoader


def validate(
        self,
        val_ds, 
        num_components=1, 
        connectivity=None, 
        workers=8,
    ):
    """Load the best checkpoint, run inference, and save
    predictions plus Dice/HD95/ASSD metrics."""
    
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
    
    # Initialise multi-class options
    if self.out_channels == 1:
        sigmoid = True
        softmax = False
        threshold = 0.5
        argmax = False
        to_onehot = None
        include_background = True
        is_onehot = False
    else:
        sigmoid = False
        softmax = True
        threshold = None
        argmax = True
        to_onehot = self.out_channels
        include_background = False
        is_onehot = True
    
    # Initialize post prediction transformation
    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=val_ds.transform,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            Activationsd(keys="pred", softmax=softmax, sigmoid=sigmoid),
            AsDiscreted(keys="pred", threshold=threshold, argmax=argmax, to_onehot=to_onehot),
            AsDiscreted(keys="label", to_onehot=to_onehot),
            KeepLargestConnectedComponentd(
                keys="pred", 
                num_components=num_components,
                connectivity=connectivity,
                is_onehot=is_onehot,
            ),
            SaveImaged(
                keys="pred", 
                meta_keys="pred_meta_dict", 
                folder_layout=self.prediction_layout,
            ),
        ]
    )
    
    # Validation handler
    val_handlers = [
        MetricsSaver(
            save_dir=self.metrics_dir, 
            metrics=[
                "val_mean_dice", 
                "val_hausdorff_distance", 
                "val_surface_distance",
            ],
            metric_details="*",
            batch_transform=from_engine("image_meta_dict"),
            summary_ops="*"
        ),   
    ]
    
    # Validation engine
    evaluator = SupervisedEvaluator(
        device=self.device,
        val_data_loader=val_loader,
        network=self.net,
        inferer=self.inferer,
        postprocessing=post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(
                output_transform=from_engine(["pred", "label"]),
                include_background=include_background,
            ),
        },
        additional_metrics={
            "val_hausdorff_distance": HausdorffDistance(
                output_transform=from_engine(["pred", "label"]),
                percentile=95,
            ),
            "val_surface_distance": SurfaceDistance(
                output_transform=from_engine(["pred", "label"]),
                symmetric=True,
            ),
        },
        val_handlers=val_handlers   
    )
    
    # Perform evaluation
    evaluator.run()