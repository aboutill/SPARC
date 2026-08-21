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


def prediction(
        self,
        test_ds, 
        workers=8,
        num_components=1, 
        connectivity=None, 
        labels=False,
    ):
    """Run inference on and save predictions; also computes
    Dice/HD95/ASSD if `labels=True`."""
    
    # Initialize data loaders
    test_loader = DataLoader(
        dataset=test_ds, 
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
    post_transforms_list = [
        Invertd(
            keys="pred",
            transform=test_ds.transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        Activationsd(keys="pred", softmax=softmax, sigmoid=sigmoid),
        AsDiscreted(keys="pred", threshold=threshold, argmax=argmax, to_onehot=to_onehot),
    ]
    if labels:
        post_transforms_list.append(AsDiscreted(keys="label", to_onehot=to_onehot))
    post_transforms_list += [
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
    post_transforms = Compose(post_transforms_list)
        
    # Validation handler
    if labels:
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
        key_val_metric = {
            "val_mean_dice": MeanDice(
                output_transform=from_engine(["pred", "label"]),
                include_background=include_background,
            ),
        }
        additional_metrics = {
            "val_hausdorff_distance": HausdorffDistance(
                output_transform=from_engine(["pred", "label"]),
                percentile=95,
            ),
            "val_surface_distance": SurfaceDistance(
                output_transform=from_engine(["pred", "label"]),
                symmetric=True,
            ),
        }
        
    else:
        val_handlers = None
        key_val_metric = None
        additional_metrics = None
        
    # Validation engine
    evaluator = SupervisedEvaluator(
        device=self.device,
        val_data_loader=test_loader,
        network=self.net,
        inferer=self.inferer,
        postprocessing=post_transforms,
        key_val_metric=key_val_metric,
        additional_metrics=additional_metrics,
        val_handlers=val_handlers   
    )
    
    # Perform evaluation
    evaluator.run()