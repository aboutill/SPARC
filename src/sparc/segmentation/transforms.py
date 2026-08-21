from monai.utils import set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped, 
    Compose,
    LoadImaged,  
    RandAffined,
    NormalizeIntensityd,
    RandFlipd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandBiasFieldd,
    RandAdjustContrastd,
    Spacingd,
)


def init_train_transforms(
        pixdim=(1,1,1),
        roi_size=(96,96,96),
        padd_mode="reflect",
        prob_contrast=0.2, 
        gamma_range=(0.5, 4.5), 
        prob_bias_field=0.2, 
        bias_field_range=(0, 0.2), 
        prob_flip=(0.2, 0.2, 0.2),
        prob_affine=0.2,
        translate_range=(0.1, 0.1, 0.1),
        rotate_range=(40, 40, 40),
        scale_range=(0.1, 0.1, 0.1),
    ):
    """Training augmentation transforms: bias field, contrast, spatial
   padding/cropping, flips, and affine jitter."""
    
    # Initialize seed
    set_determinism()

    # Initialize train transforms
    train_transforms_list = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        NormalizeIntensityd(
            keys="image",
            nonzero=True,
        ),
        RandAdjustContrastd(
            keys="image", 
            prob=prob_contrast, 
            gamma=gamma_range,
        ),
        RandBiasFieldd(
            keys="image", 
            prob=prob_bias_field,
            coeff_range=bias_field_range, 
        ),
        Spacingd(
            keys=["image", "label"], 
            pixdim=pixdim, 
            mode=("bilinear", "nearest"),
        ),
        SpatialPadd(
            keys=["image", "label"], 
            spatial_size=roi_size,
            mode=padd_mode,
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size, 
        ),
    ]
    for i, prob in enumerate(prob_flip):
        train_transforms_list.append(
            RandFlipd(
                keys=["image", "label"], 
                prob=prob, 
                spatial_axis=i,
            )
        )
    train_transforms_list.append(
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"), # Actually this is trilinear, cf. https://docs.monai.io/en/stable/transforms.html#randaffined
            prob=prob_affine,
            rotate_range=rotate_range,
            translate_range=translate_range,
            scale_range=scale_range,
        )
    )
    train_transforms = Compose(train_transforms_list)
    
    return train_transforms
    
    
def init_val_transforms(
        pixdim=(1,1,1),
    ):
    """Validation transforms, resampled to the training spacing `pixdim`."""
    
    # Initialize seed
    set_determinism()
    
    # Initialize validation transforms
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
            ),
            Spacingd(
                keys=["image", "label"], 
                pixdim=pixdim, 
                mode=("bilinear", "nearest"),
            ),
        ]
    )
    
    return val_transforms


def init_val_org_transforms(
        pixdim=(1,1,1),
    ):
    """Validation transforms for evaluation in original image space:
    label kept at native spacing, image resampled for inference."""
    
    # Initialize seed
    set_determinism()
    
    # Initialize validation transforms on original grid
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
            ),
            Spacingd(
                keys="image", 
                pixdim=pixdim, 
                mode="bilinear",
            ),
        ]
    )
    
    return val_transforms


def init_test_transforms(
        pixdim=(1,1,1),
    ):
    """Inference-only transform pipeline (no label) for unlabelled data.
    Note: not currently called from trainer.py or tester.py — confirm
    whether this is still needed elsewhere before removing."""
    
    # Initialize seed
    set_determinism()
    
    # Initialize validation transforms
    test_transforms = Compose(
        [
            LoadImaged(keys="image", image_only=False),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys="image"),
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
            ),
            Spacingd(
                keys="image", 
                pixdim=pixdim, 
                mode="bilinear",
            ),
        ]
    )
    
    return test_transforms

