from monai.utils import set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    SpatialPadd,
    CenterSpatialCropd,
    RandBiasFieldd,
    RandAdjustContrastd,
    Spacingd,
    ToTensord,
    RandZoomd,
)


def init_train_transforms(
    pixdim=(1, 1, 1),
    roi_size=(96, 96, 96),
    prob_contrast=0.2,
    gamma_range=(0.5, 4.5),
    prob_bias_field=0.2,
    bias_field_range=(0, 0.2),
    prob_zoom=0.2,
    zoom_range=(0.9, 1.1),
    labels=False,
):
    """Training augmentation pipeline: canonical RAS reorientation,
    bias field/contrast jitter, spatial padding/cropping, and random zoom."""

    # Initialize seed
    set_determinism()

    # Set keys, mode
    if labels:
        keys = ["image", "label"]
        mode = ["bilinear", "nearest"]
        align_corners = [False, None]
    else:
        keys = "image"
        mode = "bilinear"
        align_corners = False

    # Initialize train transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
            EnsureTyped(keys=keys),
            Orientationd(
                keys=keys,
                axcodes="RAS",
                as_closest_canonical=True,
            ),
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
                keys=keys,
                pixdim=pixdim,
                mode=mode,
            ),
            SpatialPadd(
                keys=keys,
                spatial_size=roi_size,
                mode="constant",
            ),
            CenterSpatialCropd(
                keys=keys,
                roi_size=roi_size,
            ),
            RandZoomd(
                keys=keys,
                prob=prob_zoom,
                min_zoom=zoom_range[0],
                max_zoom=zoom_range[1],
                mode=mode,
                padding_mode="constant",
                align_corners=align_corners,
            ),
            ToTensord(keys=keys),
        ]
    )

    return train_transforms


def init_val_transforms(
    pixdim=(1, 1, 1),
    roi_size=(96, 96, 96),
    labels=False,
):
    """Validation transform pipeline (no augmentation), matching the
    training spacing/ROI size."""

    # Set keys, mode
    if labels:
        keys = ["image", "label"]
        mode = ["bilinear", "nearest"]
    else:
        keys = "image"
        mode = "bilinear"

    # Initialize seed
    set_determinism()

    # Initialize validation transforms
    val_transforms = Compose(
        [
            LoadImaged(
                keys=keys,
                image_only=False,
            ),
            EnsureChannelFirstd(keys=keys),
            EnsureTyped(keys=keys),
            Orientationd(
                keys=keys,
                axcodes="RAS",
                as_closest_canonical=True,
            ),
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
            ),
            Spacingd(
                keys=keys,
                pixdim=pixdim,
                mode=mode,
            ),
            SpatialPadd(
                keys=keys,
                spatial_size=roi_size,
                mode="constant",
            ),
            CenterSpatialCropd(
                keys=keys,
                roi_size=roi_size,
            ),
            ToTensord(
                keys=keys,
            ),
        ]
    )

    return val_transforms


def init_test_transforms(
    pixdim=(1, 1, 1),
    roi_size=(96, 96, 96),
):
    """Inference-only transform pipeline for unlabelled data."""

    # Initialize seed
    set_determinism()

    # Initialize test transforms
    test_transforms = Compose(
        [
            LoadImaged(keys="image", image_only=False),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys="image"),
            Orientationd(
                keys="image",
                axcodes="RAS",
                as_closest_canonical=True,
            ),
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
            ),
            Spacingd(
                keys="image",
                pixdim=pixdim,
                mode="bilinear",
            ),
            SpatialPadd(
                keys="image",
                spatial_size=roi_size,
                mode="constant",
            ),
            CenterSpatialCropd(
                keys="image",
                roi_size=roi_size,
            ),
            ToTensord(
                keys="image",
            ),
        ]
    )

    return test_transforms
