from monai.transforms import (CenterSpatialCropd, Compose, NormalizeIntensityd,
                              RandAffined, RandGaussianNoised,
                              RandSpatialCropd)


def get_train_transforms():
    """
    Compose data transforms for training inspired partially by the Odelia paper's description of their data augmentation pipeline.
    Ref:
    https://arxiv.org/pdf/2506.00474 (paper)
    https://github.com/mueller-franzes/odelia_breast_mri/blob/main/odelia/data/datasets/dataset_3d_odelia.py (code)
    """
    transforms = Compose([
        # Crop to 224×224×32 (random center)
        RandSpatialCropd(
            keys=["image"],
            roi_size=(224, 224, 32),
            random_center=True,
            random_size=False,
        ),
        
        # Z-normalization per channel
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True, # Avoids background dominating stats
            channel_wise=True,
        ),

        # Random 90° rotation
        RandAffined(
            keys=["image"],
            rotate_range=(0.0, 0.0, 0.785),  # 90 degrees in radians
            mode="bilinear",
            prob=0.5,
            padding_mode="border",
        ),
        
        # Random Gaussian noise
        RandGaussianNoised(keys=["image"], prob=0.4, mean=0.0, std=0.1),
    ])
    return transforms

def get_val_transforms():
    """
    Compose data transforms for validation/testing.
    Only includes normalization and center crop - no data augmentation.
    """
    transforms = Compose([
        # Center crop to 224×224×32 (deterministic center)
        CenterSpatialCropd(
            keys=["image"],
            roi_size=(224, 224, 32),
        ),
        
        # Z-normalization per channel (same as training)
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True, # Avoids background dominating stats
            channel_wise=True,
        ),
    ])
    return transforms