"""Image augmentation presets."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(preset: str = "mpa", image_size: int = 224):
    """Return train and evaluation transforms for a named augmentation preset."""
    preset = preset.lower()
    normalize = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    if preset == "none":
        train = A.Compose([A.Resize(image_size, image_size), *normalize])
    elif preset == "stnet":
        train = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.02,
                    p=0.5,
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                *normalize,
            ]
        )
    elif preset == "mpa":
        train = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=30,
                    p=0.5,
                ),
                *normalize,
            ]
        )
    else:
        raise ValueError(f"Unknown augmentation preset: {preset}")

    eval_transform = A.Compose([A.Resize(image_size, image_size), *normalize])
    return train, eval_transform
